import os
from functools import partial

import botorch
import gpytorch
import numpy as np
import pandas as pd
import torch
import logging

from botorch.acquisition import qExpectedImprovement, ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf, gen_batch_initial_conditions
from botorch.posteriors import GPyTorchPosterior
from botorch.utils import standardize
from botorch.utils.transforms import unnormalize, normalize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.mlls import PredictiveLogLikelihood
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from lolbo.utils.bo_utils.approximate_gp import SingleTaskVariationalGP
from lolbo.utils.bo_utils.base import DenseNetwork
from lolbo.utils.bo_utils.ppgpr import GPModelDKL
from les.utils.utils import array_to_tensor, get_device

# torch.set_default_dtype(torch.float64)
BO_DTYPE = torch.float64
LOGGER = logging.getLogger("BO")


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, out_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module("linear1", torch.nn.Linear(data_dim, 1000))
        self.add_module("relu1", torch.nn.ReLU())
        self.add_module("linear2", torch.nn.Linear(1000, 500))
        self.add_module("relu2", torch.nn.ReLU())
        self.add_module("linear3", torch.nn.Linear(500, 50))
        self.add_module("relu3", torch.nn.ReLU())
        self.add_module("linear4", torch.nn.Linear(50, out_dim))


class SparseGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(
            self.base_covar_module,
            inducing_points=train_x[:500, :].clone(),
            likelihood=likelihood,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, out_dim):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=out_dim)),
        #     num_dims=out_dim, grid_size=100
        # )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=out_dim)
        )
        x_shape = train_x.shape[1]
        if len(train_x.shape) == 3:
            x_shape *= train_x.shape[2]
        self.feature_extractor = LargeFeatureExtractor(x_shape, out_dim)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
        self.num_outputs = 1

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # if len(x.shape) == 3:
        #     x = x.view(x.shape[0], -1)
        # if torch.cuda.is_available():
        #     x = x.cuda()
        dvc, dtype = x.device, x.dtype
        self.feature_extractor = self.feature_extractor.to(dtype=dtype, device=dvc)
        self.scale_to_bounds = self.scale_to_bounds.to(dtype=dtype, device=dvc)
        self.mean_module = self.mean_module.to(dtype=dtype, device=dvc)
        self.covar_module = self.covar_module.to(dtype=dtype, device=dvc)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        mean_x = mean_x.to(dtype=dtype, device=dvc)
        covar_x = covar_x.to(dtype=dtype, device=dvc)
        # LOGGER.debug(f"mean_x: {covar_x}")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        dvc, dtype = X.device, X.dtype

        X = X.to(dtype=dtype, device=dvc)
        self.eval()  # make sure model is in eval mode
        # self.model.eval()
        self.likelihood.eval()
        # print(f"X shape: {X.shape}")
        self.likelihood = self.likelihood.to(dtype=dtype, device=dvc)
        # dists = self(X)
        # dist = dist.to(dtype=dtype, device=dvc)
        dist = self.likelihood(self(X))

        return GPyTorchPosterior(distribution=dist)


def _clip_norm(grad, max_norm):
    norms = torch.linalg.vector_norm(grad, dim=1)
    norms = norms.unsqueeze(1)
    max_norm = max_norm.unsqueeze(1)
    norms = torch.where(norms > max_norm, max_norm / norms, torch.ones_like(norms))
    grad = grad * norms
    return grad


def _scale_norm(grad, scale_norm=1):
    norms = torch.linalg.vector_norm(grad, dim=1)
    norms = norms.unsqueeze(1)
    norms = torch.where(
        norms > 0, torch.ones_like(norms) / scale_norm, torch.ones_like(norms)
    )
    grad = grad * norms
    return grad


def _normalize_grad(grad, target_norm):
    norms = torch.linalg.vector_norm(grad, dim=1)
    norms = norms.unsqueeze(1)
    target_norm = target_norm.unsqueeze(1)
    norms = torch.where(norms > 0, target_norm / norms, torch.ones_like(norms))
    grad = grad * norms
    return grad


def _make_one_norm(grad):
    max_norm = 1
    norms = torch.linalg.vector_norm(grad, dim=1)
    norms = norms.unsqueeze(1)
    norms = torch.where(norms > 0, max_norm / norms, torch.ones_like(norms))
    grad = grad * norms
    return grad


def optimize_ei(
    Z,
    y,
    z_init,
    n_steps=10,
    penalty=None,
    alpha=0.5,
    es_fn=None,
    learning_rate=0.5,
    botorch=False,
):
    if len(y.shape) == 1:
        y = y.view(-1, 1).to(dtype=Z.dtype, device=Z.device)

    q = z_init.shape[0]

    if penalty is None and es_fn is None and botorch:
        gp = SingleTaskGP(Z.clone().detach().double(), y.clone().detach().double())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        best_f = y.max().item()
        ei = qExpectedImprovement(model=gp, best_f=best_f)
        # make gp double
        bounds = torch.stack(
            [torch.ones(Z.shape[1]) * -10, torch.ones(Z.shape[1]) * 10]
        )
        # make z_init double
        z_init = z_init.double()
        # z_init = z_init.detach().clone().double()
        print(f"init gp values bo: {gp(z_init).mean}")

        bounds = bounds.double()
        candidate, acq_value = optimize_acqf(
            ei,
            bounds=bounds,
            q=q,
            raw_samples=q,
            num_restarts=1,
            return_best_only=False,
        )
        # print(acq_value)
        x = candidate.to(dtype=Z.dtype, device=Z.device)
    else:
        gp = SingleTaskGP(Z.clone().detach(), y.clone().detach())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        best_f = y.max().item()
        print(f"init gp values: {gp(z_init).mean}")

        ei = expected_improvement(gp, best_f=-20)
        x = Z[0:5, ...].detach().clone()
        x.requires_grad = True
        # print(x)

        # optimizer = torch.optim.Adam([x], lr=0.1)
        for i in range(n_steps):
            # print(i)

            grad = torch.autograd.grad(ei(x).sum(), x)[0].detach()
            grad = _clip_norm(grad)
            if penalty is not None:
                grad_penalty = penalty(x.detach().clone())
                # clip the norm of the gradient
                grad_penalty = _clip_norm(grad_penalty)
                # grad = _clip_norm(grad)
                grad = alpha * grad_penalty + grad
                grad = _clip_norm(grad)

            grad *= learning_rate
            if es_fn is not None:
                es_vec = es_fn(x.detach().cpu() + grad)
                es_vec = np.array([es_vec] * x.shape[1], dtype=int).T
                es_vec = torch.tensor(es_vec).to(dtype=x.dtype)
                # print("pick")
                grad *= 1 - es_vec
            x = x + grad
            # print(x.mean())

    return x, gp(x).mean.detach()


def expected_improvement(model, best_f):
    def ei(X):
        posterior = model.posterior(X=X)
        samples = torch.squeeze(posterior.rsample(torch.Size([100])))
        ei_values = torch.max(samples - best_f, torch.zeros_like(samples))
        return ei_values.max(dim=1)[0].mean()

    return ei


def train_dkl_model(model, likelihood, train_x, train_y, training_iterations=60):
    model.train()
    likelihood.train()
    device = get_device()
    model.to(device=device, dtype=torch.float32)
    train_x = train_x.to(device=device, dtype=torch.float32)
    train_y = train_y.to(device=device, dtype=torch.float32)

    # # data_loader = torch.utils.data.DataLoader(list(zip(train_x, train_y)), batch_size=64, shuffle=True)
    # tensor_dataset = TensorDataset(train_x, train_y)
    # data_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model.feature_extractor.parameters()},
            {"params": model.covar_module.parameters()},
            {"params": model.mean_module.parameters()},
            {"params": model.likelihood.parameters()},
        ],
        lr=0.01,
    )

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    mll = mll.to(device=device, dtype=torch.float32)

    # def train():
    iterator = tqdm(range(training_iterations), desc="training dkl model")
    for _ in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
    return model, mll


def update_surr_model(
    model, mll, learning_rte, train_z, train_y, n_epochs, test_z=None, test_y=None
):
    model = model.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": learning_rte}], lr=learning_rte
    )

    train_bsz = min(len(train_y), 32)
    train_dataset = TensorDataset(train_z, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    iterator = tqdm(range(n_epochs), desc="training surrogate model")
    for _ in iterator:
        for inputs, scores in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = -mll(output, scores)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # print the norm of the gradient
            # print(torch.linalg.vector_norm(model.feature_extractor[0].weight.grad))

            optimizer.step()

            iterator.set_postfix(loss=loss.item())
    model = model.eval()

    return model


def train_gpdkl_model(
    model,
    likelihood,
    train_x,
    train_y,
    training_iterations=60,
    Z_test=None,
    y_test=None,
):
    # model = model.train()
    # model.likelihood = likelihood.train()
    device = get_device()
    model.to(device=device, dtype=torch.float32)
    train_x = train_x.to(device=device, dtype=torch.float32)
    train_y = train_y.to(device=device, dtype=torch.float32)

    # "Loss" for GPs - the marginal log likelihood
    mll = PredictiveLogLikelihood(likelihood, model, num_data=train_x.size(-2))

    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    mll = mll.to(device=device, dtype=torch.float32)

    # mll = mll.eval()
    # def train():
    model = update_surr_model(model, mll, 0.1, train_x, train_y, training_iterations)
    return model, mll


class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256)):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1), hidden_dims=hidden_dims
        ).to(inducing_points.device)
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1  # must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor
        self.device = inducing_points.device

    def forward(self, x):
        # x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        # self.model.eval()
        self.likelihood.eval()
        dist = self.likelihood(self(X))

        return GPyTorchPosterior(distribution=dist)


if __name__ == "__main__":
    # Example usage:
    # Set random seed for reproducibility
    torch.manual_seed(42)
    dim = 256

    def _dgp(X):
        return torch.sum(torch.sin(X), dim=1)

    # pth = os.path.join("results", "bb_opt", "selfies_test_save_large")
    pth = os.path.join("results", "bb_opt", "selfies_zale")
    Z_mat = pd.read_csv(os.path.join(pth, "Z.csv"), header=None).values
    Z = array_to_tensor(Z_mat, device=get_device())
    y_vec = pd.read_csv(os.path.join(pth, "y.csv"), header=None).values
    y = array_to_tensor(y_vec, device=get_device())[:, 1]
    print(y[0:5])

    # remove rows with na in either Z or y
    na_idx = torch.isnan(y) | torch.isnan(Z).any(dim=1)
    Z = Z[~na_idx, :]
    y = y[~na_idx]
    y = (y - y.mean()) / y.std()
    print(f"Z shape {Z.shape}, y shape {y.shape}")
    # take 1000 data point from Z to be Z_fit
    Z_fit = Z[:2500, :]
    y_fit = y[:2500]
    Z_test = Z[1000:2200, :]
    y_test = y[1000:2200]
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # model = GPRegressionModel(Z_fit, y_fit, likelihood, 4)
    # model, mll = train_dkl_model(model, likelihood, Z_fit, y_fit,
    #                                 training_iterations=300)

    n_pts = min(Z_fit.shape[0], 1024)
    Z_fit = Z_fit.to(dtype=BO_DTYPE, device=get_device())
    model = GPModelDKL(Z_fit[:n_pts, :], likelihood=likelihood)
    model, mll = train_gpdkl_model(
        model, likelihood, Z, y, training_iterations=200, Z_test=Z_test, y_test=y_test
    )

    model.eval()
    likelihood.eval()
    y_hat = model.posterior(Z_test).mean.detach()
    # print(y_hat)
    # plot y vs y_hat
    import matplotlib.pyplot as plt

    y_hat = y_hat.cpu().numpy()
    y_test = y_test.cpu().numpy()
    plt.scatter(y_test, y_hat)
    # add x y titles
    x_title = "y"
    y_title = "y_hat"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig("y_vs_y_hat.png")

    # model = GPRegressionModel(X_train, y_train, likelihood, 2)
    # model, mll = train_dkl_model(model,likelihood, X_train, y_train)
    # model.eval()
    # likelihood.eval()
    # # ei = qExpectedImprovement(model, best_f=-20)

    # # mll = ExactMarginalLogLikelihood(likelihood, model)
    # # fit_gpytorch_mll(mll)

    # ei = expected_improvement(model, best_f=-20)
    # # optimize acquisition function using pytorch
    # x = Variable(torch.randn(1, dim), requires_grad=True)
    # # x_2 = x.clone()
    # # print(_dgp(x))
    # print(model(x).mean)
    # optimizer = torch.optim.Adagrad([x], lr=1)
    # for i in range(30):
    #     optimizer.zero_grad()
    #     loss = -ei(x)
    #     loss.backward()
    #     optimizer.step()

    # print(model(x).mean)

    # covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    # # X_test = torch.randn(5, dim)
    # # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # # model = GPModelDKL(X_train, likelihood=likelihood)
    # # mll = PredictiveLogLikelihood(model.likelihood, model, num_data=X.shape[0])
    # # Create a Gaussian Process model
    # gp_model = SingleTaskGP(X_train, y_train.unsqueeze(-1), covar_module=covar_module)
    # mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    # # mll.to(self.Z_norm.double())
    # fit_gpytorch_mll(mll)

    # # Forward pass through the GP model
    # # mean, variance = gp_model(X_test)
    # ei = expected_improvement(gp_model, best_f=-20)
    # # optimize acquisition function using pytorch
    # x = Variable(torch.randn(1, dim), requires_grad=True)
    # # x_2 = x.clone()
    # # print(_dgp(x))
    # print(gp_model(x).mean)
    # optimizer = torch.optim.Adagrad([x], lr=1)
    # for i in range(30):
    #     optimizer.zero_grad()
    #     loss = -gp_model(x).mean
    #     loss.backward()
    #     optimizer.step()

    # # print(_dgp(x))
    # print(gp_model(x).mean)

    # # do the same with botorch
    # # define the GP model
    # gp = SingleTaskGP(X_train, y_train.unsqueeze(-1), covar_module=covar_module)
    # mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    # fit_gpytorch_mll(mll)
    # # define the acquisition function
    # ei = ExpectedImprovement(gp, best_f=-20)
    # # optimize acquisition function using botorch
    # x = Variable(torch.randn(1, dim), requires_grad=True)

    # print(gp_model(x).mean)

    # Z_new, y_new = optimize_acqf(
    #     ei, bounds=torch.stack([torch.ones(dim) * -100, torch.ones(dim) * 100]),
    #     q=1, batch_initial_conditions=x, num_restarts=1, return_best_only=True,
    #     raw_samples=1)
    # print(gp_model(Z_new.unsqueeze(0)).mean)
