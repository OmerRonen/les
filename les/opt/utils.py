import logging

import gpytorch
import torch

from botorch.posteriors import GPyTorchPosterior
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from lolbo.utils.bo_utils.base import DenseNetwork
from lolbo.utils.bo_utils.ppgpr import GPModelDKL
from les.utils.utils import get_device

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


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, out_dim):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
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
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        dvc, dtype = X.device, X.dtype

        X = X.to(dtype=dtype, device=dvc)
        self.eval()  # make sure model is in eval mode
        self.likelihood.eval()
        self.likelihood = self.likelihood.to(dtype=dtype, device=dvc)

        dist = self.likelihood(self(X))

        return GPyTorchPosterior(distribution=dist)


def normalize_grad(grad, target_norm):
    norms = torch.linalg.vector_norm(grad, dim=1)
    norms = norms.unsqueeze(1)
    target_norm = target_norm.unsqueeze(1)
    norms = torch.where(norms > 0, target_norm / norms, torch.ones_like(norms))
    grad = grad * norms
    return grad


def train_dkl_model(model, likelihood, train_x, train_y, training_iterations=60):
    model.train()
    likelihood.train()
    device = get_device()
    model.to(device=device, dtype=torch.float32)
    train_x = train_x.to(device=device, dtype=torch.float32)
    train_y = train_y.to(device=device, dtype=torch.float32)

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
            optimizer.step()
            iterator.set_postfix(loss=loss.item())
    model = model.eval()

    return model


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
