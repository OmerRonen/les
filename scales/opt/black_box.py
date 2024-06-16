import os
import torch
import pickle
import gpytorch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch.nn.functional as F

# from botorch import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood


from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
try:
    from botorch.acquisition.logei import qLogExpectedImprovement
except ImportError:
    pass
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from botorch.utils.transforms import normalize, unnormalize

from scales import LOGGER
from scales.opt.gp import GPRegressionModel, _normalize_grad, train_dkl_model
from scales.utils.scales import calculate_scales
from scales.utils.opt import get_black_box_function
from scales.utils.utils import get_mu_sigma


class BlackBoxOptimizer:
    def __init__(self, Z, y, vae, es_rule=None, penalty=None, botorch=True, black_box_function=None,
                 pth=None, dataset=None, bound=0.5, normalize=True, objective=None):
        self.vae = vae
        d = Z.shape[1]
        self.device = vae.device
        self.dtype = vae.dtype
        self.bounds = torch.tensor([[-bound * 0] * d, [bound] * d], device=self.device, dtype=self.dtype)
        self.es_rule = es_rule
        self.penalty = penalty
        self.botorch = botorch
        self.initialized = False
        self.black_box_function = black_box_function
        self.pth = pth
        self._bb_obj = objective
        self._normalize = normalize
        # create pth if not exists recursively
        if self.pth is not None:
            if not os.path.exists(self.pth):
                os.makedirs(self.pth)
        self.expr_path = os.path.dirname(os.path.dirname(self.pth))
        self.dataset = dataset
        self.Z = Z.to(device=self.device, dtype=self.dtype)
        self._z_max, self._z_min = 3, -3#self.Z.max().detach(), self.Z.min().detach()
        LOGGER.debug(f"z max: {self._z_max}, z min: {self._z_min}")
        self.y = y.to(device=self.device, dtype=self.dtype)
        self.Z_norm = self.normalize(self.Z)
        LOGGER.debug(f"Z average: {self.Z_norm.mean()}, Z average: {self.Z.mean()}")
        self._top_10 = [-np.inf] * 10
        self._top_10_objects = [None] * 10

    @property
    def top_10(self):
        return self._top_10


    def update_top_10(self, values, objects):
        # add sort values and add to top 10 if they are better
        # remove nan values from values
        values = [v for v in values if not np.isnan(v)]
        all_values_list = self._top_10 + values
        all_objects_list = self._top_10_objects + objects

        self._top_10 = sorted(self._top_10 + values, reverse=True)[:10]
        # do the same for objects
        self._top_10_objects = [o for o, v in sorted(zip(all_objects_list, all_values_list), key=lambda pair: pair[1], reverse=True)][:10]
        # sort objects by _top_10



    def normalize(self, z):
        if not self._normalize:
            return z
        d = z.shape[1]
        bounds = torch.tensor([[self._z_min] * d, [self._z_max] * d],
                              device=self.device, dtype=self.dtype)
        z_norm =  normalize(z, bounds).to(device=self.device, dtype=self.dtype)
        return z_norm

    def unnormalize(self, z):
        if not self._normalize:
            return z
        
        d = z.shape[1]
        bounds = torch.tensor([[self._z_min] * d, [self._z_max] * d],
                              device=self.device, dtype=self.dtype)
        z_unnorm = unnormalize(z, bounds).to(device=self.device, dtype=self.dtype)
        return z_unnorm
    
    def surrogate_model(self, z):
        raise NotImplementedError

    def objective(self, z):
        raise NotImplementedError

    def test_surrogate_model(self, z_test, y_test):
        z_test_norm = self.normalize(z_test) if self._normalize else z_test
        y_hat_numpy = torch.squeeze(self.surrogate_model(z_test_norm)).detach().cpu().numpy()
        y_test_numpy = torch.squeeze(y_test).detach().cpu().numpy()
        corr, mse = np.corrcoef(y_hat_numpy, y_test_numpy)[0, 1], np.sqrt(np.mean((y_hat_numpy - y_test_numpy) ** 2))
        y_test_std = y_test_numpy.std()
        # LOGGER.info(f"corr: {corr}, mse: {mse}, std: {y_test_std}")
        # plot y_hat vs y_test
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(y_hat_numpy, y_test_numpy)
        ax.set_xlabel("y_hat")
        ax.set_ylabel("y_test")
        ax.set_title(f"corr: {float(np.round(corr, 2))}, "
                     f"mse: {float(np.round(mse, 2))}, "
                     f"std: {float(np.round(y_test_std, 2))}")
        plt.savefig(os.path.join(self.expr_path, "y_hat_vs_y_test.png"))
        plt.close()
        # return loss
    
    def _get_new_batch(self, z_init, alpha, learning_rate):
        # remove self.Z self.y and seld.Z_norm from memory
        z = z_init.detach().clone().to(device=self.device, dtype=self.dtype)

        for _ in range(10):
            z_grad = z.detach().clone()
            z_grad_unnorm = self.unnormalize(z_grad.clone().to(device=self.device, dtype=self.dtype))
            z_grad = z_grad.clone().to(device=self.device, dtype=self.dtype).requires_grad_(True)
            if torch.isnan(z_grad).any():
                LOGGER.info("z_grad has nan values")
            _b_s = 1
            n_batches = z_grad.shape[0] // _b_s + z_grad.shape[0] % _b_s
            for i in range(n_batches):
                z_grad_i = z_grad[i * _b_s: (i + 1) * _b_s, ...]
                grad_i = torch.autograd.grad(self.objective(z_grad_i), z_grad_i)[0].detach()
                if i == 0:
                    grad = grad_i
                else:
                    grad = torch.cat([grad, grad_i], dim=0)

            grad = _normalize_grad(grad, torch.ones_like(grad).mean(dim=1))
            grad_norm = torch.linalg.vector_norm(grad, dim=1).to(device=self.device, dtype=self.dtype)
            grad = grad.to(device=self.device, dtype=self.dtype)

            if self.penalty is not None and alpha > 0:
                LOGGER.debug("using penalty")
                # grad_penalty = self.penalty(z_grad.to(device=self.device, dtype=self.dtype))
                grad_penalty = self.penalty(z_grad_unnorm)
                grad_penalty = grad_penalty.to(device=self.device, dtype=self.dtype)
                # if self._normalize:
                grad_penalty = self.normalize(grad_penalty)
                # clip grad norm to be at least 0.1
                # grad_penalty = _clip_norm(grad_penalty, grad_norm)
                grad_penalty = _normalize_grad(grad_penalty,  torch.ones_like(grad_penalty).mean(dim=1))
                # grad_penalty = _clip_norm(grad_penalty, torch.ones_like(grad_norm) * 0.5)

                grad_penalty = grad_penalty.to(device=self.device, dtype=self.dtype)
                grad = alpha * grad_penalty + grad
                grad = _normalize_grad(grad, grad_norm)

            grad = grad * learning_rate
            if self.es_rule is not None:
                z_proposed = z.clone().cpu() + grad.cpu()
                z_proposed = z_proposed.to(device=self.device, dtype=self.dtype)
                # print(z_proposed)
                # self.es_rule(z_init)
                es_vec = self.es_rule(self.unnormalize(z_proposed))
                es_vec = np.array([es_vec] * z.shape[1], dtype=int).T
                es_vec = torch.tensor(es_vec).to(dtype=z.dtype)
                # print(f"es vec: {es_vec.detach().cpu().numpy()}")

                grad *= (es_vec).to(device=self.device, dtype=self.dtype)
            z = z + grad
        candidate = z.detach().clone()
        # LOGGER.debug(f"opt gp values bo: {self.surrogate_model(z)}")

        return candidate

    def get_new_batch(self, z_init, alpha=0.1, learning_rate=0.1):
        raise NotImplementedError

    def evaluate_solution(self, z, tau):
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        self.vae.decoder.to(dtype=self.vae.dtype, device=self.vae.device)
        z = z.to(device=self.device, dtype=self.dtype)
        x_decoded = self.vae.decode(z.detach().clone())
        if z.shape[0] == 1:
            x_decoded = x_decoded.unsqueeze(0)
        if self.dataset in ["selfies", "smiles"]:
            x_decoded = x_decoded.argmax(dim=-1).long()
            if len(x_decoded.shape) == 1:
                x_decoded = x_decoded.unsqueeze(0)
        if self.dataset == "smiles":
            x_decoded = F.one_hot(x_decoded, num_classes=self.vae.decoder.vocab_size).float()
        bb_func = get_black_box_function(self.dataset, norm=False, objective=self._bb_obj)
        bb_vals = bb_func(x_decoded).detach().cpu().numpy()
        objective_vals = np.zeros(len(z))
        quality = self.vae.get_decoding_quality(z)
        quality_idx = quality == 1
        self.update_top_10(bb_vals[quality_idx].tolist(), x_decoded[quality_idx].tolist())
        # LOGGER.info(f"top 10: {self.top_10}")
        mu, sigma = get_mu_sigma(self.vae)
        if self.dataset == "selfies":
            bs = 4
        else:
            bs = 32
        try:
            density = calculate_scales(model=self.vae, X=z.clone().to(device=self.device, dtype=self.dtype),
                                        mu=mu, sigma=sigma, batch_size=bs, tau=tau).detach().cpu().numpy()
        except Exception as e:
            # create an array for na's
            density = np.zeros(len(z))
            # make all values na
            density[...] = np.nan


        # LOGGER.info(f"bb values: {bb_vals * quality}")
        return {"bb_vals": bb_vals, "objective_vals": objective_vals, "quality": quality,
                "z": z.detach().cpu().numpy(), "x": x_decoded.detach().cpu().numpy(), "density": density}


    def _get_z_batch(self, n_batch, seed):
        np.random.seed(seed)
        return self.Z[np.random.choice(np.arange(len(self.Z)), n_batch, replace=False), ...]


class BlackBoxOptimizerBO(BlackBoxOptimizer):

    def _init_surr_model(self, Z, y):
        # if self.initialized:
        #     return self._model
        if self.dataset in ["selfies"]:
            weights_file = os.path.join("trained_models", "selfies", f"dkl_weights_{self._bb_obj}_new.pt")
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            
            # model =  GPModelDKL(self.Z[:10, :], likelihood=likelihood )
            model = GPRegressionModel(Z, y, likelihood, 4)
            if os.path.isfile(weights_file):
                model.load_state_dict(torch.load(weights_file))
            else:
                # model, _ =  train_gpdkl_model(model, likelihood, self.Z, self.y, training_iterations=60)
                model, _ = train_dkl_model(model, likelihood, Z, y,
                                training_iterations=200)
                # save model
                torch.save(model.state_dict(), weights_file)
            model.eval()
            likelihood.eval()
            self._model = model
            self.initialized = True
        return model

    def surrogate_model(self, z):
        if not hasattr(self, "gp"):
            self.gp = self.get_fitted_model()
        self.gp.to(device=self.device, dtype=self.dtype)
        z = z.to(device=self.device, dtype=self.dtype)
        return self.gp(z).mean.to(device=self.device, dtype=self.dtype)

    def objective(self, z):
        if not hasattr(self, "gp"):
            self.gp = self.get_fitted_model()
        
        gp = self.gp.to(device=self.device, dtype=self.dtype)
        # gp.mean_cache = gp.mean_cache.to(device=self.device, dtype=self.dtype)
        z = z.to(device=self.device, dtype=self.dtype)
        try:
            ei = qLogExpectedImprovement(model=gp, best_f=self.best_f).to(device=self.device, dtype=self.dtype)
        except NameError:
            ei = qExpectedImprovement(model=gp, best_f=self.best_f).to(device=self.device, dtype=self.dtype)
        return ei(z)
    
    def objective_cpu(self, z):
        if not hasattr(self, "gp"):
            self.gp = self.get_fitted_model(cpu=True)
        gp = self.gp.cpu()
        best_f = self.best_f.cpu()
        try:
            ei = qLogExpectedImprovement(model=gp, best_f=best_f)
        except NameError:
            ei = qExpectedImprovement(model=gp, best_f=best_f)
        # ei = qExpectedImprovement(model=self.gp, best_f=0.1)
        device = "cpu"
        z = z.to(device=device, dtype=self.dtype)
        return ei(z).to(device=device, dtype=self.dtype)

    def get_fitted_model(self, cpu=False):
        # initialize and fit model

        Z_fit = self.Z_norm.double() if self._normalize else self.Z.double()
        y_fit = self.y_norm.double()
        # remove nas
        na_idx_both = torch.isnan(Z_fit).any(dim=1) | torch.isnan(y_fit)
        Z_fit = Z_fit[~na_idx_both, ...]
        y_fit = y_fit[~na_idx_both, ...]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        dvc = "cpu" if cpu else self.device
        Z_fit, y_fit = Z_fit.to(device=dvc, dtype=self.dtype), y_fit.to(device=dvc, dtype=self.dtype)
        likelihood = likelihood.to(device=dvc, dtype=self.dtype)
        if self.dataset in ["selfies"]:
            model = self._init_surr_model(Z = Z_fit, y = y_fit)
            model, mll = train_dkl_model(model, likelihood, Z_fit, y_fit,
                                         training_iterations=10)
            # model.eval()
            # likelihood.eval()
            model.eval()
            likelihood.eval()
            model = model.to(device=self.device, dtype=self.dtype)
            # mll = mll.to(device=self.device, dtype=self.dtype)
            return model.double()
        # likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if self.dataset == "expressions":
            model = SingleTaskGP(
                train_X=Z_fit.float(),
                train_Y=y_fit.unsqueeze(-1).float(),
                # covar_module=covar_module,
                likelihood=likelihood
            )
        elif self.dataset == "smiles":
            model = SingleTaskGP(
                train_X=Z_fit.float(),
                train_Y=y_fit.unsqueeze(-1).float(),
                # covar_module=covar_module,
                likelihood=likelihood
            )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(Z_fit)
        fit_gpytorch_mll(mll)
        model.eval()
        model = model.to(device=dvc, dtype=self.dtype)
        # likelihood.eval()
        return model

    @property
    def y_norm(self):
        # return self.y
        return standardize(self.y)
        # return y_norm

    @property
    def best_f(self):
        # encourage exploration
        # prdict the values on the training set
        # return self.gp(self.Z_norm).mean.max()
        return self.y_norm.max()

    def _get_new_batch_botorch(self, z_init):

        # LOGGER.info(f"init gp values bo: {gp(z_init).mean}")
        q = z_init.shape[0]
    
        try:
            z_new, _ = optimize_acqf(
                self.objective, bounds=self.bounds, q=q, batch_initial_conditions=z_init,
                num_restarts=10, return_best_only=False, sequential=True)
        except Exception:
            # move everything to cpu
            z_init = z_init.cpu()
            bounds = self.bounds.cpu()
            z_new, _ = optimize_acqf(self.objective_cpu, bounds=bounds, q=q, batch_initial_conditions=z_init,
                                     num_restarts=10, return_best_only=False, sequential=False)
        # LOGGER.info(f"opt gp values bo: {gp(z_new).mean}")
        # LOGGER.info(f"z_new: {z_new.shape}")
        return torch.squeeze(z_new).to(device=self.device, dtype=self.dtype)

    def get_new_batch(self, z_init, alpha=0.1, learning_rate=0.1):
        X_init = self.vae.decode(z_init).detach()
        if self.dataset in ["selfies", "smiles"]:
            X_init = X_init.argmax(dim=-1).long()
        if self.dataset == "smiles":
            # make into one hot
            X_init = F.one_hot(X_init, num_classes=self.vae.decoder.vocab_size).float()
        bb_init = self.black_box_function(X_init).detach().cpu().numpy()
        if self._normalize:
            z_init = self.normalize(z_init)
        if self.botorch:
            LOGGER.debug("using botorch")
            z_new = self._get_new_batch_botorch(z_init.double())
            if len(z_new.shape) == 1:
                z_new = z_new.unsqueeze(0)
        else:
            LOGGER.debug("not using botorch")
            z_new = self._get_new_batch(z_init, alpha=alpha, learning_rate=learning_rate)
        
        z_new = self.unnormalize(z_new)
        # print(f"mean z_new: {z_new.mean()}")
        z_init = self.unnormalize(z_init)
        X_end = self.vae.decode(z_new).detach()
        if self.dataset in ["selfies", "smiles"]:
            X_end = X_end.argmax(dim=-1).long()
        if self.dataset == "smiles":
            # make into one hot
            X_end = F.one_hot(X_end, num_classes=self.vae.decoder.vocab_size).float()
        bb_end = self.black_box_function(X_end).detach().cpu().numpy()
        bb_diff = bb_end - bb_init
        # objective_diff = self.objective(z_new) - self.objective(z_init)
        # objective_diff = objective_diff.detach().cpu().numpy()
        diff_norm = torch.linalg.vector_norm(z_new - z_init, dim=1)
        LOGGER.debug(f"average improvement: {np.nanmean(bb_diff)}, max improvement: {np.nanmax(bb_diff)}")
        # LOGGER.debug(f"average objective improvement: {np.nanmean(objective_diff)}")
        LOGGER.debug(f"diff norm mean: {diff_norm.mean()}")
        return z_new, bb_diff

    def update_gp_data(self, Z_new, y_new):
        y_new = y_new.to(device=self.device, dtype=self.dtype)
        Z_new = Z_new.to(device=self.device, dtype=self.dtype)
        Z_new = torch.squeeze(Z_new)
        if len(Z_new.shape) == 1:
            Z_new = Z_new.unsqueeze(0)

        self.y = torch.cat([self.y, y_new], dim=0)
        self.Z = torch.cat([self.Z, Z_new], dim=0)
        self.Z_norm = self.normalize(self.Z)
        # self.gp = self.get_fitted_model()

    def optimize(self, n_batch, n_steps, alpha, learning_rate, tau):
        for i in range(n_steps):
            self.gp = self.get_fitted_model()

            sol_file_name = os.path.join(self.pth, f"sol_{i}.pickle")

            if os.path.isfile(sol_file_name):
                s_e = pickle.load(open(sol_file_name, "rb"))
                y_new = torch.tensor(s_e["bb_vals"], device=self.device, dtype=self.dtype)
                nan_idx = s_e['quality'] == 0
                Z_new = torch.tensor(s_e["z"], device=self.device, dtype=self.dtype)
                sol_eval = self.evaluate_solution(Z_new, tau)
                # sol_eval["bb_diff"] = s_e["bb_vals"]

            else:
                # print(self.dataset)
                Z_init = self._get_z_batch(n_batch, i)
                # print(calculate_uncertainty(model = self.vae, X=Z_init))
                Z_new, bb_diff = self.get_new_batch(Z_init, alpha=alpha, learning_rate=learning_rate)
                Z_new = torch.tensor(Z_new.detach().cpu().numpy(), device=self.device, dtype=self.dtype)
                self.vae.decoder.eval()
                X_new_decoded = self.vae.decode(Z_new).detach()
                if self.dataset in ["selfies", "smiles"]:
                    X_new_decoded = X_new_decoded.argmax(dim=-1).long()
                if self.dataset == "smiles":
                    # make into one hot
                    X_new_decoded = F.one_hot(X_new_decoded, num_classes=self.vae.decoder.vocab_size).float()
                y_new = self.black_box_function(X_new_decoded).to(device=self.device, dtype=self.dtype)
                sol_eval = self.evaluate_solution(Z_new,tau)
                nan_idx = sol_eval['quality'] == 0
                sol_eval["bb_diff"] = bb_diff

            for k, v in sol_eval.items():
                if torch.is_tensor(v):
                    sol_eval[k] = v.detach().cpu().numpy()
            # save sol eval to file
            if self.pth is not None:
                if not os.path.exists(self.pth):
                    os.makedirs(self.pth)
                # save to pickle file
                pickle.dump(sol_eval, open(sol_file_name, "wb"))

            # remove nan idx
            if self.dataset != "selfies":
                y_new[nan_idx] = self.y.min()

                # Z_new = Z_new[~nan_idx, ...]
                # y_new = y_new[~nan_idx, ...]
            else:
                pass
                # y_new[nan_idx] = self.y.min()
            if len(Z_new) > 0:
                self.update_gp_data(Z_new, y_new)
    
    def results(self, steps, k):
        solution_files = [os.path.join(self.pth, f"sol_{i}.pickle") for i in range(steps)]
        # read_all_files
        solutions = [pickle.load(open(s, "rb")) for s in solution_files]
        all_quality = [s["quality"] for s in solutions]
        all_quality = np.concatenate(all_quality)
        nan_idx = all_quality == 0


        all_bb_vals = [s["bb_vals"] for s in solutions]
        all_bb_vals = np.concatenate(all_bb_vals)
        all_bb_vals = all_bb_vals[~nan_idx]

        top_k_idx = np.argsort(all_bb_vals)[-k:]
        top_k = all_bb_vals[top_k_idx]
        
        all_x = [s["x"] for s in solutions]
        all_x = np.concatenate(all_x)
        all_x = all_x[~nan_idx]
        top_k_x = all_x[top_k_idx]

        average_quality = all_quality.mean()
        return {"top_10": top_k, "top_10_x": top_k_x, "average_quality": average_quality}
        
        