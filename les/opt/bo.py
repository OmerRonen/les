from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import gpytorch
import numpy as np
import torch
from fire import Fire
from tqdm import tqdm

from les import LOGGER
from les.nets.template import VAE
from les.opt.turbo import TurboState, update_state
from les.opt.utils import GPRegressionModel, train_dkl_model
from les.utils.opt_utils import get_objective


from .optimizer import AcquisitionOptimizer, OptimizerSpec
from .initilizer import Initializer, InitializerSpec, InitializerTurboConfig
from .turbo import TurboConfig

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll

from gpytorch import ExactMarginalLogLikelihood


def normalize(z: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return (z - lb) / (ub - lb)


def unnormalize(z: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return z * (ub - lb) + lb


def fit_mll(
    model: gpytorch.models.ExactGP,
    mll: gpytorch.mlls.ExactMarginalLogLikelihood,
    Z: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 100,
):
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
        Z = Z.cuda()
        y = y.cuda()
        mll = mll.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(n_epochs):
        # clear gradients
        optimizer.zero_grad()
        # forward pass through the model to obtain the output MultivariateNormal
        output = model(Z)
        # Compute negative marginal log likelihood
        loss = -mll(output, y)
        # back prop gradients
        loss.backward()
        optimizer.step()
    model.eval()
    if torch.cuda.is_available():
        model = model.cpu()
        Z = Z.cpu()
        y = y.cpu()
    return model


class BOTracker:
    def __init__(self):
        self.solutions_valid = []
        self.solutions_all = []
        self.is_valid = []
        self.n_rounds = 0

    def add_solution(self, solution: np.ndarray, is_valid: np.ndarray):
        # remove nans from solution only
        solution = solution[~np.isnan(solution)]
        self.solutions_all.append(solution)
        self.solutions_valid.append(solution[is_valid == 1])
        self.is_valid.append(is_valid)
        self.n_rounds += 1

    def summarize(self):
        # for each round report the cumulative average of the top 20 solutions and the best solution. report the pct valid over all rounds
        summary = {
            "top_20": [],
            "best": [],
            "pct_valid": float(np.mean(np.concatenate(self.is_valid))),
        }
        for i in range(self.n_rounds):
            all_valid_solutions = np.concatenate(self.solutions_valid[: i + 1])
            # sort by objective value
            all_valid_solutions = all_valid_solutions[
                np.argsort(-1 * all_valid_solutions)
            ]
            if all_valid_solutions.shape[0] > 20:
                top_20_solutions = np.mean(all_valid_solutions[:20], axis=0)
            else:
                top_20_solutions = np.nan
            if all_valid_solutions.shape[0] > 1:
                best_solution = all_valid_solutions[0]
            else:
                best_solution = np.nan
            summary["top_20"].append(float(top_20_solutions))
            summary["best"].append(float(best_solution))
        return summary


@dataclass
class BOConfig:
    initializer_spec: InitializerSpec
    optimizer_spec: OptimizerSpec
    blackbox_function: Callable
    turbo_config: TurboConfig
    vae: VAE
    n_batch: int
    n_steps: int
    dataset_name: str
    normalize: bool = False
    z_bounds: Optional[Tuple[float, float]] = None
    use_dkl: bool = False
    use_turbo: bool = False


class BayesianOptimizer:
    def __init__(self, bo_config: BOConfig):
        self.bo_config = bo_config
        self._Z = bo_config.initializer_spec.Z
        self.y = bo_config.initializer_spec.y
        self._normalize_z = bo_config.normalize
        self._z_min = bo_config.z_bounds[0]
        self._z_max = bo_config.z_bounds[1]
        self.tracker = BOTracker()
        self.use_turbo = bo_config.use_turbo
        self.turbo_state = TurboState(
            dim=self._Z.shape[1],
            batch_size=self.bo_config.n_batch,
            best_value=self.y.max(),
            length=self.bo_config.turbo_config.initial_length,
            failure_tolerance=self.bo_config.turbo_config.failure_tolerance,
            success_tolerance=self.bo_config.turbo_config.success_tolerance,
        )

    @property
    def Z(self):
        return self.normalize(self._Z)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        if self._normalize_z:
            LOGGER.info("Normalizing z")
            lb = torch.tensor(
                [self._z_min] * z.shape[1], device=z.device, dtype=z.dtype
            )
            ub = torch.tensor(
                [self._z_max] * z.shape[1], device=z.device, dtype=z.dtype
            )
            return normalize(z, lb, ub)
        return z

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        if self._normalize_z:
            lb = torch.tensor(
                [self._z_min] * z.shape[1], device=z.device, dtype=z.dtype
            )
            ub = torch.tensor(
                [self._z_max] * z.shape[1], device=z.device, dtype=z.dtype
            )
            return unnormalize(z, lb, ub)
        return z

    def _init_surr_model(self, Z: torch.Tensor, y: torch.Tensor, train: bool = True):
        if self.bo_config.use_dkl:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self._dkl_dim = 12
            model = GPRegressionModel(Z, y, likelihood, self._dkl_dim)
            if train:
                model, _ = train_dkl_model(
                    model, likelihood, Z, y, training_iterations=200
                )
            model.eval()
            likelihood.eval()
            self._model = model
            self.initialized = True
        return model

    def acq_function(self, z: torch.Tensor) -> torch.Tensor:
        best_f = torch.squeeze(self.y).max()
        # make sure gp and z are on the same device
        gp = self.gp.to(z.device)
        qei = qLogExpectedImprovement(model=gp, best_f=best_f)
        return qei(z)

    def fit_gp(self):
        Z_fit = self.Z
        y_fit = self.y

        # remove nas
        na_idx_both = torch.squeeze(torch.isnan(y_fit))
        Z_fit = Z_fit[~na_idx_both, ...]
        y_fit = torch.squeeze(y_fit[~na_idx_both, ...])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if self.bo_config.use_dkl:
            model = self._init_surr_model(
                Z=Z_fit, y=y_fit, train=hasattr(self, "_gp_state_dict")
            )
            if hasattr(self, "_gp_state_dict"):
                model.load_state_dict(self._gp_state_dict)
                return model.double()
            model.eval()
            likelihood.eval()
            self.gp = model
            self._gp_state_dict = model.state_dict()
            return model.double()
        else:
            model = SingleTaskGP(
                train_X=Z_fit.double(),
                train_Y=y_fit.double(),
            )
        if hasattr(self, "_gp_state_dict"):
            model.load_state_dict(self._gp_state_dict)
            return model
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(Z_fit)
        model = fit_mll(model, mll, Z_fit, torch.squeeze(y_fit))
        # model = fit_gpytorch_mll(mll)
        self._gp_state_dict = model.state_dict()
        self.gp = model

    def _get_turbo_bounds(self):
        z = self.Z
        x_center = z[self.y.argmax(), :].clone()
        weights = self.gp.covar_module.base_kernel.lengthscale.squeeze().detach()
        if self.bo_config.use_dkl:
            weights = torch.ones(z.shape[1])
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        weights = weights
        x_center = x_center
        tr_lb = torch.clamp(
            x_center - weights * self.turbo_state.length / 2.0, 0.0, 1.0
        )
        tr_ub = torch.clamp(
            x_center + weights * self.turbo_state.length / 2.0, 0.0, 1.0
        )
        bounds = torch.stack([tr_lb, tr_ub], dim=0)
        return bounds

    def optimize(self):
        self.fit_gp()

        if self.use_turbo:
            init_turbo_config = InitializerTurboConfig(
                acq_function=self.acq_function,
                turbo_bounds=self._get_turbo_bounds(),
            )
            self.bo_config.initializer_spec.turbo_config = init_turbo_config

        initializer = Initializer(self.bo_config.initializer_spec)
        optimizer = AcquisitionOptimizer(self.bo_config.optimizer_spec)

        if self.bo_config.optimizer_spec.method == "lbfgs":
            if self.use_turbo:
                bounds = self._get_turbo_bounds()
            else:
                d = self._Z.shape[1]
                bounds = (
                    torch.tensor([self.bo_config.z_bounds[0]] * d, device=self.device),
                    torch.tensor([self.bo_config.z_bounds[1]] * d, device=self.device),
                )
            optimizer.optimizer_spec.bounds = bounds
        initializer.Z = self.Z
        for step in tqdm(range(self.bo_config.n_steps), desc="BO steps"):
            z_init = initializer.get_z_init(self.bo_config.n_batch, seed=step)
            z_init = self.normalize(z_init)
            z_next = optimizer.optimize(z_init, self.acq_function)
            z_next = self.unnormalize(z_next)
            is_valid = self.bo_config.vae.check_if_valid(z_next)
            x_next = self.bo_config.vae.decode(z_next)
            if self.bo_config.dataset_name in ["selfies", "smiles"]:
                x_next = x_next.argmax(dim=-1).long()
            y_next = self.bo_config.blackbox_function(x_next)
            na_idx = torch.isnan(y_next)
            self.tracker.add_solution(y_next, is_valid)

            z_next = z_next[~na_idx, ...]
            y_next = y_next[~na_idx].unsqueeze(-1)
            self._Z = torch.cat([self._Z, z_next], dim=0)
            self.y = torch.cat([self.y, y_next], dim=0)
            initializer.Z = self.Z
            initializer.y = self.y
            if self.use_turbo and len(y_next) > 0:
                self.turbo_state = update_state(self.turbo_state, y_next)
            self.fit_gp()
        summary = self.tracker.summarize()
        return summary


def test_bo(dataset_name: str, architecture: str, beta: float):
    import yaml
    from les.nets.utils import get_vae
    from les.utils.les import LES
    from les.utils.opt_utils import get_black_box_function, get_train_test_data

    if dataset_name == "expressions":
        lr = 0.8
    elif dataset_name == "smiles":
        lr = 0.003
    elif dataset_name == "selfies":
        lr = 0.03
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    vae = get_vae(dataset_name, architecture, beta)[0]

    objective = "pdop"

    (Z, y), _ = get_train_test_data(
        dataset_name,
        vae=vae,
        sample_size=500,
        true_y=False,
        run=42,
        objective=objective,
        old=False,
    )
    penalties = [LES(vae), None]
    for penalty in penalties:
        optimizer_spec = OptimizerSpec(
            method="ga",
            n_steps=10,
            alpha=0.5,
            learning_rate=lr,
            penalty=penalty,
        )
        train_idx = np.random.choice(np.arange(len(Z)), 500, replace=False)
        Z_train = Z[train_idx, ...]
        y_train = y[train_idx].unsqueeze(-1)
        initializer_spec = InitializerSpec(
            Z=Z_train,
            y=y_train,
            use_turbo=False,
        )
        bo_config = BOConfig(
            initializer_spec=initializer_spec,
            optimizer_spec=optimizer_spec,
            blackbox_function=get_black_box_function(dataset_name, objective=objective),
            vae=vae,
        )
        bo = BayesianOptimizer(bo_config)
        summary = bo.optimize(n_batch=10, n_steps=10)
        # save to yaml file
        penalty_name = "les" if penalty is not None else "none"
        with open(
            f"bo_{dataset_name}_{architecture}_{beta}_{penalty_name}.yaml", "w"
        ) as f:
            yaml.dump(summary, f)


if __name__ == "__main__":
    Fire(test_bo)
