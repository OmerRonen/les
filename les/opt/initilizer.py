import torch

import numpy as np

from dataclasses import dataclass
from typing import Callable, Tuple

from botorch.optim.initializers import gen_batch_initial_conditions


@dataclass
class InitializerTurboConfig:
    acq_function: Callable = None
    turbo_bounds: Tuple[float, float] = None


@dataclass
class InitializerSpec:
    Z: np.ndarray
    y: np.ndarray
    use_turbo: bool = False
    turbo_config: InitializerTurboConfig = None


class Initializer:
    def __init__(
        self,
        initializer_spec: InitializerSpec,
    ):
        self.initializer_spec = initializer_spec

    def get_z_init(self, n_batch, seed) -> torch.Tensor:
        np.random.seed(seed)
        if self.initializer_spec.use_turbo:
            z_batch = gen_batch_initial_conditions(
                acq_function=self.initializer_spec.turbo_config.acq_function,
                bounds=self.initializer_spec.turbo_config.turbo_bounds,
                q=n_batch,
                raw_samples=n_batch,
                num_restarts=1,
            )
            z_batch = torch.squeeze(z_batch)
            return z_batch
        idx = np.random.choice(
            np.arange(len(self.initializer_spec.Z)), n_batch, replace=False
        )
        return self.initializer_spec.Z[idx, ...]


def test_initializer():
    from les.utils.opt_utils import get_train_test_data
    from les.nets.utils import get_vae
    from botorch.models import SingleTaskGP
    from gpytorch import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.logei import qLogExpectedImprovement

    def _get_turbo_bounds(Z, y):
        z = Z
        gp = SingleTaskGP(train_X=z, train_Y=y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        mll.to(Z)
        # try:
        fit_gpytorch_mll(mll)
        x_center = z[y.argmax(), :].clone()
        weights = gp.covar_module.base_kernel.lengthscale.squeeze().detach()

        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        weights = weights.to(device=Z.device, dtype=Z.dtype)
        x_center = x_center.to(device=Z.device, dtype=Z.dtype)
        tr_lb = torch.clamp(x_center - weights * 1.0 / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * 1.0 / 2.0, 0.0, 1.0)
        bounds = torch.stack([tr_lb, tr_ub], dim=0)
        return bounds

    dataset_name = "expressions"
    vae = get_vae(dataset_name, "transformer", 1)[0]
    (Z, y), _ = get_train_test_data(
        dataset_name,
        vae=vae,
        sample_size=20,
        true_y=False,
        run=42,
        objective=None,
        old=False,
    )
    train_idx = np.random.choice(np.arange(len(Z)), 10, replace=False)
    Z_train = Z[train_idx, ...]
    y_train = y[train_idx].unsqueeze(-1)
    gp = SingleTaskGP(train_X=Z_train, train_Y=y_train)
    acq = qLogExpectedImprovement(gp, best_f=y_train.max())
    initializer_spec = InitializerSpec(
        Z=Z_train,
        y=y_train,
        use_turbo=True,
        turbo_config=InitializerTurboConfig(
            acq_function=acq,
            turbo_bounds=_get_turbo_bounds(Z_train, y_train),
        ),
    )
    initializer = Initializer(initializer_spec)
    initializer.get_z_init(10, 42)
    initializer_spec = InitializerSpec(
        Z=Z_train,
        y=y_train,
        use_turbo=False,
    )
    initializer = Initializer(initializer_spec)
    initializer.get_z_init(10, 42)
    # print(z_init)


if __name__ == "__main__":
    test_initializer()
