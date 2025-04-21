from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np

import torch

from les.utils.les import LES, net_derivative

from les.opt.utils import normalize_grad

from botorch.optim.optimize import optimize_acqf


@dataclass
class OptimizerSpec:
    method: str  # optimization method "ga" or "lbfgs"
    n_steps: int  # number of steps for gradient ascent only relevant for "ga"
    alpha: float  # penalty parameter only relevant for "ga"
    learning_rate: float  # learning rate for gradient ascent only relevant for "ga"
    penalty: Optional[Callable] = None  # penalty function only relevant for "ga"
    es_rule: Optional[Callable] = None  # early stopping rule only relevant for "ga"
    bounds: Optional[Tuple[List[float], List[float]]] = (
        None  # bounds for the optimization only relevant for "lbfgs"
    )


class AcquisitionOptimizer:
    def __init__(self, optimizer_spec: OptimizerSpec):
        self.optimizer_spec = optimizer_spec

    def optimize(self, z_init, objective):
        if self.optimizer_spec.method == "lbfgs":
            return self.optimize_lbfgs(z_init, objective)
        elif self.optimizer_spec.method == "ga":
            return self.optimize_ga(z_init, objective)

    def optimize_ga(self, z_init, objective):
        z = z_init.clone()
        if isinstance(self.optimizer_spec.penalty, LES):
            # precompute a_omega
            a_omega = net_derivative(z, self.optimizer_spec.penalty.model.decoder)
        for s in range(self.optimizer_spec.n_steps):
            z_grad = z.detach().clone()
            z_grad = z_grad.clone().requires_grad_(True)
            _b_s = 1
            n_batches = z_grad.shape[0] // _b_s + z_grad.shape[0] % _b_s

            for i in range(n_batches):
                z_grad_i = (
                    z_grad[i * _b_s : (i + 1) * _b_s, ...]
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )
                grad_i = torch.autograd.grad(objective(z_grad_i), z_grad_i)[0].detach()
                grad_i = normalize_grad(grad_i, torch.ones_like(grad_i).mean(dim=1))
                if (
                    self.optimizer_spec.penalty is not None
                    and self.optimizer_spec.alpha > 0
                ):
                    z_grad_i = (
                        z_grad[i * _b_s : (i + 1) * _b_s, ...]
                        .clone()
                        .detach()
                        .requires_grad_(True)
                    )
                    if isinstance(self.optimizer_spec.penalty, LES):
                        pen = partial(
                            self.optimizer_spec.penalty,
                            a_omega=a_omega[i * _b_s : (i + 1) * _b_s, ...],
                        )
                    else:
                        pen = self.optimizer_spec.penalty
                    grad_pen = torch.autograd.grad(pen(z_grad_i), z_grad_i)[0].detach()
                    grad_pen = normalize_grad(
                        grad_pen, torch.ones_like(grad_pen).mean(dim=1)
                    )
                    grad_i += self.optimizer_spec.alpha * grad_pen

                if i == 0:
                    grad = grad_i
                else:
                    grad = torch.cat([grad, grad_i], dim=0)

            grad = grad * self.optimizer_spec.learning_rate

            if self.optimizer_spec.es_rule is not None:
                z_proposed = z.clone().cpu() + grad.cpu()
                es_vec = self.optimizer_spec.es_rule(z_proposed)
                es_vec = np.array([es_vec] * z.shape[1], dtype=int).T
                es_vec = torch.tensor(es_vec).to(dtype=z.dtype)

                grad *= es_vec
            z = z + grad
        candidate = z.detach().clone()

        return candidate

    def optimize_lbfgs(self, z_init, objective):
        lb = self.optimizer_spec.bounds[0].clone()
        ub = self.optimizer_spec.bounds[1].clone()
        bounds = torch.stack([lb, ub], dim=0)
        z_new, _ = optimize_acqf(
            objective,
            bounds=bounds,
            q=z_init.shape[0],
            batch_initial_conditions=z_init,
            num_restarts=1,
            return_best_only=False,
            sequential=False,
        )
        return torch.squeeze(z_new)


def test_optimizer():
    from les.utils.les import LES
    from les.nets.utils import get_vae
    from les.utils.opt_utils import get_train_test_data

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
    penalty = LES(vae)
    optimizer_spec = OptimizerSpec(
        method="ga",
        n_steps=10,
        alpha=0.1,
        learning_rate=0.1,
        penalty=penalty,
    )
    optimizer = AcquisitionOptimizer(optimizer_spec)
    # is_valid_init = vae.is_valid(z_init)
    z_init_idx = np.random.choice(np.arange(len(Z)), 10, replace=False)
    z_init = Z[z_init_idx, ...]
    z_next = optimizer.optimize(z_init, lambda x: x.sum())
    lb = torch.tensor([-1] * z_init.shape[1], dtype=z_init.dtype)
    ub = torch.tensor([1] * z_init.shape[1], dtype=z_init.dtype)
    bounds = torch.stack([lb, ub], dim=0)
    optimizer_spec = OptimizerSpec(
        method="lbfgs",
        n_steps=10,
        alpha=0.1,
        learning_rate=0.1,
        penalty=penalty,
        bounds=bounds,
    )
    optimizer = AcquisitionOptimizer(optimizer_spec)
    z_next = optimizer.optimize(z_init, lambda x: x.sum())


if __name__ == "__main__":
    test_optimizer()
