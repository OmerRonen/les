import time
import yaml
import torch

from functools import partial

from les.nets.utils import get_vae
from les.utils.uncertainty import calculate_uncertainty
from les.utils.les import LES


def calculate_les(z, model):
    les = LES(model)
    return les(z)


def calculate_les_derivative(z, model):
    les = LES(model)
    return torch.autograd.grad(les(z), z)[0]


def wall_clock(f, z):
    s = time.time()
    f(z)
    e = time.time()
    elapsed = float(e - s)
    # round to 3 decimal places
    return round(elapsed, 3)


def main():
    results = {}
    for dataset in ["expressions", "smiles", "selfies"]:
        vae, _ = get_vae(dataset)

        z_5 = torch.randn(5, vae.latent_dim).to(device=vae.device, dtype=vae.dtype)
        z_10 = torch.randn(10, vae.latent_dim).to(device=vae.device, dtype=vae.dtype)
        uc_10_10 = partial(calculate_uncertainty, model=vae, n_models=10, n_preds=10)
        uc_2_2 = partial(calculate_uncertainty, model=vae, n_models=2, n_preds=2)
        density = partial(calculate_les, model=vae)
        density_derivative = partial(calculate_les_derivative, model=vae)
        z_10_time = {
            "density": wall_clock(density, z_10),
            "density_derivative": wall_clock(density_derivative, z_10),
            "uc_10_10": wall_clock(uc_10_10, z_10),
            "uc_2_2": wall_clock(uc_2_2, z_10),
        }
        z_5_time = {
            "density": wall_clock(density, z_5),
            "density_derivative": wall_clock(density_derivative, z_5),
            "uc_10_10": wall_clock(uc_10_10, z_5),
            "uc_2_2": wall_clock(uc_2_2, z_5),
        }

        results[dataset] = {"z_5": z_5_time, "z_10": z_10_time}
    with open("wall_clock.yaml", "w") as f:
        yaml.dump(results, f)
    print(results)


if __name__ == "__main__":
    main()
