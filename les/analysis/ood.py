from functools import partial
import os
import fire
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from les import LOGGER
from les.nets.template import VAE
from les.nets.utils import get_vae
from les.utils.datasets.expressions import get_expressions_data
from les.utils.datasets.molecules import get_molecule_data
from les.utils.les import LES, Likelihood, Prior
from les.utils.uncertainty import calculate_uncertainty
from les.utils.utils import array_to_tensor
from lolbo.vae import get_selfies_data


def calculate_distance_to_train(z, model, dataset, k=3, n=None):
    # Sample dataset if needed
    if n is not None and len(dataset) > n:
        dataset = dataset[:n]

    # Move everything to CPU for consistent processing
    original_device = z.device
    z = z.to(device="cpu")
    if torch.cuda.is_available():
        dataset = dataset.to(device="cuda")
        model = model.to(device="cuda")
    else:
        dataset = dataset.to(device="cpu")
        model = model.to(device="cpu")

    # Encode training data
    try:
        z_train = model.encoder(dataset)[0]
    except Exception:
        z_train = model.encode(dataset)

    z_train = z_train.to(device="cpu")

    distances_mean = []
    b_s = 100
    n_batches = len(z_train) // b_s + 1
    for z_i in z:
        distances = []
        for i in range(n_batches):
            z_batch = z_train[i * b_s : (i + 1) * b_s]
            distances.append(torch.norm(z_i - z_batch, dim=1))
        distances = torch.cat(distances, dim=0)
        distances = torch.topk(distances, k, largest=False).values
        distances_mean.append(distances.mean())

    d = -1 * torch.tensor(distances_mean)
    return d.to(device=original_device)


def _encode_data(model, dataset):
    try:
        z_data = model.encoder(dataset)[0]
    except Exception:
        model = model.to(dataset.device)
        model.decoder = model.decoder.to(dataset.device)
        dataset = dataset.to(dataset.device)
        z_data = model.encode(dataset)
    return z_data


def _calc_score_in_batches(z, scores_func, batch_size):
    n_batches = len(z) // batch_size + int(len(z) % batch_size > 0)
    scores_list = []
    for i in range(n_batches):
        z_batch = z[i * batch_size : (i + 1) * batch_size]
        scores = scores_func(z_batch)
        scores_list.append(scores)
    return torch.cat(scores_list, dim=0)


def get_methods_auc(model: VAE, dataset: torch.Tensor, model_name: str, batch_size=4):
    z_data = _encode_data(model, dataset)
    z_prior = torch.randn_like(z_data) * model.eps_std
    z_far = torch.randn_like(z_data) * 5 * model.eps_std
    model_path = os.path.join("results", "vaes", model_name)

    scores_func = {
        "les": LES(model=model, polarity=False),
        "polarity": LES(model=model, polarity=True),
        "train_distances": partial(
            calculate_distance_to_train, model=model, dataset=dataset, k=3
        ),
        "likelihood": Likelihood(model=model),
        "prior": Prior(model=model),
        "uncertainty": partial(
            calculate_uncertainty,
            n_models=10,
            n_preds=40,
            dataset=dataset,
            model=model,
        ),
    }

    # Create results dictionary
    results = {"is_valid": []}
    for k in scores_func.keys():
        results[k] = []

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Process different latent vectors
    for z in [z_data, z_prior, z_far]:
        is_valid = model.check_if_valid(z)
        results["is_valid"] += is_valid.tolist()

        for k in scores_func.keys():
            if k in ["les", "polarity"]:
                r = (
                    _calc_score_in_batches(z, scores_func[k], batch_size)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                r = scores_func[k](z)

            if isinstance(r, torch.Tensor):
                r = r.detach().cpu().numpy()
            results[k] += r.tolist()

    # Convert to numpy arrays for AUC calculation
    for k in results.keys():
        if k == "is_valid":
            results[k] = np.array(results[k], dtype=np.int32)
            continue
        results[k] = np.array(results[k])

    # Calculate AUROC for each method
    auroc_results = {}
    for k in results.keys():
        if k != "is_valid":
            score_k = results[k]
            # normalize to be between 0 and 1
            score_k = (score_k - score_k.min()) / (score_k.max() - score_k.min())
            auroc = roc_auc_score(y_true=results["is_valid"], y_score=score_k)
            auroc_results[k] = auroc
    LOGGER.info(
        f"AUROC results:\n{pd.DataFrame(auroc_results, index=[0]).round(3).to_string(index=False)}"
    )
    # return auroc_results


def get_tensor_data(ds, n=500, pretrained=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ds == "smiles":
        return array_to_tensor(get_molecule_data(n=n), device)
    elif ds == "expressions":
        return array_to_tensor(get_expressions_data(n=n), device)
    elif ds == "selfies" and not pretrained:
        return array_to_tensor(get_molecule_data(n=n, selfies=True), device)
    elif ds == "selfies" and pretrained:
        return array_to_tensor(get_selfies_data(n=n), device)
    else:
        raise ValueError(f"Dataset {ds} not supported")


def main(dataset: str, architecture: str, beta: float, pretrained: bool = False):
    n = 500

    dataset_tensor = get_tensor_data(dataset, n=n, pretrained=pretrained)
    vae = get_vae(
        dataset=dataset, architecture=architecture, beta=beta, pretrained=pretrained
    )
    if torch.cuda.is_available():
        vae = vae.to("cuda")
        vae.decoder = vae.decoder.to("cuda")
        vae.encoder = vae.encoder.to("cuda")
    results = get_methods_auc(
        model=vae, dataset=dataset_tensor, model_name=f"{dataset}_{architecture}_{beta}"
    )
    return results


if __name__ == "__main__":
    fire.Fire(main)
