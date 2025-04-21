import logging
from functools import partial

import numpy as np
import torch

import torch.nn.functional as F


from les.utils.datasets.expressions import get_black_box_objective_expression
from les.utils.datasets.molecules import (
    SELFIES_VOCAB_PRETRAINED,
    get_black_box_objective_molecules,
)
from les.utils.datasets.utils import get_dataset

LOGGER = logging.getLogger("OptUtils")


def _encode(X, vae, batch_size=32):
    n_batches = int(np.ceil(len(X) / batch_size))
    vae.encoder.eval()
    Z = []
    for batch in range(n_batches):
        X_batch = X[batch * batch_size : (batch + 1) * batch_size, ...]
        z_i = vae.encode(X_batch).detach()

        # LOGGER.debug(f"z_i: {z_i.shape}")
        if len(z_i.shape) == 3 and z_i.shape[0] == 1:
            z_i = torch.squeeze(z_i, dim=0)
        Z.append(z_i)
    Z = torch.cat(Z, dim=0)
    return Z


def get_objective(Z, vae, black_box_function, dataset_name, batch_size=32):
    X = []
    n_batches = int(np.ceil(len(Z) / batch_size))
    vae.decoder.eval()
    for batch in range(n_batches):
        x_i = vae.decode(Z[batch * batch_size : (batch + 1) * batch_size, ...]).detach()
        X.append(x_i)
    X = torch.cat(X, dim=0)
    if dataset_name in ["selfies", "smiles"]:
        X = X.argmax(dim=-1).long()
    if dataset_name == "smiles":
        # make into one hot
        X = F.one_hot(X, num_classes=vae.decoder.vocab_size).float()
    y = black_box_function(X).to(device=vae.device, dtype=vae.dtype)
    # make y into a numpy array
    # y = y.detach().cpu().numpy()
    return y


def get_black_box_function(dataset_name, norm=True, objective=None, pretrained=False):
    if dataset_name == "expressions":
        return partial(get_black_box_objective_expression, black_box_dict={})
    elif dataset_name == "selfies":
        return partial(
            get_black_box_objective_molecules,
            property=objective,
            norm=norm,
            pretrained=pretrained,
        )  # ,
    elif dataset_name in ["smiles", "molecules"]:
        property = "logp"
        return partial(
            get_black_box_objective_molecules,
            property=property,
            is_selfies=False,
            na_value=np.nan,
            norm=norm,
        )  # ,

    else:
        raise ValueError(f"dataset {dataset_name} is not supported")


def get_train_test_data(
    dataset_name,
    sample_size,
    vae,
    true_y=False,
    run=None,
    objective=None,
    pretrained=False,
):
    dataset = get_dataset(dataset_name, pretrained=pretrained)
    # vae, _ = get_vae(dataset_name)
    black_box_function = get_black_box_function(
        dataset_name, objective=objective, pretrained=pretrained
    )

    train_data = dataset["train"]
    # print(train_data)
    test_data = dataset["test"]
    if run is not None:
        # set seed
        np.random.seed(run)
        ind_run = np.random.randint(0, len(train_data), sample_size)
        np.random.seed(run)
        ind_test = np.random.randint(0, len(test_data), sample_size)
    else:
        ind_run = np.random.randint(0, len(train_data), sample_size)
        ind_test = np.random.randint(0, len(test_data), sample_size)

    X_train = train_data[ind_run]
    X_test = train_data[ind_test]

    old_selfies = dataset_name == "selfies" and pretrained
    if old_selfies:
        X_test = X_test.long()
        X_train = X_train.long()
    batch_size = 32
    # encode train and test
    Z_train = _encode(X_train, vae, batch_size=batch_size)
    Z_test = _encode(X_test, vae, batch_size=batch_size)

    if old_selfies:
        # make into one hot again
        X_train = F.one_hot(X_train, num_classes=len(SELFIES_VOCAB_PRETRAINED)).float()
        X_test = F.one_hot(X_test, num_classes=len(SELFIES_VOCAB_PRETRAINED)).float()

    # LOGGER.debug(f"Z_train: {Z_train.shape}, Z_test: {Z_test.shape}")

    if true_y:
        y_train = get_objective(
            Z_train, vae, black_box_function, dataset_name, batch_size=batch_size
        )
        y_test = get_objective(
            Z_test, vae, black_box_function, dataset_name, batch_size=batch_size
        )
    else:
        # print(X_train.shape)
        y_train = (
            black_box_function(X_train)
            .to(device=vae.device, dtype=vae.dtype)
            .detach()
            .cpu()
            .numpy()
        )
        y_test = (
            black_box_function(X_test)
            .to(device=vae.device, dtype=vae.dtype)
            .detach()
            .cpu()
            .numpy()
        )

    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    # remove nans
    ind_train = torch.isnan(y_train)
    ind_test = torch.isnan(y_test)
    y_train = y_train[~ind_train]
    y_test = y_test[~ind_test]
    Z_train = Z_train[~ind_train]
    Z_test = Z_test[~ind_test]
    LOGGER.debug(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    return (Z_train, y_train), (Z_test, y_test)
