import torch

import numpy as np

from functools import partial
import torch.nn.functional as F

from scales.utils.data_utils.utils import get_dataset
from scales.utils.vae import get_vae
from scales.utils.data_utils.expressions import get_black_box_objective_expression
from scales.utils.data_utils.images import get_black_box_objective_mnist
from scales.utils.data_utils.molecules import get_black_box_objective_molecules


def _encode(X, vae, batch_size=32):
    n_batches = int(np.ceil(len(X) / batch_size))
    vae.encoder.eval()
    Z = []
    for batch in range(n_batches):
        z_i = vae.encode(X[batch * batch_size:(batch + 1) * batch_size, ...], add_noise=False).detach()
        # LOGGER.info(f"z_i: {z_i.shape}")
        if len(z_i.shape) == 3 and z_i.shape[0] == 1:
            z_i = torch.squeeze(z_i, dim=0)
        Z.append(z_i)
    Z = torch.cat(Z, dim=0)
    return Z


def _get_objective(Z, vae, black_box_function, dataset_name, batch_size=32):
    X = []
    n_batches = int(np.ceil(len(Z) / batch_size))
    vae.decoder.eval()
    for batch in range(n_batches):
        x_i = vae.decode(Z[batch * batch_size:(batch + 1) * batch_size, ...]).detach()
        X.append(x_i)
    X = torch.cat(X, dim=0)
    if dataset_name in ["selfies", "smiles"]:
        X = X.argmax(dim=-1).long()
    if dataset_name == "smiles":
        # make into one hot
        X = F.one_hot(X, num_classes=vae.decoder.vocab_size).float()
    y = black_box_function(X).to(device=vae.device, dtype=vae.dtype)
    # make y into a numpy array
    y = y.detach().cpu().numpy()
    return y


def get_black_box_function(dataset_name, norm=True, objective = None):
    if dataset_name == "expressions":
        return partial(get_black_box_objective_expression,
                       black_box_dict={})
        # black_box_dict=pickle.load(open("data/grammer/black_box_data.pickle", "rb")))
    elif dataset_name == "selfies":
        # property = "logp"
        return partial(get_black_box_objective_molecules, property=objective, norm=norm)#,
                       # dict=pickle.load(open(f"data/molecules/{property}_dict.pkl", "rb")))
    elif dataset_name == "smiles":
        property = "logp"
        return partial(get_black_box_objective_molecules, property=property, is_selfies=False, na_value=np.nan, norm=norm)#,
    elif dataset_name == "mnist":
        return get_black_box_objective_mnist
    else:
        raise ValueError(f"dataset {dataset_name} is not supported")

def get_train_test_data(dataset_name, sample_size, true_y=False, run=None, objective=None):
    dataset = get_dataset(dataset_name)
    vae, _ = get_vae(dataset_name)
    black_box_function = get_black_box_function(dataset_name, objective=objective)

    train_data = dataset["train"]
    test_data = dataset["test"]
    if run is not None:
        # set seed
        np.random.seed(run)
        ind_run = np.random.randint(0, len(train_data), sample_size)
        np.random.seed(run)
        ind_test = np.random.randint(0, len(test_data), sample_size)
    else:
        ind_run = np.random.choice(len(train_data), sample_size, replace=False)
        ind_test = np.random.choice(len(test_data), sample_size, replace=False)

    X_train = train_data[ind_run].to(device=vae.device, dtype=vae.dtype)
    X_test = train_data[ind_test].to(device=vae.device, dtype=vae.dtype)

    if dataset_name in ["selfies"]:
        X_test = X_test.long()
        X_train = X_train.long()
    batch_size = 32

    # encode train and test
    Z_train = _encode(X_train, vae, batch_size=batch_size)
    Z_test = _encode(X_test, vae, batch_size=batch_size)

    if true_y:

        y_train = _get_objective(Z_train, vae, black_box_function, dataset_name, batch_size=batch_size)
        y_test = _get_objective(Z_test, vae, black_box_function, dataset_name, batch_size=batch_size)
    else:
        y_train = black_box_function(X_train).to(device=vae.device, dtype=vae.dtype).detach().cpu().numpy()
        y_test = black_box_function(X_test).to(device=vae.device, dtype=vae.dtype).detach().cpu().numpy()

    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    # remove nans
    ind_train = torch.isnan(y_train)
    ind_test = torch.isnan(y_test)
    y_train = y_train[~ind_train]
    y_test = y_test[~ind_test]
    Z_train = Z_train[~ind_train]
    Z_test = Z_test[~ind_test]
    # LOGGER.info(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    return (Z_train, y_train), (Z_test, y_test)

