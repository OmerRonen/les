from lolbo.vae import get_selfies_data

from .expressions import get_expressions_data
from .images import get_mnist_data
from .molecules import get_molecule_data
from ..utils import split_data


def get_dataset(dataset_name, digit=3):
    if dataset_name == "expressions":
        data = get_expressions_data()
    elif dataset_name == "mnist":
        data = get_mnist_data(digit=digit).data
    elif dataset_name == "smiles":
        data = get_molecule_data()
    elif dataset_name == "selfies":
        data = get_selfies_data()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    train, test = split_data(data)
    return {"train": train, "test": test}
