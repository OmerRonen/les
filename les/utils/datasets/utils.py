from lolbo.vae import get_selfies_data

from .expressions import get_expressions_data
from .molecules import get_molecule_data
from ..utils import split_data


def get_dataset(dataset_name, pretrained=False):
    if dataset_name == "expressions":
        data = get_expressions_data()
    elif dataset_name == "smiles":
        data = get_molecule_data(n=2000)
    elif dataset_name == "selfies" and not pretrained:
        data = get_molecule_data(selfies=True, n=2000)
    elif dataset_name == "selfies" and pretrained:
        data = get_selfies_data()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    train, test = split_data(data)
    return {"train": train, "test": test}
