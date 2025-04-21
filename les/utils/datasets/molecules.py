import os
import yaml
import networkx as nx
import rdkit
import torch
import pickle

import numpy as np
import pandas as pd
import selfies as sf
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops, Crippen
from rdkit.Contrib.SA_Score import sascorer
from guacamol import standard_benchmarks

# from tdc import Oracle
from tqdm import tqdm

from les import LOGGER
from les.utils.utils import one_hot_encode, one_hot_to_eq


MAX_LENGTH = 120
MAX_LENGTH_SELFIES = 72
VOCAB = yaml.load(open("data/molecules/vocab.yaml", "r"), Loader=yaml.FullLoader)
SELFIES_VOCAB = yaml.load(
    open("data/molecules/vocab_selfies.yaml", "r"), Loader=yaml.FullLoader
)
SELFIES_VOCAB_PRETRAINED = yaml.load(
    open("data/molecules/vocab_selfies_pretrained.yaml", "r"), Loader=yaml.FullLoader
)


med1 = standard_benchmarks.median_camphor_menthol()  #'Median molecules 1'
med2 = standard_benchmarks.median_tadalafil_sildenafil()  #'Median molecules 2',
pdop = standard_benchmarks.perindopril_rings()  # 'Perindopril MPO',
osmb = standard_benchmarks.hard_osimertinib()  # 'Osimertinib MPO',
adip = standard_benchmarks.amlodipine_rings()  # 'Amlodipine MPO'
siga = standard_benchmarks.sitagliptin_replacement()  #'Sitagliptin MPO'
zale = standard_benchmarks.zaleplon_with_other_formula()  # 'Zaleplon MPO'
valt = standard_benchmarks.valsartan_smarts()  #'Valsartan SMARTS',
dhop = standard_benchmarks.decoration_hop()  # 'Deco Hop'
shop = standard_benchmarks.scaffold_hop()  # Scaffold Hop'
rano = standard_benchmarks.ranolazine_mpo()  #'Ranolazine MPO'
fexo = (
    standard_benchmarks.hard_fexofenadine()
)  # 'Fexofenadine MPO'... 'make fexofenadine less greasy'


guacamol_objs = {
    "med1": med1,
    "pdop": pdop,
    "adip": adip,
    "rano": rano,
    "osmb": osmb,
    "siga": siga,
    "zale": zale,
    "valt": valt,
    "med2": med2,
    "dhop": dhop,
    "shop": shop,
    "fexo": fexo,
}


def smile_to_guacamole_score(obj_func_key, smile):
    if smile is None or len(smile) == 0:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    func = guacamol_objs[obj_func_key]
    score = func.objective.score(smile)
    if score is None:
        return None
    if score < 0:
        return None
    return float(score)


def verify_smile(smile):
    return (
        (smile != "")
        and pd.notnull(smile)
        and (rdkit.Chem.MolFromSmiles(smile) is not None)
    )


def compute_target_logP(smile, default_value, norm=True):
    try:
        train_stats = pickle.load(
            open(
                "data/molecules/train_stats.pkl",
                "rb",
            )
        )
    except Exception:
        raise Exception(
            "Train stats file (data/molecules/train_stats.pkl) not found, please make sure to download it from git"
        )
    if not verify_smile(smile):
        return default_value
    try:
        mol = rdkit.Chem.MolFromSmiles(smile)

        logP_score = Descriptors.MolLogP(mol)

        if logP_score > 20 or logP_score < -20:
            return default_value

        SAS_score = -sascorer.calculateScore(mol)

        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6

        cycle_score = -cycle_length
        if norm:
            logP_score_normalized = (
                logP_score - train_stats["logP_mean"]
            ) / train_stats["logP_std"]
            SAS_score_normalized = (SAS_score - train_stats["SAS_mean"]) / train_stats[
                "SAS_std"
            ]
            cycle_score_normalized = (
                cycle_score - train_stats["cycles_mean"]
            ) / train_stats["cycles_std"]

            return logP_score_normalized + SAS_score_normalized + cycle_score_normalized
        else:
            LOGGER.info(f"LogP: {logP_score}, SAS: {SAS_score}, Cycle: {cycle_score}")
            return logP_score + SAS_score + cycle_score
    except Exception:
        return default_value


def get_black_box_objective_molecules(
    X, property, na_value=np.nan, is_selfies=True, norm=True, pretrained=False
):
    # convert from one hot to string

    oracle_values = []
    X = torch.squeeze(X)
    if is_selfies and len(X.shape) == 2:
        vcb = SELFIES_VOCAB_PRETRAINED if pretrained else SELFIES_VOCAB
        X = F.one_hot(X, num_classes=len(vcb)).float()
    if len(X.shape) == 2:
        X = X.unsqueeze(0)

    for i in range(X.shape[0]):
        x = X[i, ...]
        if is_selfies:
            vcb = SELFIES_VOCAB if not pretrained else SELFIES_VOCAB_PRETRAINED
            if x.shape[1] == len(vcb):
                x = x.transpose(0, 1)
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            selfie = one_hot_to_eq(x, vcb, return_blanks=False)
            # print(f"selfie: {selfie}")
            mol_str = sf.decoder(selfie)
            # print(f"mol_str: {mol_str}")
        else:
            if x.shape[1] == len(VOCAB):
                x = x.transpose(0, 1)
            mol_str = one_hot_to_eq(x.cpu().detach().numpy(), VOCAB)

        if property == "logp":
            oracle_values.append(compute_target_logP(mol_str, na_value, norm=norm))
        else:
            scr = smile_to_guacamole_score(property, mol_str)
            scr = np.nan if scr is None else scr
            oracle_values.append(scr)
    return torch.tensor(oracle_values)


def get_molecule_data(n=None, selfies=False, save=False):
    # read the data
    f_name_pre = "data/molecules"
    f_name_pickle = f"{f_name_pre}/250k_rndm_zinc_drugs_clean.pkl"
    vcb = SELFIES_VOCAB if selfies else VOCAB
    if selfies:
        f_name_pickle = f"{f_name_pre}/250k_rndm_zinc_drugs_clean_selfies.pkl"
    if os.path.exists(f_name_pickle) and not save:
        dataset = pickle.load(open(f_name_pickle, "rb"))
    else:
        f_name = f"{f_name_pre}/250k_rndm_zinc_drugs_clean"
        if selfies:
            f_name = f"{f_name}.self"
        else:
            f_name = f"{f_name}.smi"
        try:
            expressions_lst = open(f_name, "r").read().splitlines()
        except Exception:
            raise Exception(
                f"File {f_name} not found, please make sure to download it from git"
            )
        # convert to one hot
        max_length = MAX_LENGTH_SELFIES if selfies else MAX_LENGTH
        expressions_lst_oh = [
            one_hot_encode(e, vcb, max_length=max_length) for e in tqdm(expressions_lst)
        ]
        max_len = max([e.shape[0] for e in expressions_lst_oh])
        LOGGER.info(f"Max length: {max_len}")
        expressions_lst_oh = [
            one_hot_encode(e, vcb, max_length=max_len) for e in tqdm(expressions_lst)
        ]

        dataset = np.array(expressions_lst_oh)
        pickle.dump(dataset, open(f_name_pickle, "wb"))

    if n is not None:
        # sample n molecules at random
        idx = np.random.choice(dataset.shape[0], n, replace=False)
        dataset = dataset[idx, ...]

    torch_data = torch.from_numpy(np.array(dataset))

    return torch_data


if __name__ == "__main__":
    # # translate_to_selfies("data/molecules/250k_rndm_zinc_drugs_clean.smi")
    # ds = get_molecule_data(n=None, selfies=True, max_length=72, save=False)
    # print(ds.shape)
    # # dataset = get_molecule_data(n=20000)
    # save vocab, vocab_selfies and vocab_selfies_old as yaml files
    data = get_molecule_data(n=2000)
    # with open("data/molecules/vocab.yaml", "w") as f:
    #     yaml.dump(VOCAB, f)
    # with open("data/molecules/vocab_selfies.yaml", "w") as f:
    #     yaml.dump(SELFIES_VOCAB, f)
    # with open("data/molecules/vocab_selfies_pretrained.yaml", "w") as f:
    #     yaml.dump(SELFIES_VOCAB_OLD, f)
