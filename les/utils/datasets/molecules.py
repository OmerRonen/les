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

# from tdc import Oracle
from tqdm import tqdm

from lolbo.utils.mol_utils.selfies_vae.data import DEFAULT_SELFIES_VOCAB
from les import LOGGER
from ..utils import one_hot_encode, one_hot_to_eq
from ugo.utils.quality_filters import pass_quality_filter

VOCAB = [
    "C",
    "(",
    ")",
    "c",
    "1",
    "2",
    "o",
    "=",
    "O",
    "N",
    "3",
    "F",
    "[",
    "@",
    "H",
    "]",
    "n",
    "-",
    "#",
    "S",
    "l",
    "+",
    "s",
    "B",
    "r",
    "/",
    "4",
    "\\",
    "5",
    "6",
    "7",
    "I",
    "P",
    "8",
    " ",
]
EOS_TOKEN_IDX = len(VOCAB) - 1
VOCAB_SIZE = len(VOCAB)
VOCAB = {v: i for i, v in enumerate(VOCAB)}
EXPR_LENGTH = 120

EXPR_LENGTH_SELFIES = 72
SELFIES_VOCAB = [
    "[#Branch1]",
    "[#Branch2]",
    "[#C-1]",
    "[#C]",
    "[#N+1]",
    "[#N]",
    "[#O+1]",
    "[=B]",
    "[=Branch1]",
    "[=Branch2]",
    "[=C-1]",
    "[=C]",
    "[=N+1]",
    "[=N-1]",
    "[=NH1+1]",
    "[=NH2+1]",
    "[=N]",
    "[=O+1]",
    "[=OH1+1]",
    "[=O]",
    "[=PH1]",
    "[=P]",
    "[=Ring1]",
    "[=Ring2]",
    "[=S+1]",
    "[=SH1]",
    "[=S]",
    "[=Se+1]",
    "[=Se]",
    "[=Si]",
    "[B-1]",
    "[BH0]",
    "[BH1-1]",
    "[BH2-1]",
    "[BH3-1]",
    "[B]",
    "[Br+2]",
    "[Br-1]",
    "[Br]",
    "[Branch1]",
    "[Branch2]",
    "[C+1]",
    "[C-1]",
    "[CH1+1]",
    "[CH1-1]",
    "[CH1]",
    "[CH2+1]",
    "[CH2]",
    "[C]",
    "[Cl+1]",
    "[Cl+2]",
    "[Cl+3]",
    "[Cl-1]",
    "[Cl]",
    "[F+1]",
    "[F-1]",
    "[F]",
    "[H]",
    "[I+1]",
    "[I+2]",
    "[I+3]",
    "[I]",
    "[N+1]",
    "[N-1]",
    "[NH0]",
    "[NH1+1]",
    "[NH1-1]",
    "[NH1]",
    "[NH2+1]",
    "[NH3+1]",
    "[N]",
    "[O+1]",
    "[O-1]",
    "[OH0]",
    "[O]",
    "[P+1]",
    "[PH1]",
    "[PH2+1]",
    "[P]",
    "[Ring1]",
    "[Ring2]",
    "[S+1]",
    "[S-1]",
    "[SH1]",
    "[S]",
    "[Se+1]",
    "[Se-1]",
    "[SeH1]",
    "[SeH2]",
    "[Se]",
    "[Si-1]",
    "[SiH1-1]",
    "[SiH1]",
    "[SiH2]",
    "[Si]",
    "[C@@H1]",
    " ",
]
# load missed tokens.yaml
missed_tokens_file = "missed_tokens.yaml"
if os.path.exists(missed_tokens_file):
    # read the yaml file
    with open(missed_tokens_file, "r") as f:
        missed_tokens = yaml.safe_load(f)
        # add missed tokens to the vocab
        for token in missed_tokens:
            if token not in SELFIES_VOCAB:
                SELFIES_VOCAB.append(token)
        SELFIES_VOCAB = {v: i for i, v in enumerate(SELFIES_VOCAB)}
        SELFIES_VOCAB_INV = {v: k for k, v in SELFIES_VOCAB.items()}
        # print(f"vocab size: {len(SELFIES_VOCAB)}")
SELFIES_VOCAB = {v: i for i, v in enumerate(SELFIES_VOCAB)}
SELFIES_VOCAB_OLD = {v: i for i, v in enumerate(DEFAULT_SELFIES_VOCAB)}

from guacamol import standard_benchmarks

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


def save_train_stats(smiles, file_name):
    logP_values = []
    SAS_values = []
    cycle_values = []
    for smile in tqdm(smiles):
        try:
            mol = rdkit.Chem.MolFromSmiles(smile)

            logP_score = Descriptors.MolLogP(mol)

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

            logP_values.append(logP_score)
            SAS_values.append(SAS_score)
            cycle_values.append(cycle_score)
        except:
            pass

    logP_mean = np.mean(logP_values)
    logP_std = np.std(logP_values)

    SAS_mean = np.mean(SAS_values)
    SAS_std = np.std(SAS_values)

    cycles_mean = np.mean(cycle_values)
    cycles_std = np.std(cycle_values)

    train_stats = {
        "logP_mean": logP_mean,
        "logP_std": logP_std,
        "SAS_mean": SAS_mean,
        "SAS_std": SAS_std,
        "cycles_mean": cycles_mean,
        "cycles_std": cycles_std,
    }

    with open(file_name, "wb") as f:
        pickle.dump(train_stats, f)


def calculate_logP(smile, option):
    mol = Chem.MolFromSmiles(smile)
    if option == 1:
        return Crippen.MolLogP(mol)
    elif option == 2:
        return Descriptors.MolLogP(mol)


def compute_target_logP(smile, default_value, norm=True):
    train_stats = pickle.load(
        open(
            "/accounts/campus/omer_ronen/projects/lso_splines/data/molecules/train_stats.pkl",
            "rb",
        )
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
    X, property, na_value=np.nan, is_selfies=True, norm=True, old=False
):
    # convert from one hot to string

    oracle_values = []
    X = torch.squeeze(X)
    if is_selfies and len(X.shape) == 2:
        vcb = SELFIES_VOCAB_OLD if old else SELFIES_VOCAB
        X = F.one_hot(X, num_classes=len(vcb)).float()
    if len(X.shape) == 2:
        X = X.unsqueeze(0)

    for i in range(X.shape[0]):
        x = X[i, ...]
        if is_selfies:
            vcb = SELFIES_VOCAB if not old else SELFIES_VOCAB_OLD
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


def translate_to_selfies(smiles_file):
    out_file = smiles_file.replace(".smi", ".self")
    # if os.path.exists(out_file):
    #     return
    smiles = open(smiles_file, "r").read().splitlines()
    selfies = []
    # max_len = 0
    with open(out_file, "w") as f:
        for smile in tqdm(smiles):
            if pass_quality_filter([smile])[0] == 0:
                continue
            try:
                selfie = sf.encoder(smile)
                selfies.append(selfie)
                # max_len = max(max_len, len(selfie))
                f.write(selfie + "\n")
            except:
                pass
    # print(max_len)


def get_molecule_data(n=None, max_length=120, selfies=False, save=False):
    # read the data
    f_name_pre = "/accounts/campus/omer_ronen/projects/lso_splines/data/molecules"
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
        expressions_lst = open(f_name, "r").read().splitlines()
        # convert to one hot
        expressions_lst_oh = [
            one_hot_encode(e, vcb, max_length=None) for e in tqdm(expressions_lst)
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
    # translate_to_selfies("data/molecules/250k_rndm_zinc_drugs_clean.smi")
    ds = get_molecule_data(n=None, selfies=True, max_length=72, save=False)
    print(ds.shape)
    # dataset = get_molecule_data(n=20000)
