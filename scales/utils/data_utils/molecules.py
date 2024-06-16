import os
import sys
import rdkit
import torch
import pickle


import numpy as np
import pandas as pd
import selfies as sf
import networkx as nx

import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import RDConfig, Descriptors, rdmolops, Crippen
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# from tdc import Oracle
from tqdm import tqdm

# from lolbo.utils.mol_utils.mol_utils import smile_to_guacamole_score
from lolbo.utils.mol_utils.selfies_vae.data import DEFAULT_SELFIES_VOCAB
from scales import LOGGER
from ..utils import one_hot_encode, one_hot_to_eq, probabilities_to_one_hot

VOCAB = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
         '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
         '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
EOS_TOKEN_IDX = len(VOCAB) - 1
VOCAB_SIZE = len(VOCAB)
VOCAB = {v: i for i, v in enumerate(VOCAB)}
EXPR_LENGTH = 120

from guacamol import standard_benchmarks 

med1 = standard_benchmarks.median_camphor_menthol() #'Median molecules 1'
med2 = standard_benchmarks.median_tadalafil_sildenafil() #'Median molecules 2',
pdop = standard_benchmarks.perindopril_rings() # 'Perindopril MPO',
osmb = standard_benchmarks.hard_osimertinib()  # 'Osimertinib MPO',
adip = standard_benchmarks.amlodipine_rings()  # 'Amlodipine MPO' 
siga = standard_benchmarks.sitagliptin_replacement() #'Sitagliptin MPO'
zale = standard_benchmarks.zaleplon_with_other_formula() # 'Zaleplon MPO'
valt = standard_benchmarks.valsartan_smarts()  #'Valsartan SMARTS',
dhop = standard_benchmarks.decoration_hop() # 'Deco Hop'
shop = standard_benchmarks.scaffold_hop() # Scaffold Hop'
rano= standard_benchmarks.ranolazine_mpo() #'Ranolazine MPO' 
fexo = standard_benchmarks.hard_fexofenadine() # 'Fexofenadine MPO'... 'make fexofenadine less greasy'


guacamol_objs = {"med1":med1,"pdop":pdop, "adip":adip, "rano":rano, "osmb":osmb,
        "siga":siga, "zale":zale, "valt":valt, "med2":med2,"dhop":dhop, "shop":shop, 
        'fexo':fexo} 

def smile_to_guacamole_score(obj_func_key, smile):
    if smile is None or len(smile)==0:
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
    return score 

def verify_smile(smile):
    return (smile != '') and pd.notnull(smile) and (rdkit.Chem.MolFromSmiles(smile) is not None)


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

            cycle_score = - cycle_length

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

    train_stats = {'logP_mean': logP_mean, 'logP_std': logP_std,
                   'SAS_mean': SAS_mean, 'SAS_std': SAS_std,
                   'cycles_mean': cycles_mean, 'cycles_std': cycles_std}

    with open(file_name, 'wb') as f:
        pickle.dump(train_stats, f)

def calculate_logP(smile, option):
    mol = Chem.MolFromSmiles(smile)
    if option ==1 :
        return Crippen.MolLogP(mol)
    elif option == 2:
        return Descriptors.MolLogP(mol)


# def smile_to_penalized_logP(smile):
#     """ calculate penalized logP for a given smiles string """
#     if smile is None:
#         return None
#     mol = Chem.MolFromSmiles(smile)
#     if mol is None:
#         return None
#     logp = Crippen.MolLogP(mol)
#     sa = sascorer.calculateScore(mol)
#     cycle_length = _cycle_score(mol)
#     """
#     Calculate final adjusted score.
#     These magic numbers are the empirical means and
#     std devs of the dataset.
#
#     I agree this is a weird way to calculate a score...
#     but this is what previous papers did!
#     """
#     score = (
#             (logp - 2.45777691) / 1.43341767
#             + (-sa + 3.05352042) / 0.83460587
#             + (-cycle_length - -0.04861121) / 0.28746695
#     )
#     return max(score, -float("inf"))

def compute_target_logP(smile, default_value, norm=True):
    train_stats = pickle.load(open("data/molecules/train_stats.pkl", "rb"))
    if not verify_smile(smile):
        return default_value
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

        cycle_score = - cycle_length
        if norm:
            logP_score_normalized = (logP_score - train_stats['logP_mean']) / train_stats['logP_std']
            SAS_score_normalized = (SAS_score - train_stats['SAS_mean']) / train_stats['SAS_std']
            cycle_score_normalized = (cycle_score - train_stats['cycles_mean']) / train_stats['cycles_std']

            return logP_score_normalized + SAS_score_normalized + cycle_score_normalized
        else:
            LOGGER.info(f"LogP: {logP_score}, SAS: {SAS_score}, Cycle: {cycle_score}")
            return logP_score + SAS_score + cycle_score
    except:
        return default_value


def get_black_box_objective_molecules(X, property, dict=None, na_value=-1, is_selfies=True, norm=True):
    # convert from one hot to string

    oracle_values = []
    if is_selfies:
        X = F.one_hot(X.long(), num_classes=len(DEFAULT_SELFIES_VOCAB))
    # else:
    #     X = F.one_hot(X.long(), num_classes=VOCAB_SIZE)
    X = torch.squeeze(X)
    if len(X.shape) == 2:
        X = X.unsqueeze(0)
    print(X.shape)

    for i in range(X.shape[0]):
        x = X[i, ...]
        if is_selfies:
            tokens = x.argmax(dim=-1)
            dec = [DEFAULT_SELFIES_VOCAB[t] for t in tokens]
            stop = dec.index("<stop>") if "<stop>" in dec else None  # want first stop token
            selfie = dec[0:stop]  # cut off stop tokens
            while "<start>" in selfie:
                start = (1 + dec.index("<start>"))
                selfie = selfie[start:]
            selfie = "".join(selfie)
            mol_str = sf.decoder(selfie)
        else:
            mol_str = one_hot_to_eq(x.cpu().detach().numpy(), VOCAB)
            # print(mol_str)

        # print(mol_str)
        # if dict is not None and mol_str in dict:
        #     oracle_values.append(dict[mol_str])
        #     continue
        if property == "logp":
            oracle_values.append(compute_target_logP(mol_str, na_value, norm=norm))
        else:
            scr = smile_to_guacamole_score(property, mol_str)
            if scr is None:
                print("Error: ", mol_str, property)
                scr = na_value

            oracle_values.append(scr)
    return torch.tensor(oracle_values)


def get_molecule_data():
    # read the data
    f_name_pre = "data/molecules/250k_rndm_zinc_drugs_clean"
    f_name_pickle = f"{f_name_pre}.pkl"
    if os.path.exists(f_name_pickle):
        dataset = pickle.load(open(f_name_pickle, "rb"))
    else:
        f_name = f"{f_name_pre}.smi"
        expressions_lst = open(f_name, "r").read().splitlines()
        # convert to one hot
        expressions_lst_oh = [one_hot_encode(e, VOCAB, EXPR_LENGTH) for e in expressions_lst]
        dataset = np.array(expressions_lst_oh)
        pickle.dump(dataset, open(f_name_pickle, "wb"))
    # train_dataset = np.array([one_hot_encode(prods_to_eq([productions[p] for p in prods]), vocab) for prods in rules])
    torch_data =  torch.from_numpy(np.array(dataset))
    # make into long format
    # torch_long = torch.argmax(torch_data, dim=-1).long()
    return torch_data


if __name__ == '__main__':
    smiles  = ["", "CCC", "C", "CC", "C"]
    for smile in smiles:
        log_p_1 = calculate_logP(smile, 1)
        log_p_2 = calculate_logP(smile, 2)
        assert log_p_1 == log_p_2

    X_smiles = np.array([one_hot_encode(smile, VOCAB, EXPR_LENGTH) for smile in smiles])
    X = torch.from_numpy(X_smiles).argmax(dim=-1).float()
    smiles_back = [one_hot_to_eq(F.one_hot(x.long(), num_classes=VOCAB_SIZE).cpu().detach().numpy(), VOCAB) for x in X]
    log_p_1 = get_black_box_objective_molecules(X, "logp", is_selfies=False)

    data = get_molecule_data()
    smiles = [one_hot_to_eq(d.cpu().detach().numpy(), VOCAB) for d in data]
    save_train_stats(smiles, "data/molecules/train_stats.pkl")
    # s = time.time()

    qed = get_black_box_objective_molecules(data, "logp")
    qed_dict = {one_hot_to_eq(oh.cpu().detach().numpy(), VOCAB): float(qed[i].cpu().detach().numpy()) for i, oh in
                enumerate(data)}
    with open("data/molecules/logp_dict.pkl", "wb") as f:
        pickle.dump(qed_dict, f)
    # print(time.time() - s)