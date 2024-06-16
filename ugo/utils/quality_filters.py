import json
from os import path
import time
import typing
import random
import sys
import itertools
import warnings
import numpy as np
import tqdm
from lazy import lazy
import pandas as pd

import multiprocessing as mp
from multiprocessing import Pool

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.info')

from . import rd_filters

THIS_FILE_DIR = path.dirname(__file__)


class QualityFiltersCheck:
    """
    These are the Quality Filters proposed in the GuacaMol paper, which try to rule out " compounds which are
     potentially unstable, reactive, laborious to synthesize, or simply unpleasant to the eye of medicinal chemists."
    The filter rules are from the GuacaMol supplementary material: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839
    The filter code is from: https://github.com/PatWalters/rd_filters
    Parts of the code below have been taken from the script in this module. This code put in this
     class came with this MIT Licence:
        MIT License
        Copyright (c) 2018 Patrick Walters
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """

    def __init__(self, training_data_smi: typing.List[str]):
        alert_file_name = path.join(THIS_FILE_DIR, 'rd_filters_data/alert_collection.csv')
        self.rf = rd_filters.RDFilters(alert_file_name)

        rules_file_path = path.join(THIS_FILE_DIR, 'rd_filters_data/rules.json')
        rule_dict = rd_filters.read_rules(rules_file_path)
        rule_list = [x.replace("Rule_", "") for x in rule_dict.keys() if x.startswith("Rule") and rule_dict[x]]
        rule_str = " and ".join(rule_list)
        print(f"Using alerts from {rule_str}", file=sys.stderr)
        self.rf.build_rule_list(rule_list)
        self.rule_dict = rule_dict

        self.training_data_smi = training_data_smi

    @lazy
    def _training_data_prop(self):
        training_data_quality_filters = self.call_on_smiles_no_normalization(self.training_data_smi)
        print(f"Training data filters returned {training_data_quality_filters}. Rest normalized on this.")
        return training_data_quality_filters

    def call_on_smiles_no_normalization(self, smiles: typing.List[str]):
        num_cores = 10
        print(f"using {num_cores} cores", file=sys.stderr)
        start_time = time.time()
        p = Pool(mp.cpu_count())

        num_smiles_in = len(smiles)
        input_data = [(smi, f"MOL_{i}") for i, smi in enumerate(smiles)]

        res = list(p.map(self.rf.evaluate, input_data))
        df = pd.DataFrame(res, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"])

        df_ok = df[
            (df.FILTER == "OK") &
            df.MW.between(*self.rule_dict["MW"]) &
            df.LogP.between(*self.rule_dict["LogP"]) &
            df.HBD.between(*self.rule_dict["HBD"]) &
            df.HBA.between(*self.rule_dict["HBA"]) &
            df.TPSA.between(*self.rule_dict["TPSA"]) &
            df.Rot.between(*self.rule_dict["Rot"])
            ]

        num_input_rows = df.shape[0]
        num_output_rows = df_ok.shape[0]
        fraction_passed = "{:.1f}".format(num_output_rows / num_input_rows * 100.0)
        print(f"{num_output_rows} of {num_input_rows} passed filters {fraction_passed}%", file=sys.stderr)
        elapsed_time = "{:.2f}".format(time.time() - start_time)
        print(f"Elapsed time {elapsed_time} seconds", file=sys.stderr)
        p.close()
        return (num_output_rows / num_smiles_in)

    def check_smiles_pass_quality_filters_flag(self, smiles: typing.List[str]):
        # num_cores = 10
        # print(f"using {num_cores} cores", file=sys.stderr)
        # start_time = time.time()
        # p = Pool(mp.cpu_count())
        #
        # num_smiles_in = len(smiles)
        # make nan in smiles empty string
        smiles = [x if x is not np.nan else "" for x in smiles]
        input_data = [(smi, f"MOL_{i}") for i, smi in enumerate(smiles)]

        # res = list(p.map(self.rf.evaluate, input_data))
        res = [self.rf.evaluate(x) for x in input_data]
        df = pd.DataFrame(res, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"])
        quality =  np.array(((df.FILTER == "OK") &
                         df.MW.between(*self.rule_dict["MW"]) &
                         df.LogP.between(*self.rule_dict["LogP"]) &
                         df.HBD.between(*self.rule_dict["HBD"]) &
                         df.HBA.between(*self.rule_dict["HBA"]) &
                         df.TPSA.between(*self.rule_dict["TPSA"]) #&
                         # df.Rot.between(*self.rule_dict["Rot"])
                         ).astype(int)
                        ).squeeze()
        # make quality 0 if smiles is empty string
        quality = quality * (np.array(smiles) != "").astype(int)
        return quality



def pass_quality_filter(smiles):
    qf = QualityFiltersCheck(training_data_smi=[])
    return qf.check_smiles_pass_quality_filters_flag(smiles)


def permute_string(input_string):
    """
    Permute (shuffle) the characters in a string.

    Parameters:
    - input_string (str): The input string to permute.

    Returns:
    - str: The permuted string.
    """
    # Convert the string to a list of characters for shuffling
    string_list = list(input_string)

    # Shuffle the list of characters
    random.shuffle(string_list)

    # Convert the list back to a string
    permuted_string = ''.join(string_list)

    return permuted_string

if __name__ == '__main__':


    pass_quality_filter(["", "CCC", "C", "CC", "C"])
    def perturb_string(str):
        # perturb the string
        perm = np.random.permutation(len(str))
        str = list(str)
        str[perm[0]] = 'C'

    smiles_train = open("data/molecules/guacamol_v1_test.smiles").readlines()
    smiles_train = [x.strip() for x in smiles_train[0:1000]]
    qf = QualityFiltersCheck(training_data_smi=smiles_train)
    print(qf.call_on_smiles_no_normalization(smiles_train))
    # pass

    smiles_train = [permute_string(x.strip()) for x in smiles_train[0:1000]]
    qf = QualityFiltersCheck(training_data_smi=smiles_train)
    print(qf.call_on_smiles_no_normalization(smiles_train))
