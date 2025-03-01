import os
import pickle

import h5py
import nltk
import numpy as np
import torch

from numexpr import evaluate

from numpy import sin, exp

from nltk import CFG, ChartParser
from les.utils.utils import probabilities_to_one_hot, one_hot_encode, one_hot_to_eq

gram = """S -> S '+' T
S -> S '*' T
S -> S '/' T
S -> T
T -> '(' S ')'
T -> 'sin(' S ')'
T -> 'exp(' S ')'
T -> 'x'
T -> '1'
T -> '2'
T -> '3'
Nothing -> None"""
grammar = CFG.fromstring(gram)
VOCAB = ["x", "+", "(", ")", "1", "2", "3", "*", "/", "s", "i", "n", "e", "p", " "]
EOS_TOKEN_IDX = len(VOCAB) - 1
VOCAB_SIZE_EXPRESSION = len(VOCAB)
VOCAB = {v: i for i, v in enumerate(VOCAB)}
EXPR_LENGTH_EXPRESSION = 19


def save_black_box_data():
    dataset = get_expressions_data()
    objectives = get_black_box_objective_expression(dataset, black_box_dict={})
    expressions = [one_hot_to_eq(d.numpy(), VOCAB) for d in dataset]
    bbdata = {e: o.numpy() for e, o in zip(expressions, objectives)}
    pickle.dump(bbdata, open("data/grammer/black_box_data.pickle", "wb"))


def get_black_box_objective_expression(
    X, black_box_dict, true_eq="1/3 + x + sin(x * x)"
):
    mses = []

    true_vals = eval_eq(true_eq, {})
    # replace inf with 1e10
    # true_vals[true_vals == float('inf')] = 1e+10
    na_val = -100

    if len(X.shape) == 2:
        X = X.unsqueeze(0)

    for i in range(X.shape[0]):
        eq = probabilities_to_one_hot(X[i, ...])
        # convert from one hot to string
        eq = one_hot_to_eq(eq.cpu().numpy(), VOCAB)
        if not is_valid_expression(eq):
            mses.append(np.nan)
            continue
        if eq in black_box_dict and False:
            mses.append(black_box_dict[eq])
            continue
        try:
            vals = eval_eq(eq, black_box_dict)
            # replace inf with 1e10
            # vals[vals == float('inf')] = 1e+10
            mse = np.minimum(1000, np.mean((vals - true_vals) ** 2))
            mses.append(-np.log(1 + mse))
        except Exception as e:
            mses.append(np.nan)
            continue

    mse = torch.from_numpy(np.array(mses))
    return mse  # .to(dtype=DTYPE, device=DEVICE)


def eval_eq(eq, bb_dict):
    # remove SOS and EOF from eq
    eq = eq.replace("SOS", "").replace("EOF", "")
    if eq.strip() in bb_dict and False:
        return bb_dict[eq.strip()]
    grid = np.linspace(-10, 10, 1000)

    # Replace 'x' with 'grid' in the equation
    if "x" in eq:
        eq = eq.replace("x", "grid")
        eq = eq.replace("egridp", "exp")

    # Evaluate the expression using numexpr for faster computation
    true_vals = evaluate(eq)
    return true_vals


def _add_spaces(eq):
    eq = eq.replace("(", " ( ").replace(")", " ) ")
    eq = eq.replace("+", " + ").replace("*", " * ").replace("/", " / ")
    eq = eq.replace("sin (", "sin(").replace("exp (", "exp(")
    # eq = eq.replace("sin", " sin ").replace("exp", " exp ")
    eq = eq.replace("  ", " ")
    return eq


def is_valid_expression(expression, grammar=grammar):
    # remove SOS and EOF from expression
    # expression = expression.replace("SOS", "").replace("EOF", "")
    parser = ChartParser(grammar)
    try:
        for _ in parser.parse(_add_spaces(expression).split()):
            return True
    except ValueError:
        return False
    return False


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == "Nothing":
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1 :]
                break

    try:
        return "".join(seq)
    except:
        return ""


def load_data(data_path):
    """Returns the h5 dataset as numpy array"""
    f = h5py.File(data_path, "r")
    return f["data"][:]


def get_expressions_data(n=None, seed=42):
    data_path = os.path.join(
        "data",
        "grammer",
        "eq2_grammar_dataset.h5",
    )
    try:
        data = load_data(data_path)
    except Exception:
        raise Exception(
            f"File {data_path} not found, please make sure to download it from git"
        )
    # Turn it into a float32 PyTorch Tensor
    train_dataset = torch.from_numpy(data).float()
    GCFG = nltk.CFG.fromstring(gram)
    productions = GCFG.productions()
    # print(is_valid_expression("x + sin( x "))
    rules = np.where(train_dataset == 1)[2].reshape(-1, 15)

    train_dataset = []
    for prods in rules:
        eq = prods_to_eq([productions[p] for p in prods])
        oh_eq = one_hot_encode(eq, VOCAB, EXPR_LENGTH_EXPRESSION)
        assert eq == one_hot_to_eq(oh_eq, VOCAB)
        train_dataset.append(oh_eq)
    train_dataset = np.array(train_dataset)
    torch_df = torch.from_numpy(np.array(train_dataset)).float()
    if n is not None:
        # sample n random indices
        np.random.seed(seed)
        ind = np.random.randint(0, len(torch_df), n)
        torch_df = torch_df[ind]
    return torch_df


if __name__ == "__main__":
    save_black_box_data()
