import os
import pickle
import argparse

from functools import partial

import yaml

import numpy as np
import pandas as pd
import seaborn as sns


from matplotlib import pyplot as plt

from scales import LOGGER
from scales.utils.vae import DATASETS, get_vae
from scales.opt.black_box import BlackBoxOptimizerBO
from scales.utils.scales import calculate_scales_derivative, calculate_nll_derivative, calculate_scales,calculate_nll
from scales.utils.opt import get_train_test_data, get_black_box_function
from scales.utils.uncertainty import calculate_uncertainty
from scales.utils.utils import get_mu_sigma

# torch.cuda.empty_cache()
# torch.autograd.set_detect_anomaly(True)
sns.set_palette("bright")  # You can replace "Set1" with other Seaborn palettes
# set ticks text size to 16
size = 26
plt.rcParams.update({'font.size': size})
# set labels text size to 20
plt.rc('axes', labelsize=size)
# set title text size to 20
plt.rc('axes', titlesize=size)
# set legend font size to 16
plt.rc('legend', fontsize=size)



def get_params(dataset_name):
    if dataset_name == "expressions":
        return {"learning_rate": 0.5, "n_steps": 20, "n_batch": 5, "k": 10, "thres": 12, "rho": 0, "n_gp": 500,
                "true_y": False, "normalize_x": False, "tau":1}
    elif dataset_name == "mnist":
        return {"learning_rate": 2, "n_steps": 4, "n_batch": 20, "k": 3, "thres": 30, "rho": 0, "option": 1, "true_y": False, "tau":1}
    elif dataset_name == "selfies":
        return {"learning_rate": 1, "n_steps": 50, "n_batch": 20, "k": 20, "thres": 12, "rho": 0.2, "n_gp": 1000,
                "true_y": True, "tau":.003, "normalize_x": False, "objective": "rano"}
    elif dataset_name == "smiles":
        return {"learning_rate": 0.01, "n_steps": 25, "n_batch": 20, "k": 20, "thres": 30, "rho": 0.2, "n_gp": 500,
                "true_y": False, "normalize_x": False, "tau":0.001}


def get_penalty_func(method, model, option=4, tau=1):
    mu, sigma = get_mu_sigma(model)
    if method == "scales":
        penalty = partial(calculate_scales_derivative, model=model, mu=mu, sigma=sigma,
                          option=option,  thres=12, tau=tau)
    elif method == "nll":
        penalty = partial(calculate_nll_derivative, mu=mu, sigma=sigma)
    else:
        raise NotImplementedError
    return penalty


def _get_es_thresholds(quantile_es, method, model_path):
    train_dist = pickle.load(open(
        os.path.join(model_path, f"{method}_dict.pickle"),
        "rb"))['train']
    if type(train_dist) == dict:
        train_dist = train_dist['score']
    thres = float(np.quantile(train_dist, quantile_es))
    return thres


def get_es_rule(method, model_path, quantile_es=0.95, n_models=10, n_preds=10, tau=1):
    # if true we accept the gradient update
    thres = _get_es_thresholds(quantile_es, method, model_path)
    if method == "scales":
        mu, sigma = get_mu_sigma(model)
        def es_rule(Z):
            return calculate_scales(model=model, X=Z, mu=mu, sigma=sigma, tau=tau).detach().cpu().numpy() > thres
    elif method == "nll":
        def es_rule(Z):
            return calculate_nll(model=model, X=Z, mu=mu, sigma=sigma).detach().cpu().numpy() > thres
    elif method == "uncertainty":
        def es_rule(Z):
            uc = calculate_uncertainty(model=model, X=Z,n_models=n_models, n_preds=n_preds).detach().cpu().numpy()
            # print(f"uncertainty: {uc}, thres: {thres}")
            # LOGGER.info(f"uncertainty: {uc}")
            return uc < thres
    else:
        raise NotImplementedError
    return es_rule


def get_bb_opt(Z, y, model, black_box_function, tau, method_name, dir_name, dataset_name, quant=None):
        pth = dir_name#os.path.join(dir_name, method_name)
        if method_name == "no_reg":
            LOGGER.info("No regularization")
            return BlackBoxOptimizerBO(Z, y, vae=model, es_rule=None, penalty=None, botorch=True,
                                        black_box_function=black_box_function,pth=pth,
                                        dataset=dataset_name, normalize=False, objective=objective)
        elif method_name == "scales":
            LOGGER.info("Using ScaLES regularization")
            penalty = get_penalty_func("scales", model, tau=tau)
            return BlackBoxOptimizerBO(Z, y, vae=model, es_rule=None, penalty=penalty, botorch=False,
                                        black_box_function=black_box_function, pth=pth,
                                        dataset=dataset_name, normalize=False, objective=objective)
        elif method_name == "nll":
            LOGGER.info("Using prior regularization")
            penalty = get_penalty_func("nll", model, tau=tau)
            return BlackBoxOptimizerBO(Z, y, vae=model, es_rule=None, penalty=penalty, botorch=False,
                                        black_box_function=black_box_function, pth=pth,
                                        dataset=dataset_name, normalize=False, objective=objective)
        elif method_name == "uc":
            LOGGER.info("Using uncertainty regularization")
            uc_es = get_es_rule("uncertainty", model_path, quantile_es=quant, n_models=10, n_preds=10)
            return BlackBoxOptimizerBO(Z, y, vae=model, es_rule=uc_es, penalty=None, botorch=False,
                                        black_box_function=black_box_function, pth=pth,
                                        dataset=dataset_name, normalize=False, objective=objective)
        elif method_name == "lbfgs":
            LOGGER.info("Using L-BFGS optimization")
            return BlackBoxOptimizerBO(Z, y, vae=model, es_rule=None, penalty=None, botorch=True,
                                        black_box_function=black_box_function, pth=pth,
                                        dataset=dataset_name, normalize=False, objective=objective)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name, one of [expressions, selfies, smiles]")
    parser.add_argument("--expr_name", type=str, help="experiment name")
    parser.add_argument("--reg_method", type=str, default="scales", help="regularization method, one of [scales, nll, uc, no_reg, lbgfs]")
    parser.add_argument("--reg_param", type=float, default=None, help="regularization parameter")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    model, model_path = get_vae(dataset_name)
    mu, sigma = get_mu_sigma(model)
    dir_name = f"bo_{args.expr_name}/{args.reg_method}_{args.reg_param}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    model.encoder.eval()
    model.encoder.to(device=model.device, dtype=model.dtype)

    # option = 4  # if dataset_name == "selfies" else 2
    params_file = os.path.join(dir_name, "params.yaml")
    if not os.path.isfile(params_file):
        params = get_params(dataset_name)

        with open(params_file, "w") as f:
            yaml.dump(params, f)
    else:
        with open(params_file, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    
    lr = params["learning_rate"]
    n_steps = params["n_steps"]
    n_batch = params["n_batch"]
    k = params["k"]
    tau = params["tau"]
    thres = params["thres"]
    option = params.get("option", 4)
    normalize_x = params.get("normalize_x", True)
    objective = params.get("objective", None)
    n_gp = params.get("n_gp", 500)
    true_y = params.get("true_y", True)
    black_box_function = get_black_box_function(dataset_name, objective=objective)
    (Z, y), (Z_test, y_test) = get_train_test_data(dataset_name, sample_size=n_gp, true_y=true_y, objective=objective)
    y_std, y_mean = y.std(), y.mean()
    y = (y - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    alpha = args.reg_param
    quant = alpha if args.reg_method == "uc" else None
    
    bb_opt = get_bb_opt(Z, y, model, black_box_function, tau, args.reg_method, dir_name, dataset_name, quant)
    bb_opt.optimize(n_batch=n_batch, n_steps=n_steps, alpha=alpha, learning_rate=lr, tau=tau)
    print(bb_opt.results(steps=n_steps, k=10))