import os
import time
from sklearn.metrics import auc, roc_curve
import torch
import pickle
import logging
import numpy as np
import seaborn as sns

from functools import partial

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

from ..utils.data_utils.utils import get_dataset
from ..utils.vae import get_vae
from ..utils.scales import calculate_scales, calculate_nll
from ..utils.uncertainty import calculate_uncertainty
from ..utils.utils import get_mu_sigma

sns.set_palette("bright")  # You can replace "Set1" with other Seaborn palettes
# sns.set_style("paper")
# set ticks text size to 16
size = 34
plt.rcParams.update({'font.size': size})
# set labels text size to 20
plt.rc('axes', labelsize=size)
# set title text size to 20
plt.rc('axes', titlesize=size)
# set legend font size to 16
plt.rc('legend', fontsize=size)
# set ticks font size to 16
plt.rc('xtick', labelsize=int(0.8 * size))
plt.rc('ytick', labelsize=int(0.8 * size))
# set title size
plt.rc('axes', titlesize=size)


LOGGER = logging.getLogger("OOD")
logging.basicConfig(level=logging.DEBUG)

# Hyperparameters
def _get_fn_score(method, mu=None, sigma=None, dataset=None, tau=1):

    if method == "scales":
        return partial(calculate_scales, mu=mu, sigma=sigma, use_probs=True, tau=tau)
    elif method == "polarity":
        return partial(calculate_scales, mu=mu, sigma=sigma, use_probs=False, tau=tau)
    elif method == "nll":
        return partial(calculate_nll, mu=mu, sigma=sigma)
    elif method == "uncertainty":
        return partial(calculate_uncertainty, dataset=dataset)
    else:
        raise ValueError(f"method {method} is not supported")


def latent_sample(n, std, data=None, model=None):
    if data is not None and model is not None:
        model.encoder.eval()
        data = data.to(device = model.device)
        return model.encode(data, add_noise=False)

    torch.manual_seed(42)
    z = torch.randn(n, std.shape[0]).to(std.device)
    return z * std

def compare_image_quaility(model, dir_name, train_data, test_data, n_hist, dataset):
    if not os.path.exists(dir_name):
        # print(f"dir created {dir_name}")
        os.makedirs(dir_name)
    hidden_size = 2#model.hidden_size

    sigma = torch.ones(hidden_size)
    mu = torch.zeros(hidden_size)
    uncertainty_fn = _get_fn_score("uncertainty", dataset=dataset)
    density_fn = _get_fn_score("scales", mu, torch.diag(sigma), dataset=dataset)

    # set random seed
    np.random.seed(42)
    train_sample = train_data[np.random.choice(train_data.shape[0], n_hist, replace=True)]
    np.random.seed(41)
    test_sample = test_data[np.random.choice(test_data.shape[0], n_hist, replace=True)]
    results = {}
    for sample, data, std in [("prior", None, 1)]:
        z = latent_sample(n_hist, sigma * std, data=data, model=model)

        energy_score = density_fn(model=model, X=z.to(model.device)).cpu().detach().numpy()
        # clamp energy score
        # energy_score = np.clip(energy_score, -10, 100)
        uncertainty_score = uncertainty_fn(model=model, X=z.to(model.device)).cpu().detach().numpy()
        quality_vec_probs = model.get_decoding_quality(z).detach().cpu().numpy()
        quality_vec = np.array(quality_vec_probs >= 0.1, dtype=np.float32)
        LOGGER.debug(f"sample {sample} quality {quality_vec.mean()}\n"
                    f"uncertainty correlation {np.corrcoef(quality_vec, uncertainty_score)[0,1]}, "
                    f"energy correlation {np.corrcoef(quality_vec, energy_score)[0,1]}")
        low_quality = quality_vec == 0
        energy_mean = energy_score.mean()
        uncertainty_mean = uncertainty_score.mean()
        fig, axs = plt.subplots(2, 3, figsize=(30,20))
        # take 4 lowest quality samples
        low_quality_idx = np.argsort(quality_vec_probs)[:3]
        for i, idx in enumerate(low_quality_idx):
            ax = axs[0, i]
            x = model.decode(z[idx].unsqueeze(0)).detach().cpu()
            x = torch.tensor(x.view(28, 28, 2))[..., 0].numpy()
            # remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(x, cmap="gray")
            ax.set_title(f"Energy {energy_score[idx]:.1f}, Uncertainty {uncertainty_score[idx]:.1f}")
        # take 4 highes quality samples
        high_quality_idx = np.argsort(quality_vec_probs)[-3:]
        for i, idx in enumerate(high_quality_idx):
            ax = axs[1, i]
            x = model.decode(z[idx].unsqueeze(0)).detach().cpu()
            x = torch.tensor(x.view(28, 28, 2))[..., 0].numpy()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(x, cmap="gray")
            ax.set_title(f"Energy {energy_score[idx]:.1f}, Uncertainty {uncertainty_score[idx]:.1f}")
        plt.tight_layout()
        plt.savefig(os.path.join(dir_name, f"quality.png"))
        plt.close()


def get_plot_data(model, dir_name, method_scoring, train_data, test_data, n_hist, dataset, tau=1):
    # dir_name = os.path.dirname(fig_name)
    if not os.path.exists(dir_name):
        # print(f"dir created {dir_name}")
        os.makedirs(dir_name)
    f_name_dict = os.path.join(dir_name, f"{method_scoring}_dict.pickle")
    mu, sigma = get_mu_sigma(model)
    if method_scoring in ["scales","nll", "uncertainty", "polarity"]:
        fn_method = _get_fn_score(method_scoring, mu, sigma, dataset=dataset, tau=tau)
    else:
        fn_method = _get_fn_score(method_scoring, dataset=dataset)
    # get_dist = _get_dist_fn(method_scoring)
    if not os.path.isfile(f_name_dict):
        sigma = sigma[np.arange(sigma.shape[0]), np.arange(sigma.shape[0])] * model.eps_std
        # set random seed
        np.random.seed(42)
        train_sample = train_data[np.random.choice(train_data.shape[0], n_hist, replace=True)]
        np.random.seed(41)
        test_sample = test_data[np.random.choice(test_data.shape[0], n_hist, replace=True)]
        results = {}
        for sample, data, std in [("ood", None, 20),("train", train_sample, 1), ("test", test_sample, 1),
                                   ("prior", None, 1)]:
            z = latent_sample(n_hist, sigma * std, data=data, model=model)
            s = time.time()
            score = fn_method(model=model.decoder, X=z.to(model.device)).cpu().detach().numpy()
            LOGGER.debug(f"Calculating {method_scoring} for n={n_hist} took {time.time() - s} seconds")
            if dataset == "selfies":
                quality, smiles = model.get_decoding_quality(z, return_smiles=True)
                results[sample] = {"score": score, "quality": quality, "smiles": smiles}
            else:
                quality = model.get_decoding_quality(z)
                if type(quality) == torch.Tensor:
                    quality = quality.cpu().detach().numpy()
                # print(quality)
                results[sample] = {"score": score, "quality": quality}

            LOGGER.debug(f"sample {sample} quality {quality.mean()}")
            quality = model.get_decoding_quality(z)
            LOGGER.debug(f"sample rep {sample} quality {quality.mean()}")

            # results[sample] = {"score": score, "quality": quality}

        pickle.dump(results, open(f_name_dict, "wb"))
    # load data from pickle file
    results = pickle.load(open(f_name_dict, "rb"))
    return results


def plot_results(results, method_scoring, dir_name):
    fig, ax = plt.subplots(1, 1, figsize=(11,11))
    fig_name = os.path.join(dir_name, f"{method_scoring}.png")
    samples = ["train", "test", "prior", "ood"]
    min_val = results["ood"]["score"].min()
    max_val = results["train"]["score"].max()
    if method_scoring == "uncertainty":
        min_val = results["train"]["score"].min()
        max_val = results["ood"]["score"].max()
    if method_scoring == "scales":
        min_val = -10
    n_bins = 75
    bins = np.linspace(min_val, max_val, n_bins)

    # plt.figure()
    for i, sample in enumerate(samples):
        scores = results[sample]["score"]
        # remove non finite values and nans
        # scores = scores[np.isfinite(scores)]
        # scores[~np.isfinite(scores)] = np.sign(scores[~np.isfinite(scores)]) * scores[np.isfinite(scores)].max()
        # scores = scores[~np.isnan(scores)]
        # log_score = np.sign(scores) * np.log(np.abs(scores)  + 1)
        # clip scores at -100

        scores = np.clip(scores, min_val, max_val)
        # bins = np.arange(-50, scores.max(), 1)
        ax.hist(scores, label=sample, density=True, alpha=0.7, bins=bins)
        # sns.kdeplot(scores, label=sample, ax=ax, linewidth=2)

    # ax.set_xscale("symlog")
    ax.legend()
    x_label = method_scoring.capitalize() if method_scoring != "scales" else "Energy"
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="OOD detection")
    parser.add_argument("--dataset", type=str, default="expressions", help="Dataset to use")
    # add output directory
    parser.add_argument("--output_dir", type=str, default="results/valid", help="Output directory")
    return parser.parse_args()

def _get_tau(dataset_name):
    if dataset_name == "mnist":
        return 0.5
    elif dataset_name == "expressions":
        return 1
    elif dataset_name == "selfies":
        return 0.003
    elif dataset_name == "smiles":
        return 0.001
    else:
        raise ValueError(f"dataset {dataset_name} is not supported")

if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    output_dir = args.output_dir
    model, model_path = get_vae(dataset_name)
    dataset = get_dataset(dataset_name)
    train_data = dataset["train"]
    test_data = dataset["test"]
    ood_data = get_dataset(dataset_name,digit=0)["test"]
    n_hist = 500
    tau  = _get_tau(dataset_name)
    dir_name = output_dir
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    methods = ["scales", "uncertainty", "nll"]

    data_valid = {m: [] for m in methods}
    quants = np.arange(0, 1, 0.05)
    scores_uc_scales = {"Scales": [], "UC": [], "Valid":[]}
    for method in methods:
        results = get_plot_data(model, dir_name, method, train_data, ood_data, n_hist=n_hist, dataset=dataset_name, tau=tau)
        plot_results(results, method, dir_name)
        samples = ["test", "train" , "prior", "ood"]
        quality_vec = np.concatenate([results[sample]["quality"] for sample in samples])
        # print(quality_vec.mean())
        scores_vec = np.concatenate([results[sample]["score"] for sample in samples])
        if method == "scales":
            print(scores_vec)
            scores_uc_scales["Scales"] = scores_vec
            scores_uc_scales['Valid'] = quality_vec
        elif method == "uncertainty":
            scores_uc_scales["UC"] = scores_vec
        if method == "uncertainty":
            scores_vec = -scores_vec
            nan_idx = np.isnan(scores_vec)
            # remove nans
            quality_vec = quality_vec[~nan_idx]
            scores_vec = scores_vec[~nan_idx]
        if len(np.unique(quality_vec)) > 3:
            quality_vec = np.array(quality_vec >= 0.5, dtype=np.float32)
        # quants = np.arange(0, 1, 0.05)
        quantiles = np.quantile(scores_vec, quants)
        # print(quantiles)
        pct_valid_vec = [np.nanmean(quality_vec[scores_vec >= q]) for q in quantiles]
        # data_valid[method] = pct_valid_vec
        data_valid[method] = {"quality": quality_vec, "scores": scores_vec, "pct_valid": pct_valid_vec}
    # do a scatter plot of the uncertainty vs the scales
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    scales_log_scale = scores_uc_scales["Scales"]
    if dataset_name == "selfies":
        # clip the min of scores scales to be -100
        scales_log_scale = np.clip(scores_uc_scales["Scales"], -100, 100)
    # else:
    #     scales_log_scale = scores_uc_scales["Scales"]
    # scales_log_scale = scores_uc_scales["Scales"]#np.log(scores_uc_scales["Scales"] - scores_uc_scales["Scales"].min() + 1)
    uc_log_scale = scores_uc_scales["UC"]#np.log(scores_uc_scales["UC"] - scores_uc_scales["UC"].min() + 1)
    scales_norm = (scales_log_scale - scales_log_scale.min()) / (scales_log_scale.max() - scales_log_scale.min())
    uc_norm = (uc_log_scale - uc_log_scale.min()) / (uc_log_scale.max() - uc_log_scale.min())
    # ax.scatter(scales_norm, uc_norm)
    # scatter scales vs uc and color by validity
    ax.scatter(scales_norm, uc_norm, c=scores_uc_scales["Valid"], cmap="coolwarm")
    ax.set_xlabel("ScaLES")
    ax.set_ylabel("Uncertainty")
    # # add legend for valid and invalid, blue is valid and red invalid
    # # make the legend color be blud and red from "coolwarm" cmap
    # # Create a color map
    if dataset_name == "expressions":
        cmap = plt.cm.coolwarm

        # Create legend handles manually
        valid_patch = mpatches.Patch(color=cmap(0.0), label='Valid')  # Blue for Valid
        invalid_patch = mpatches.Patch(color=cmap(1.0), label='Invalid')  # Red for Invalid

        # Add legend to the plot
        ax.legend(handles=[valid_patch, invalid_patch])
    # make title upper case if not expressions else capatalize
    ttl = dataset_name.capitalize() if dataset_name == "expressions" else dataset_name.upper()
    # make title dataset name
    ax.set_title(ttl)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f"scales_uc.png"))
    methods_dict = {"nll": "Prior", "scales": "ScaLES", "uncertainty": "Uncertainty (minus)",
                    "polarity": "Polarity", "entropy": "Entropy"}


    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    for method in methods:
        # plot roc curve with auc. the labels are the quality and the predictions are the scores
        y_test = data_valid[method]["quality"]
        y_score = data_valid[method]["scores"]
        # nan_idx = np.isnan(y_score)
        # y_test = y_test[~nan_idx]
        # y_score = y_score[~nan_idx]
        # clip scores at -100
        y_score = np.clip(y_score, -1e+6, 1e+6)
        LOGGER.debug("y_score min %f, max %f", y_score.min(), y_score.max())

        # normalize scores to be between 0 and 1
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
        # make y_true into binary
        y_test = np.array(y_test >= 0.5, dtype=np.float32)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{methods_dict[method]} (AUC = {roc_auc:.2f})", linewidth=5)
        # ax.plot(100*(1-quants), data_valid[method], label=methods_dict[method], linewidth=5)
    # add legend at the buttom right
    ax.legend(loc="lower right")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_xlabel("% of samples kept based on scores")
    # if dataset_name == "mnist":
    #     ax.set_ylabel("Proportion of valid digits")
    # elif dataset_name == "expressions":
    #     ax.set_ylabel("Proportion of valid expressions")
    # elif dataset_name == "selfies":
    #     ax.set_ylabel("Proportion of valid molecules")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f"valid.png"))
