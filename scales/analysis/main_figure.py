from functools import partial
import gpytorch
from matplotlib.colors import Normalize
import torch
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from scales.utils.data_utils.utils import get_dataset
from scales.utils.vae import get_vae, DATASETS
from scales.opt.gp import _normalize_grad
from scales.utils.data_utils.expressions import VOCAB as VOCAB_EX
from scales.utils.data_utils.molecules import VOCAB as VOCAB_MOL
from scales.utils.scales import calculate_scales, calculate_scales_derivative
from scales.utils.opt import get_train_test_data
from scales.utils.uncertainty import calculate_uncertainty
from scales.utils.utils import get_mu_sigma, one_hot_to_eq


sns.set_palette("bright")  # You can replace "Set1" with other Seaborn palettes
# sns.set_style("paper")
# set ticks text size to 16
size = 16
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

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def get_expression(x):
    return one_hot_to_eq(x.cpu().numpy(), VOCAB_EX)

def _get_tau(dataset_name):
    if dataset_name == "expressions":
        return 1
    elif dataset_name == "mnist":
        return 1
    elif dataset_name == "selfies":
        return .1
    elif dataset_name == "smiles":
        return 100
    
def get_gp_model(dataset_name):
        (Z, y), _ = get_train_test_data(dataset_name, sample_size=200, true_y=True)
        y_std, y_mean = y.std(), y.mean()
        y = (y - y_mean) / y_std
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SingleTaskGP(
        train_X=Z.float(),
        train_Y=y.unsqueeze(-1).float(),
        # covar_module=covar_module,
        likelihood=likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(Z)
        fit_gpytorch_mll(mll)
        model.eval()
        return model, y.max()

def objective(z, model, best_f):
    # print(f"model = {model}, best_f = {best_f}")
    ei = qLogExpectedImprovement(model=model, best_f=best_f)
    return ei(z)


def scales_valid_pca(dataset_name):
    vae, _  = get_vae(dataset_name)
    dataset = get_dataset(dataset_name)

    mu, sigma = get_mu_sigma(vae)
    tau = _get_tau(dataset_name)
    n = 1000
    x = dataset["train"][0:n]
    z_train = vae.encode(x)
    z_test = vae.encode(dataset["test"][0:n])
    z_small = torch.randn(n, vae.latent_dim).to(device =vae.device, dtype = vae.dtype)
    z_mid = torch.randn(n, vae.latent_dim).to(device =vae.device, dtype = vae.dtype) * 0.1
        # z_mid = torch.randn(n, vae.latent_dim).to(device =vae.device, dtype = vae.dtype) * 2

    # z_large = torch.randn(n, vae.latent_dim).to(device =vae.device, dtype = vae.dtype) * 8
    z = torch.cat([z_small, z_mid, z_train, z_test], dim=0)
    scales = calculate_scales(z, vae, mu, sigma, tau=tau).detach().cpu().numpy()
    # print(f"scales min = {scales.min()}, scales max = {scales.max()}")
    # clip scales between 10th and 90th percentile
    tenth = np.percentile(scales, 10)
    ninetieth = np.percentile(scales, 90)
    scales = np.clip(scales, tenth, ninetieth)
    # normalize scales to be between 0 and 1
    scales = (scales - scales.min()) / (scales.max() - scales.min())
    uc = calculate_uncertainty(z, vae, n_models=10, n_preds=10).detach().cpu().numpy()
    uc = np.clip(uc, -1, 5)
    uc = (uc - uc.min()) / (uc.max() - uc.min())
    valid = vae.get_decoding_quality(z)
    print(f"corr = {np.corrcoef(scales, valid)[0, 1]}, corr = {np.corrcoef(-1 * uc, valid)[0, 1]}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # do pca on z
    pca = PCA(n_components=2)
    pca_fit = pca.fit(z.detach().cpu().numpy())
    # print(z.shape)
    z_pca = pca_fit.transform(z.detach().cpu().numpy())
    # create a grid in the 2d space
    grid_bins = 40
    grid_values = np.linspace(z_pca.min(), z_pca.max(), grid_bins+1)
    scales_grid = []
    valid_grid = []
    for i in range(grid_bins):
        for j in range(grid_bins):
            y_min, y_max = grid_values[i], grid_values[i + 1]
            x_min, x_max = grid_values[j], grid_values[j + 1]
            z_pca_idx_0 = np.logical_and(z_pca[:, 0] > x_min, z_pca[:, 0] < x_max)
            z_pca_idx_1 = np.logical_and(z_pca[:, 1] > y_min, z_pca[:, 1] < y_max)
            idx = np.logical_and(z_pca_idx_0, z_pca_idx_1)
            if idx.sum() < 3:
                scales_grid.append(np.nan)
                valid_grid.append(np.nan)
                continue
            av_scales = scales[idx].mean()
            av_valid = valid[idx].mean()
            scales_grid.append(av_scales)
            valid_grid.append(av_valid)

    scales_grid = np.array(scales_grid).reshape(grid_bins, grid_bins)
    valid_grid = np.array(valid_grid).reshape(grid_bins, grid_bins)
    # valid_grid_norm = Normalize()(valid_grid)

    sns.heatmap(scales_grid, ax=ax[0], cmap="viridis", cbar=False)
    sns.heatmap(valid_grid, ax=ax[1], cmap="viridis", cbar=False)
    # remove all ticks
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    # set y label to be "pca 1"
    ax[0].set_ylabel("PC 2")
    ax[0].set_xlabel("PC 1")
    ax[1].set_xlabel("PC 1")
    ax[0].set_title("ScaLES")
    ax[1].set_title("% Valid Decodings")
    plt.savefig("scales_valid_pca.png")
    return pca_fit



def _plot_trajectory(trajectory_results, pca_fit, ax,color, label):
    trajectory = np.stack(trajectory_results["trajectory"], axis=0)
    expressions_s, expressions_e = trajectory_results["expressions"][0], trajectory_results["expressions"][-1]
    trajectory_pc = []
    for step in range(trajectory.shape[0]):
        # t_step = trajectory[step]
        pc_step = pca_fit.transform(trajectory[step])
        pc_step_top_2 = pc_step[:,:2]
        # print(f"pc_step_top_2 = {pc_step_top_2.shape}")
        trajectory_pc.append(pc_step_top_2)
    # take the first two components
    trajectory_pc = np.stack(trajectory_pc, axis=0)
    num_samples = trajectory_pc.shape[1]
    size_text = 12
    for sample in range(num_samples):
        ax.plot(trajectory_pc[:, sample, 0], trajectory_pc[:, sample, 1], alpha=0.5, color=color, label=label)

        # Quiver plot with default direction (from previous point to next point)
        ax.quiver(trajectory_pc[:-1, sample, 0], trajectory_pc[:-1, sample, 1],
                trajectory_pc[1:, sample, 0] - trajectory_pc[:-1, sample, 0],
                trajectory_pc[1:, sample, 1] - trajectory_pc[:-1, sample, 1],
                color=color, scale_units='xy', angles='xy', scale=1, alpha=0.5)

        # Annotate the start and end points with the expressions in LaTeX
        offset = 0.01  # Adjust offset as needed
        start_text = ax.text(trajectory_pc[0, sample, 0] + offset, trajectory_pc[0, sample, 1] + offset,
                            r'\texttt{{{}}}'.format(expressions_s), fontsize=size_text,
                            bbox=dict(facecolor='grey', alpha=0.2, edgecolor='none', boxstyle='round,pad=0.5'))
        end_color = "green" if trajectory_results["validity"][-1] > 0.5 else "red"
        end_text = ax.text(trajectory_pc[-1, sample, 0] + offset, trajectory_pc[-1, sample, 1] + offset,
                        r'\texttt{{{}}}'.format(expressions_e), fontsize=size_text,
                        bbox=dict(facecolor=color, alpha=0.2, edgecolor='none', boxstyle='round,pad=0.5'))

        # # Calculating new axes limits
        # x_coords = trajectory_pc[:, sample, 0]
        # y_coords = trajectory_pc[:, sample, 1]
        # x_min = min(x_coords.min(), start_text.get_position()[0], end_text.get_position()[0])
        # x_max = max(x_coords.max(), start_text.get_position()[0], end_text.get_position()[0])
        # y_min = min(y_coords.min(), start_text.get_position()[1], end_text.get_position()[1])
        # y_max = max(y_coords.max(), start_text.get_position()[1], end_text.get_position()[1])

        # # Setting new limits with a margin
        # margin = 0.3  # Margin ratio, adjust as needed
        # x_range = x_max - x_min
        # y_range = y_max - y_min
        # ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        # ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

    return ax

def _traverse_latent_space(z_init, reg, n_steps, vae, objective, lr, dataset_name):
    n_samples = z_init.shape[0]
    mu, sigma = get_mu_sigma(vae)
    tau = _get_tau(dataset_name)
    results = {"trajectory": [], "scales": [], "validity": [], "expressions": []}
    z = torch.tensor(z_init)
    for j in range(n_steps):
        # z = 
        if j ==0:
            scales = calculate_scales(z, vae, mu, sigma, tau=tau).detach().cpu().numpy()
            valid = vae.get_decoding_quality(z)
            x_s = vae.decode(z).detach().cpu()
            if n_samples == 1 and len(x_s.shape) == 2:
                x_s = x_s.unsqueeze(0)
            expressions = [get_expression(x_i.transpose(0,1)) for x_i in x_s]
            results["expressions"] += expressions
            results["validity"]+=valid.tolist()
            results["trajectory"].append(z.detach().cpu().numpy())
            results["scales"].append(scales)

        
        z.requires_grad = True
        grad_objective = torch.autograd.grad(objective(z), z)[0].detach()
        grad_objective = _normalize_grad(grad_objective, torch.ones_like(grad_objective).mean(dim=1))
        grad_objective = grad_objective.detach().to(device=z.device, dtype=z.dtype)
        if reg:
            # print(z.shape)
            grad_scales = calculate_scales_derivative(z, vae, mu, sigma, tau=tau)
            grad_scales = _normalize_grad(grad_scales, torch.ones_like(grad_scales).mean(dim=1))
            grad_scales = grad_scales.detach().to(device=z.device, dtype=z.dtype)
            
            grad_sum = grad_objective + grad_scales
            grad_sum = _normalize_grad(grad_sum, torch.ones_like(grad_sum).mean(dim=1))
        else:
            grad_sum = grad_objective
        z = torch.tensor(z + lr * grad_sum)
        scales = calculate_scales(z, vae, mu, sigma, tau=tau).detach().cpu().numpy()
        valid = vae.get_decoding_quality(z)
        x_s = vae.decode(z).detach().cpu()
        if n_samples == 1 and len(x_s.shape) == 2:
                x_s = x_s.unsqueeze(0)
        expressions = [get_expression(x_i.transpose(0,1)) for x_i in x_s]
        results["expressions"] += expressions
        results["validity"]+=valid.tolist()
        results["trajectory"].append(z.detach().cpu().numpy())
        results["scales"].append(scales)
        if j == n_steps - 1:
            print(f"pct valid = {valid.mean()}")
            print(f"expression = {', '.join(expressions)}")
    return results


def plot_main_figure(dataset_name,pca_fit):
    # dataset_name = DATASETS.expressions
    vae, _  = get_vae(dataset_name)
    dataset = get_dataset(dataset_name)
    test_data = dataset["test"]
    mu, sigma = get_mu_sigma(vae)
    tau = _get_tau(dataset_name)
    n = 2000
    x = dataset["train"][0:n]
    z_train = vae.encode(x)
    z_test = vae.encode(dataset["test"][0:n])
    z_small = torch.randn(n, vae.latent_dim).to(device =vae.device, dtype = vae.dtype)
    z_mid = torch.randn(n, vae.latent_dim).to(device =vae.device, dtype = vae.dtype) * 0.1
        # z_mid = torch.randn(n, vae.latent_dim).to(device =vae.device, dtype = vae.dtype) * 2

    # z_large = torch.randn(n, vae.latent_dim).to(device =vae.device, dtype = vae.dtype) * 8
    z = torch.cat([z_small, z_mid, z_train, z_test], dim=0)
    scales = calculate_scales(z, vae, mu, sigma, tau=tau).detach().cpu().numpy()
    # print(f"scales min = {scales.min()}, scales max = {scales.max()}")
    # clip scales between 10th and 90th percentile
    tenth = np.percentile(scales, 10)
    ninetieth = np.percentile(scales, 90)
    scales = np.clip(scales, tenth, ninetieth)
    # normalize scales to be between 0 and 1
    scales = (scales - scales.min()) / (scales.max() - scales.min())
    uc = calculate_uncertainty(z, vae, n_models=10, n_preds=10).detach().cpu().numpy()
    uc = np.clip(uc, -1, 5)
    uc = (uc - uc.min()) / (uc.max() - uc.min())
    valid = vae.get_decoding_quality(z)
    print(f"corr = {np.corrcoef(scales, valid)[0, 1]}, corr = {np.corrcoef(-1 * uc, valid)[0, 1]}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # do pca on z
    pca = PCA(n_components=2)
    pca_fit = pca.fit(z.detach().cpu().numpy())
    # print(z.shape)
    z_pca = pca_fit.transform(z.detach().cpu().numpy())
    # create a grid in the 2d space
    grid_bins = 40
    grid_values = np.linspace(z_pca.min(), z_pca.max(), grid_bins+1)
    scales_grid = []
    valid_grid = []
    for i in range(grid_bins):
        for j in range(grid_bins):
            y_min, y_max = grid_values[i], grid_values[i + 1]
            x_min, x_max = grid_values[j], grid_values[j + 1]
            z_pca_idx_0 = np.logical_and(z_pca[:, 0] > x_min, z_pca[:, 0] < x_max)
            z_pca_idx_1 = np.logical_and(z_pca[:, 1] > y_min, z_pca[:, 1] < y_max)
            idx = np.logical_and(z_pca_idx_0, z_pca_idx_1)
            if idx.sum() < 3:
                scales_grid.append(np.nan)
                valid_grid.append(np.nan)
                continue
            av_scales = scales[idx].mean()
            av_valid = valid[idx].mean()
            scales_grid.append(av_scales)
            valid_grid.append(av_valid)

    scales_grid = np.array(scales_grid).reshape(grid_bins, grid_bins)
    valid_grid = np.array(valid_grid).reshape(grid_bins, grid_bins)
    # valid_grid_norm = Normalize()(valid_grid)

    sns.heatmap(scales_grid, ax=ax[0], cmap="viridis", cbar=False)
    sns.heatmap(valid_grid, ax=ax[1], cmap="viridis", cbar=False)
    # remove all ticks
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    # set y label to be "pca 1"
    ax[0].set_ylabel("PC 2")
    ax[0].set_xlabel("PC 1")
    ax[1].set_xlabel("PC 1")
    ax[0].set_title("ScaLES")
    ax[1].set_title("% Valid Decodings")
    plt.savefig("scales_valid_pca.png")
    return pca_fit



def _plot_trajectory(trajectory_results, pca_fit, ax,color, label):
    trajectory = np.stack(trajectory_results["trajectory"], axis=0)
    expressions_s, expressions_e = trajectory_results["expressions"][0], trajectory_results["expressions"][-1]
    trajectory_pc = []
    for step in range(trajectory.shape[0]):
        # t_step = trajectory[step]
        pc_step = pca_fit.transform(trajectory[step])
        pc_step_top_2 = pc_step[:,:2]
        # print(f"pc_step_top_2 = {pc_step_top_2.shape}")
        trajectory_pc.append(pc_step_top_2)
    # take the first two components
    trajectory_pc = np.stack(trajectory_pc, axis=0)
    num_samples = trajectory_pc.shape[1]
    size_text = 12
    for sample in range(num_samples):
        ax.plot(trajectory_pc[:, sample, 0], trajectory_pc[:, sample, 1], alpha=0.5, color=color, label=label)

        # Quiver plot with default direction (from previous point to next point)
        ax.quiver(trajectory_pc[:-1, sample, 0], trajectory_pc[:-1, sample, 1],
                trajectory_pc[1:, sample, 0] - trajectory_pc[:-1, sample, 0],
                trajectory_pc[1:, sample, 1] - trajectory_pc[:-1, sample, 1],
                color=color, scale_units='xy', angles='xy', scale=1, alpha=0.5)

        # Annotate the start and end points with the expressions in LaTeX
        offset = 0.01  # Adjust offset as needed
        start_text = ax.text(trajectory_pc[0, sample, 0] + offset, trajectory_pc[0, sample, 1] + offset,
                            r'\texttt{{{}}}'.format(expressions_s), fontsize=size_text,
                            bbox=dict(facecolor='grey', alpha=0.2, edgecolor='none', boxstyle='round,pad=0.5'))
        end_color = "green" if trajectory_results["validity"][-1] > 0.5 else "red"
        end_text = ax.text(trajectory_pc[-1, sample, 0] + offset, trajectory_pc[-1, sample, 1] + offset,
                        r'\texttt{{{}}}'.format(expressions_e), fontsize=size_text,
                        bbox=dict(facecolor=color, alpha=0.2, edgecolor='none', boxstyle='round,pad=0.5'))

        # # Calculating new axes limits
        # x_coords = trajectory_pc[:, sample, 0]
        # y_coords = trajectory_pc[:, sample, 1]
        # x_min = min(x_coords.min(), start_text.get_position()[0], end_text.get_position()[0])
        # x_max = max(x_coords.max(), start_text.get_position()[0], end_text.get_position()[0])
        # y_min = min(y_coords.min(), start_text.get_position()[1], end_text.get_position()[1])
        # y_max = max(y_coords.max(), start_text.get_position()[1], end_text.get_position()[1])

        # # Setting new limits with a margin
        # margin = 0.3  # Margin ratio, adjust as needed
        # x_range = x_max - x_min
        # y_range = y_max - y_min
        # ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        # ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

    return ax

def _traverse_latent_space(z_init, reg, n_steps, vae, objective, lr, dataset_name):
    n_samples = z_init.shape[0]
    mu, sigma = get_mu_sigma(vae)
    tau = _get_tau(dataset_name)
    results = {"trajectory": [], "scales": [], "validity": [], "expressions": []}
    z = torch.tensor(z_init)
    for j in range(n_steps):
        # z = 
        if j ==0:
            scales = calculate_scales(z, vae, mu, sigma, tau=tau).detach().cpu().numpy()
            valid = vae.get_decoding_quality(z)
            x_s = vae.decode(z).detach().cpu()
            if n_samples == 1 and len(x_s.shape) == 2:
                x_s = x_s.unsqueeze(0)
            expressions = [get_expression(x_i.transpose(0,1)) for x_i in x_s]
            results["expressions"] += expressions
            results["validity"]+=valid.tolist()
            results["trajectory"].append(z.detach().cpu().numpy())
            results["scales"].append(scales)

        
        z.requires_grad = True
        grad_objective = torch.autograd.grad(objective(z), z)[0].detach()
        grad_objective = _normalize_grad(grad_objective, torch.ones_like(grad_objective).mean(dim=1))
        grad_objective = grad_objective.detach().to(device=z.device, dtype=z.dtype)
        if reg:
            # print(z.shape)
            grad_scales = calculate_scales_derivative(z, vae, mu, sigma, tau=tau)
            grad_scales = _normalize_grad(grad_scales, torch.ones_like(grad_scales).mean(dim=1))
            grad_scales = grad_scales.detach().to(device=z.device, dtype=z.dtype)
            
            grad_sum = grad_objective + grad_scales
            grad_sum = _normalize_grad(grad_sum, torch.ones_like(grad_sum).mean(dim=1))
        else:
            grad_sum = grad_objective
        z = torch.tensor(z + lr * grad_sum)
        scales = calculate_scales(z, vae, mu, sigma, tau=tau).detach().cpu().numpy()
        valid = vae.get_decoding_quality(z)
        x_s = vae.decode(z).detach().cpu()
        if n_samples == 1 and len(x_s.shape) == 2:
                x_s = x_s.unsqueeze(0)
        expressions = [get_expression(x_i.transpose(0,1)) for x_i in x_s]
        results["expressions"] += expressions
        results["validity"]+=valid.tolist()
        results["trajectory"].append(z.detach().cpu().numpy())
        results["scales"].append(scales)
        if j == n_steps - 1:
            print(f"pct valid = {valid.mean()}")
            print(f"expression = {', '.join(expressions)}")
    return results


def plot_main_figure(dataset_name,pca_fit):
    # dataset_name = DATASETS.expressions
    vae, _  = get_vae(dataset_name)
    vae.eval()
    vae.decoder.eval()
    vae.encoder.eval()
    dataset = get_dataset(dataset_name)
    test_data = dataset["test"]

    # sample 500 images from the test set
    n_samples = 1
    idx = np.random.choice(len(test_data), n_samples, replace=False)
    print(idx)
    x = test_data[idx]
    # encode image
    z_init = vae.encode(x) #+ torch.randn_like(vae.encode(x)) * 0.1
    if n_samples == 1 and len(z_init.shape) == 1:
        z_init = z_init.unsqueeze(0)
    
    n_steps = 10
    lr = 0.5

    model, best_f = get_gp_model(dataset_name)
    _objective = partial(objective, model = model, best_f = best_f)
    unreg_trajectory = _traverse_latent_space(z_init, reg=False, n_steps=n_steps, vae=vae, objective=_objective, lr=lr, dataset_name=dataset_name)
    reg_trajectory = _traverse_latent_space(z_init, reg=True, n_steps=n_steps, vae=vae, objective=_objective, lr=lr, dataset_name=dataset_name)
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # ax = axs[0]
    _plot_trajectory(unreg_trajectory, pca_fit, ax, color="red", label="Unregularized")
    # ax.set_title("Regularized")

    # ax = axs[1]
    _plot_trajectory(reg_trajectory, pca_fit, ax, color="blue", label="Regularized")
    # ax.set_title("Unregularized")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    # set x and y lim between 1.5 and -1.5
    l = 1
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)
    ax.set_title("Trajectory in Latent Space")
    ax.legend()

    # set title

    # save
    plt.savefig("trajecotry.png")
    # print(expressions_norm)




if __name__ == '__main__':
    pca_fit = scales_valid_pca(DATASETS.expressions)
    # pca_fit = 0
    plot_main_figure(DATASETS.expressions, pca_fit)