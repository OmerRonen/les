import argparse
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
import seaborn as sns

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


# plt.rc('axes', titlesize=size)
NO_RHO_VAL = 1e+10
METHOD_NAME = "ScaLES"
LOW_VAL = -1e+10
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr_name", type=str)
    parser.add_argument("--rho", type=float, default=None)
    # parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    return args


def _get_sol_stats(sol, ks=[1, 5, 10, 20]):
    stats = {}
    stats['valid'] = np.nanmean(sol["quality"] > 0)
    stats['density'] = np.nanmean(sol["density"])
    stats['bb_diff'] = np.nanmean(sol["bb_diff"])
    # stats['objective'] = np.mean(sol["objective_vals"])

    sol = pd.DataFrame({"bb": sol["bb_vals"], "quality": sol["quality"]})
    # get unique bb values
    sol = sol.drop_duplicates(subset="bb")
    # print(f"Number of unique bb values: {len(sol)}")
    # sort by bb
    sol = sol.sort_values(by="bb", ascending=False)
    sol_valid = sol[sol["quality"] > 0]
    sol_valid = sol_valid.sort_values(by="bb", ascending=False)
    for k in ks:

        # calc the average of the top k bb values
        top_k_bb = np.mean(sol.head(k)["bb"])# / k
        top_k_bb_valid = np.mean(sol_valid.head(k)["bb"])# / k

        stats[f"top_{k}"] = top_k_bb
        if len(sol_valid) < k:
            stats[f"top_{k}_valid"] = np.nan
            continue
        stats[f"top_{k}_valid"] = top_k_bb_valid

    return stats


def _parse_method_name(method):
    if method == "no_reg":
        return "no_reg", 0, 0
    if "density" in method or "scales" in method:
        try:
            return "scales", float(method.split("_")[-1]), float(method.split("_")[-3])
        except ValueError:
            return "scales", float(method.split("_")[-1]), _get_rho(None)
    elif "nll" in method:
        return "prior", float(method.split("_")[-1]), 0
    elif "prior" in method:
        return "prior", float(method.split("_")[-1]), 0
    elif "es" in method:
        return "uc", float(method.split("_")[-1]), 0
    elif "random_init" in method:
        return "bound_random_init", float(method.split("_")[-1]), 0
    elif "torch" in method:
        return "bound", float(method.split("_")[-1]), 0
    elif "bound" in method:
        return "lbgfs", float(method.split("_")[-1]), 0
    elif "lbgfs" in method:
        return "lbgfs", float(method.split("_")[-1]), 0
    elif "uc" in method:
        return "uc", float(method.split("_")[-1]), 0


def _get_plot_name(method, rho):
    if method == "no_reg":
        return "No Reg"
    elif "energy" in method and rho == 0:
        return "Prior"
    if "energy" in method and (rho != NO_RHO_VAL and rho != 0):
        return f"{METHOD_NAME} $(\\rho={rho})$"
    elif "energy" in method and rho == NO_RHO_VAL:
        return METHOD_NAME
    elif method == "prior":
        return "Prior"
    elif method == "es":
        return "UC"
    elif method == "torch":
        return "L-BFGS"
    elif method == "bound":
        return "L-BFGS"
    else:
        return f"Energy $(\\rho={rho})$"


def analyze_results(expr_path, steps, ks=[1, 5, 10, 20], mi=False):
    # print(os.listdir(expr_path))
    methods = [d for d in os.listdir(expr_path) if os.path.isdir(os.path.join(expr_path, d))]
    results = {"reg method": [], "reg param": [], "validity": [], "density": [],
               "bb_diff": [], "rho": [], "run": []}
    for k in ks:
        # results[f"top {k}"] = []
        results[f"top {k} (valid)"] = []
    for method in methods:
        runs = [d for d in os.listdir(os.path.join(expr_path, method)) if
                os.path.isdir(os.path.join(expr_path, method, d))]
        for run in runs:
            run_dir = os.path.join(expr_path, method, run)
            if steps is not None:
                sol_files = [f"sol_{i}.pickle" for i in range(steps) if f"sol_{i}.pickle" in os.listdir(run_dir)]
                # print(f"sol files: {len(sol_files)}, steps: {steps}")
                if len(sol_files) < steps:
                    continue
                assert len(sol_files) == steps
            else:
                sol_files = os.listdir(run_dir)
            success = True
            sol = {"bb_vals": [], "quality": [], "density": [],
                   "bb_diff": []}
            # print(f"number of sol files: {len(sol_files)}")
            for sol_file in sol_files:
                if not sol_file.endswith(".pickle"):
                    continue
                try:
                    sol_i = pickle.load(open(os.path.join(run_dir, sol_file), "rb"))
                except EOFError:
                    success = False
                    break
                # try:
                bb_vals_i = np.squeeze(sol_i["bb_vals"])
                quality_i = np.squeeze(sol_i["quality"])
                density_i = np.squeeze(sol_i["density"])
                is_arr = len(bb_vals_i.shape) > 0
                # print(f"bb_vals_i: {bb_vals_i}, type {type(bb_vals_i)}, shape {bb_vals_i.shape}")
                sol["bb_vals"] += bb_vals_i.tolist() if is_arr else [bb_vals_i]
                sol["quality"] += quality_i.tolist() if is_arr else [quality_i]
                sol["density"] += density_i.tolist() if is_arr else [density_i]
                # except TypeError:
                #     continue
                # sol["objective_vals"] += np.squeeze(sol_i["objective_vals"]).tolist()
                if "bb_diff" in sol_i:
                    bb_diff_i = np.squeeze(sol_i["bb_diff"])
                    sol["bb_diff"] += bb_diff_i.tolist() if is_arr else [bb_diff_i]
                else:
                    if is_arr:  # add zeros
                        sol["bb_diff"] += [0] * len(bb_vals_i)
                    else:
                        sol["bb_diff"] += [0]
            if not success:
                continue
            sol = pd.DataFrame(sol)
            # print(f"Number of unique bb values: {len(sol)}")
            stats = _get_sol_stats(sol, ks)
            # print(f"method: {method}")
            # reg_method, reg_param, rho = _parse_method_name(method)
            results["reg method"].append(method)
            results["reg param"].append(0)
            results["rho"].append(0)
            results["validity"].append(stats["valid"])
            results["density"].append(stats["density"])
            results['bb_diff'].append(np.mean(sol["bb_diff"]))
            results['run'].append(run)
            # results["objective"].append(stats["objective"])
            for k in ks:
                # results[f"top {k}"].append(stats[f"top_{k}"])
                results[f"top {k} (valid)"].append(stats[f"top_{k}_valid"])
    results_df = pd.DataFrame(results)
    # average the results across runs and add std cols
    # remove the run column
    results_df = results_df.drop(columns=["run"])
    results_df = results_df.groupby(["reg method", "reg param", "rho"]).agg(
        [mean, sem])
    
    if  mi:
        return results_df
    
    results_df.columns = ['_'.join(col).strip() for col in results_df.columns.values]

    # Optional: Reset the index if you want 'reg method', 'reg param', 'rho' as columns rather than an index
    results_df.reset_index(inplace=True)
    # remove the mult
    # replace the column median with mean in cols that don't have std
    return results_df


def sem(x):
    # remove outlier from x
    # out_idx = np.logical_or(x > 1000 * np.nanmedian(x),x < -1000 * np.abs(np.nanmedian(x)))
    # x = x[~out_idx]
    # remove x values that are less than -1000
    # x = x[x > -1000]
    n = np.sum(~np.isnan(x))
    return np.nanstd(x) / np.sqrt(n)


def mean(x):
    # x = x[x > -1000]
    # out_idx = np.logical_or(x > 1000 * np.nanmedian(x), x < -1000 * np.abs(np.nanmedian(x)))
    # x = x[~out_idx]
    return np.mean(x)


def plot_top_vals_per_step(steps, methods, f_name, ks):
    # take only the energy with alpha
    plt_data = []
    steps_range = np.arange(1, steps+11, 10)
    # steps_range = np.arange(1, steps+1, 1)
    # print(steps_range)
    # steps_range[0] = 1
    for step in steps_range:
        step = min(step, steps)
        # print(step)
        df = analyze_results(expr_path=os.path.join(results_path, expr_name, "bo"), steps=step, ks=ks)

        for method in methods:
            reg_method, reg_param, rho = _parse_method_name(method)
            k = (reg_method, reg_param, rho)
            print(reg_method)
            plot_name = _get_plot_name(reg_method, rho)

            if reg_method == "prior" and reg_method not in df["reg method"].values:
                reg_method = "energy"
            idx_k = np.where(np.logical_and(np.logical_and(df.loc[:, 'reg method'] == reg_method,  df.loc[:, 'reg param'] == reg_param) , df.rho == rho))[0][0]
            # if k not in df.index:
            #     continue
            df_method = df.loc[idx_k]

            for k in ks:
                plt_data.append({"step": step, "top": k, "mean": df_method[f"top {k} (valid)_mean"], "rho": rho,
                                 "std": df_method[f"top {k} (valid)_sem"], "method": plot_name})
        if step == steps:
            break
    plt_data = pd.DataFrame(plt_data)
    # print(plt_data)
    methods = plt_data["method"].unique()
    max_y_top_1 = plt_data[plt_data["top"] == 1]["mean"].max()
    min_y_top_1 = plt_data[plt_data["top"] == 1]["mean"].min()
    # plot the results, each k at a different panel. results vs number of steps with error bars for the different rhos
    fig, ax = plt.subplots(1, len(ks), figsize=(16, 10))
    for i, k in enumerate(ks):
        plt_data_k = plt_data[plt_data["top"] == k]
        a = ax[i] if len(ks) > 1 else ax
        for method in methods:
            plt_data_k_m = plt_data_k[plt_data_k["method"] == method]
            x = plt_data_k_m["step"]
            y = plt_data_k_m["mean"]
            yerr = plt_data_k_m["std"]
            # ax[i].errorbar(plt_data_k_m["step"], plt_data_k_m["mean"], yerr=plt_data_k_m["std"],
            #                label=method,
            #                marker='o', linewidth=3, linestyle='dashed', markersize=5)
            a.plot(x, y, label=method, marker='o', linewidth=3, linestyle='dashed', markersize=5)

            # Adding shaded error region
            a.fill_between(x, y - yerr, y + yerr, alpha=0.2)
        # make the x ticks integers
        # ax[i].set_xticks(steps_range)
        a.set_xlabel("Function Evaluations")
        # ticks should be 5 times the step, we should only display 5 ticks
        n_ticks_to_display = 5
        step_size = steps // n_ticks_to_display
        a.set_xticks(np.arange(0, steps + 1, step_size))
        a.set_xticklabels(np.arange(0, steps + 1, step_size) * 5)
        ttl = "Best" if k == 1 else f"Top {k} (average)"
        a.set_title(ttl)
        if i == 0:
        #     # # add space for the legend by adjusting the subplot y axis
        #     # y_lim_max = max_y_top_1 + 0.1 * np.abs(max_y_top_1)
        #     # y_lim_min = min_y_top_1 - 0.1 * np.abs(min_y_top_1)
            # a.legend(loc="lower right")
            a.set_ylabel("Objective value $\\rightarrow$")
        # ax[i].set_ylim(1, 4)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, expr_name, f_name))


def plot_reg_vs_validity(results_df, rho, reg_params):
    # results_df = analyze_results(os.path.join(results_path, expr_name, "bo"), steps=steps)
    rho = _get_rho(rho)
    plt_data = {"energy": {"energy": [], "energy_std": [], "validity": [], "validity_std": [], "reg_param": []},
                "prior": {"energy": [], "energy_std": [], "validity": [], "validity_std": [], "reg_param": []},
                "no reg": {"energy": [], "energy_std": [], "validity": [], "validity_std": [], "reg_param": []}}
    for reg_param in reg_params:
        if reg_param == 0:
            df_no_reg = results_df.loc[("no_reg", 0, 0)]
            no_reg_validity_mean, no_reg_validity_std = df_no_reg["validity"]["mean"], df_no_reg["validity"]["sem"]
            no_reg_energy_mean, no_reg_energy_std = df_no_reg["density"]["mean"], df_no_reg["density"]["sem"]
            plt_data["no reg"]["validity"].append(no_reg_validity_mean)
            plt_data["no reg"]["validity_std"].append(no_reg_validity_std)
            plt_data["no reg"]["energy"].append(no_reg_energy_mean)
            plt_data["no reg"]["energy_std"].append(no_reg_energy_std)
            plt_data["no reg"]["reg_param"].append(reg_param)
        method_prior = "prior" if "prior" in results_df.index.get_level_values("reg method") else "energy"
        key_prior = (method_prior, reg_param, 0)
        if key_prior in results_df.index:
            # continue
            df_prior = results_df.loc[(method_prior, reg_param, 0)]
            prior_validity_mean, prior_validity_std = df_prior["validity"]["mean"], df_prior["validity"]["sem"]
            prior_energy_mean, prior_energy_std = df_prior["density"]["mean"], df_prior["density"]["sem"]
            plt_data["prior"]["validity"].append(prior_validity_mean)
            plt_data["prior"]["validity_std"].append(prior_validity_std)
            plt_data["prior"]["energy"].append(prior_energy_mean)
            plt_data["prior"]["energy_std"].append(prior_energy_std)
            plt_data["prior"]["reg_param"].append(reg_param)
        key_energy = ("energy", reg_param, rho)
        if key_energy in results_df.index:
            df_energy = results_df.loc[("energy", reg_param, rho)]
            energy_validity_mean, energy_validity_std = df_energy["validity"]["mean"], df_energy["validity"]["sem"]
            energy_energy_mean, energy_energy_std = df_energy["density"]["mean"], df_energy["density"]["sem"]
            plt_data["energy"]["validity"].append(energy_validity_mean)
            plt_data["energy"]["validity_std"].append(energy_validity_std)
            plt_data["energy"]["energy"].append(energy_energy_mean)
            plt_data["energy"]["energy_std"].append(energy_energy_std)
            plt_data["energy"]["reg_param"].append(reg_param)
    # make a scatter plot of energy vs validity with stds, make the size of the points proportional to the reg param
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    method_handles = []  # For storing method legend handles
    factor = 2000
    labels = {"Prior": "Prior", "Energy": _get_plot_name("energy", rho), "No reg": "No Reg"}
    if rho == NO_RHO_VAL:
        labels["Energy"] = METHOD_NAME
    for method in plt_data:
        if len(plt_data[method]["energy"]) == 0:
            continue

        sc = ax.scatter(plt_data[method]["energy"], plt_data[method]["validity"],
                        s=(200 + factor * (np.array(plt_data[method]['reg_param']))),  # Adjust point size as needed
                        label=labels[method.capitalize()])
        # add to handles, make sure the size is fixed
        # sc.set_sizes([200])
        method_handles.append(sc)
        # method_handles.append(sc)

    ax.set_xlabel(f"{METHOD_NAME} (Average)")
    ax.set_ylabel("% Valid")
    # tilt x axis labels 
    plt.xticks(rotation=45)

    # Create dummy scatter points for reg_param legend
    for reg_param in reg_params[1:]:
        ax.scatter([], [], s=factor * reg_param, color='black', alpha=0.5,  # Adjust alpha for visibility
                   label=f"$\\lambda = {reg_param}$")  # LaTeX formatted label

    # Combine legends
    if len(reg_params) > 1:
        reg_param_handles, reg_param_labels = ax.get_legend_handles_labels()[
                                          -len(reg_params):]  # Get reg_param legend parts
    else:
        reg_param_handles, reg_param_labels = ax.get_legend_handles_labels()
    legend = ax.legend(reg_param_handles, reg_param_labels, title="", loc="best", frameon=True)
    # make the legend transparent
    legend.get_frame().set_alpha(0.5)
    # set all points in the legend to be the same size
    for handle in legend.legendHandles:
        # if lambda in the handle label continue
        if "$" in handle.get_label():
            continue
        handle.set_sizes([1000])

    # Add legends back to the plot (avoiding overwrite)
    ax.add_artist(legend)

    plt.tight_layout()
    fig_name = f"reg_vs_validity_rho_{rho}.png" if rho != NO_RHO_VAL else "reg_vs_validity_rho_None.png"
    plt.savefig(os.path.join(results_path, expr_name, fig_name))


def _get_rho(rho):
    if rho is None:
        return NO_RHO_VAL
    return rho


if __name__ == '__main__':
    results_path = "results/bb_opt"
    args = parse_args()
    expr_name = args.expr_name
    rho = args.rho
    # reg_params = [0.1, 0.3, 0.5, 0.8]
    params = yaml.load(open(os.path.join(results_path, expr_name, "params.yaml"), "r"), Loader=yaml.FullLoader)
    steps = params["n_steps"]
    results_df = analyze_results(os.path.join(results_path, expr_name, "bo"), steps=steps, mi=True)
    results_df.to_csv(os.path.join(results_path, expr_name, "results.csv"))
    # results_df = analyze_results(os.path.join(results_path, expr_name, "bo"), steps=1, mi=True)

    # results_df_energy = results_df.loc[("energy", slice(None), _get_rho(rho))]
    # reg_params = results_df_energy.index.get_level_values("reg param").unique()
    # plot_reg_vs_validity(results_df=results_df, rho=_get_rho(rho), reg_params=[0, 0.2, 0.5, 0.8])
    # for alpha in reg_params:
    results_df = analyze_results(os.path.join(results_path, expr_name, "bo"), steps=steps)

    def _get_best_method(method):
        if method == "prior":
            method_idx = np.logical_and(results_df["reg method"] == "energy", results_df["rho"] == 0)
            if np.sum(method_idx) == 0:
                method_idx = np.logical_and(results_df["reg method"] == "prior", results_df["rho"] == 0)
        elif method == "energy":
            method_idx = np.logical_and(results_df["reg method"] == "energy", results_df["rho"] != 0)
        else:
            method_idx = results_df["reg method"] == method
        df = results_df.loc[method_idx]
        # sort according to the top 1
        df = df.sort_values(by="top 10 (valid)_mean", ascending=False)
        print(df)
        reg_param = df['reg param'].iloc[0]
        rho = df['rho'].iloc[0]
        if method == "energy":
            return f"density_rho_{rho}_alpha_{reg_param}"
        elif method == "no_reg":
            return "no_reg"
        elif method == "prior":
            return f"prior_rho_{rho}_alpha_{reg_param}"
        elif method == "es":
            return f"es_q_{reg_param}"
        elif method == "bound":
            return f"botorch_b_{reg_param}"
    
    try:
        best_methods = [_get_best_method(method) for method in ["energy","prior", "no_reg", "bound", "es"]]
    except Exception as e:
        best_methods = [_get_best_method(method) for method in ["energy","prior", "no_reg", "bound"]]
    # select the best method from results and save to csv
    plot_top_vals_per_step(steps=steps, methods=best_methods
                            , f_name=f"top_vs_steps.png",
                            ks=[1, 10])
