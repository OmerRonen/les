import hydra
import numpy as np


from omegaconf import DictConfig
import yaml

from les.nets.template import VAE
from les.nets.utils import get_vae
from les.opt.initilizer import InitializerSpec
from les.opt.turbo import TurboConfig
from les.opt.optimizer import OptimizerSpec
from les.opt.bo import BOConfig, BayesianOptimizer
from les.utils.les import LES, Prior, Likelihood
from les.utils.opt_utils import get_train_test_data, get_black_box_function
from les.utils.uncertainty import calculate_uncertainty


def get_penalty(penalty_name: str, vae: VAE):
    if penalty_name == "les":
        return LES(vae)
    elif penalty_name == "prior":
        return Prior(vae)
    elif penalty_name == "likelihood":
        return Likelihood(vae)
    elif penalty_name == "none":
        return None
    else:
        raise ValueError(f"Penalty {penalty_name} not supported")


def get_es_score(dataset_name: str, beta: str, architecture: str, quantile: str):
    vae = get_vae(dataset_name, architecture, beta)
    quantiles_file = (
        f"data/uncertainty_thresholds/{dataset_name}_{architecture}_{beta}.yaml"
    )
    uncertainty_quantile = yaml.load(open(quantiles_file), Loader=yaml.FullLoader)[
        quantile
    ]

    def es_score(Z):
        uc = calculate_uncertainty(Z, vae).detach().cpu().numpy()
        below_thres = uc < uncertainty_quantile
        return below_thres

    return es_score


@hydra.main(config_path="../configs", config_name="bayes_opt")
def main(cfg: DictConfig):
    # load the vae
    vae, _ = get_vae(cfg.data.dataset, cfg.vae.architecture, cfg.vae.beta)
    # get the penalty
    penalty = get_penalty(cfg.optimizer.penalty, vae)
    es_score = get_es_score(cfg.optimizer.es_score, vae, cfg.optimizer.es_threshold)
    (Z, y), _ = get_train_test_data(
        cfg.data.dataset,
        vae=vae,
        sample_size=cfg.data.n_init,
        run=42,
        objective=cfg.data.objective,
    )
    optimizer_spec = OptimizerSpec(
        method=cfg.optimizer.method,
        n_steps=cfg.bo.n_steps,
        alpha=cfg.optimizer.alpha,
        learning_rate=cfg.optimizer.learning_rate,
        penalty=penalty,
        es_rule=es_score,
    )
    np.random.seed(cfg.data.seed)
    train_idx = np.random.choice(np.arange(len(Z)), cfg.data.n_init, replace=False)
    Z_train = Z[train_idx, ...]
    y_train = y[train_idx].unsqueeze(-1)
    if cfg.data.optimzer.use_turbo:
        cfg.initializer.use_turbo = True
    initializer_spec = InitializerSpec(
        Z=Z_train,
        y=y_train,
        use_turbo=cfg.initializer.use_turbo,
    )
    turbo_config = TurboConfig(
        initial_length=cfg.turbo.initial_length,
        failure_tolerance=cfg.turbo.failure_tolerance,
        success_tolerance=cfg.turbo.success_tolerance,
    )
    bo_config = BOConfig(
        initializer_spec=initializer_spec,
        optimizer_spec=optimizer_spec,
        turbo_config=turbo_config,
        blackbox_function=get_black_box_function(
            dataset_name=cfg.data.dataset,
            objective=cfg.data.objective,
        ),
        vae=vae,
        dataset_name=cfg.data.dataset,
        normalize=cfg.optimizer.method == "lbfgs",
        z_bounds=cfg.bo.z_bounds,
        use_turbo=cfg.optimizer.use_turbo,
        n_batch=cfg.bo.n_batch,
        n_steps=cfg.bo.n_steps,
        use_dkl=cfg.bo.use_dkl,
    )
    bo = BayesianOptimizer(bo_config)
    summary = bo.optimize()
    print(summary)


if __name__ == "__main__":
    main()
