from collections import namedtuple
import os

import torch
import yaml

from lolbo.vae import get_trained_selfies_vae
from scales import LOGGER
from scales.nets.vae_mnist import ConvDeconv
from scales.nets.gru_vae import VAEGRUConv
from scales.nets.selfies import TransformerVAE
from scales.utils.data_utils.expressions import VOCAB_SIZE_EXPRESSION, EXPR_LENGTH_EXPRESSION
from scales.utils.data_utils.molecules import VOCAB_SIZE as VOCAB_SIZE_MOLECULES, EXPR_LENGTH as EXPR_LENGTH_MOLECULES
from scales.utils.utils import get_device


Datasets = namedtuple("DATASETS", ["expressions", "mnist", "smiles", "selfies"])
DATASETS = Datasets(expressions="expressions", mnist="mnist", smiles="smiles", selfies="selfies")


def get_vae(dataset):
    models_path = "results/vaes"
    device = get_device(True)

    if dataset == "expressions":
        model_name = "gru_expressions"
        model_path = os.path.join(models_path, model_name)

        # params_fname = os.path.join(model_path, "params.yaml")
        params_fname = os.path.join("trained_models", dataset, "params.yaml")

        with open(params_fname, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        # params = {"model_name": model_name,
        #           "batch_size": 600, "eps_std": .01,
        #           "latent_dim": 25, "models_path": models_path,
        #           "epochs": 80, "learning_rate": 1e-3, "dataset_name": "expressions",
        #           "teacher_forcing": False, "hidden_dim_decoder": 100, "encoder_type": "small",
        #           "property_hidden_dim": 100}
        params_net = {"latent_dim": params["latent_dim"], "vocab_size": VOCAB_SIZE_EXPRESSION,
                      "expr_length": EXPR_LENGTH_EXPRESSION,
                      "dropout_rate": 0.2,
                      "eps_std": params["eps_std"],
                      "hidden_size_decoder": params["hidden_dim_decoder"],
                      "encoder_size": params["encoder_type"],
                      "property_hidden": params.get("property_hidden_dim", 100),
                      "teacher_forcing": params["teacher_forcing"]}
        vae = VAEGRUConv(**params_net)
        # weights_file = os.path.join(model_path, f'w.pth')
        weights_file = os.path.join("trained_models", dataset, "w.pth")

        vae.load_state_dict(torch.load(weights_file, map_location=vae.device))
    elif dataset == "mnist":
        model_name = "conv_mnist_property"
        latent_dim = 2
        vae = ConvDeconv(latent_dim)
        model_path = os.path.join(models_path, model_name)
        # weights_file = os.path.join(model_path, f'w.pth')
        weights_file = os.path.join("trained_models", dataset, "w.pth")
        vae.load_state_dict(torch.load(weights_file, map_location=vae.device))
    elif dataset == "smiles":
        model_name = "gru_molecules_tf_True_eps_pp1_dim_56_copy_tf_linear"
        model_path = os.path.join(models_path, model_name)
        params_fname = os.path.join(model_path, "params.yaml")
        params_fname = os.path.join("trained_models", dataset, "params.yaml")

        with open(params_fname, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        params_net = {"latent_dim": params["latent_dim"], "vocab_size": VOCAB_SIZE_MOLECULES,
                      "expr_length": EXPR_LENGTH_MOLECULES,
                      "dropout_rate": 0.2,
                      "eps_std": params["eps_std"],
                      "hidden_size_decoder": params["hidden_dim_decoder"],
                      "encoder_size": params["encoder_type"],
                      "property_hidden": params.get("property_hidden_dim", 100),
                      "teacher_forcing": params["teacher_forcing"]}
        vae = VAEGRUConv(**params_net)
        # weights_file = os.path.join(model_path, f'w.pth')
        weights_file = os.path.join("trained_models", dataset, "w.pth")

        vae.load_state_dict(torch.load(weights_file, map_location=vae.device))
    elif dataset == "selfies":
        vae = get_trained_selfies_vae()
        vae = TransformerVAE(vae)
        vae = vae.to(dtype=torch.float32, device=device)
        vae.trained_vae = vae.trained_vae.to(dtype=torch.float32, device=device)
        vae.trained_vae.encoder = vae.trained_vae.encoder.to(device)
        vae.trained_vae.decoder = vae.trained_vae.decoder.to(device)
        model_path = os.path.join("results", "ood", "selfies")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return vae, model_path
    else:
        raise ValueError(f"dataset {dataset} is not supported")
    vae = vae.to(device=device, dtype=torch.float32)
    vae.encoder = vae.encoder.to(device)
    vae.decoder = vae.decoder.to(device)
    # if hasattr("property_encoder", vae):
    #     vae.property_encoder.to(device)
    n_params_decoder = sum(p.numel() for p in vae.decoder.parameters())
    hidden_dim = vae.latent_dim
    # output_dim = vae.vocab_size * vae.expr_length
    x = vae.generate(1)
    out_dim = x.shape[-1] * x.shape[-2]
    return vae, model_path
