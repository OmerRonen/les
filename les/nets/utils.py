from collections import namedtuple
import os
import tempfile

import torch
import wandb
import yaml

from les import LOGGER
from les.nets.selfies import TransformerVAE
from les.nets.vaes import VAEGRUConv, VAELSTMConv, VAETransformerConv
from lolbo.vae import get_trained_selfies_vae


Datasets = namedtuple("DATASETS", ["expressions", "smiles", "selfies"])
DATASETS = Datasets(expressions="expressions", smiles="smiles", selfies="selfies")


def get_f_config(ds):
    # cfg path in scales/scales/configs
    # the path should be relative to the scales directory, which is the grandparent of the current file
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
    )
    # print(f"current directory: {os.path.abspath(__file__)}")
    # print(f"list dir of current directory: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}")
    # raise ValueError("Stop here")
    # cfg_path = os.path.join("scales", "configs")
    if ds == "expressions":
        return os.path.join(cfg_path, f"expressions.yaml")
    elif ds in ["molecules", "smiles"]:
        return os.path.join(cfg_path, f"smiles.yaml")
    elif ds == "selfies":
        return os.path.join(cfg_path, f"selfies.yaml")
    else:
        raise ValueError(f"Dataset {ds} not supported")


def load_model_from_wandb(run_id, model_class, project, entity, model_id, ds):
    """
    Load a model from a specific Wandb run and return the model and validation loss.

    Parameters:
    - run_id: str, Wandb run ID to load the model from.
    - model_class: class, The class of the model to instantiate.
    - project: str, Wandb project name.
    - entity: str, Wandb entity name.
    - model_id: str, Wandb model ID.
    - ds: str, Dataset identifier for loading the appropriate configuration.

    Returns:
    - model: Loaded model instance.
    - val_loss: float, The validation loss of the model.
    """
    # if ds  == "smiles":
    #     ds = "molecules"
    with tempfile.TemporaryDirectory() as save_dir:
        # Initialize Wandb run
        # run = wandb.init(entity=entity, project=project, id=run_id)

        # # Use the artifact with the specified name and version
        # artifact = run.use_artifact(f"{model_id}:latest", type="model")
        # dir = artifact.download(root=save_dir)
        # ckpt_file = os.path.join(dir, "model.ckpt")
        # run.finish()
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        LOGGER.info(
            f"Loading model from run {run_id}, project {project}, entity {entity}"
        )
        loaded = False
        # Assuming there's only one run, we fetch that run
        for run in runs:
            # run = runs[0]

            # Get the artifact associated with this run
            artifacts = run.logged_artifacts()
            model_artifact = next((a for a in artifacts if a.type == "model"), None)
            # print(f"Model artifact: {model_artifact}")
            if model_artifact is None:
                continue

            # Download the model artifact
            artifact_dir = model_artifact.download(root=save_dir)
            ckpt_file = os.path.join(artifact_dir, "model.ckpt")

            # Initialize the model
            # Read configs/smiles.yaml file
            f_config = get_f_config(ds)
            with open(f_config) as f:
                cfg = yaml.safe_load(f)["model"]
            try:
                model = model_class(
                    latent_dim=cfg["latent_dim"],
                    vocab_size=cfg["vocab_size"],
                    expr_length=cfg["expression_length"],
                    dropout_rate=cfg["dropout_rate"],
                    eps_std=cfg["eps_std"],
                    teacher_forcing=cfg["teacher_forcing"],
                    encoder_size=cfg["encoder_size"],
                )
            except KeyError:
                model = model_class(latent_dim=cfg["latent_dim"])

            # print(f"Loading model from {ckpt_file}, model class: {model_class}, project: {project}, entity: {entity}, model_id: {model_id}, ds: {ds}")

            # Load the model state dict
            checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            sd = {k.replace("vae.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(sd)
            loaded = True
            break

    if not loaded:
        raise ValueError("Model not loaded.")
    return model


def get_model_class(arch):
    if arch == "gru":
        return VAEGRUConv
    elif arch == "lstm":
        return VAELSTMConv
    elif arch == "transformer":
        return VAETransformerConv
    else:
        raise ValueError(f"Architecture {arch} not supported")


def get_project_name(dataset, arch, lr, beta):
    if dataset == "mnist":
        return f"{dataset}_{arch}_lr_{lr}_b_{beta}_evra"
    if dataset == "molcules":
        dataset = "smiles"
    return f"{dataset}_{arch}_lr_{lr}_b_{beta}_ronaldo"


def get_vae(dataset, architecture, beta, pretrained=False):
    if pretrained:
        if dataset != "selfies":
            raise ValueError("Pretrained model is only available for selfies")
        vae = get_trained_selfies_vae()
        vae = TransformerVAE(vae)
        LOGGER.info("SELFIES Pretrained model loaded")
        if torch.cuda.is_available():
            vae.trained_vae = vae.trained_vae.to(dtype=torch.float32, device="cuda")
            vae.trained_vae.encoder = vae.trained_vae.encoder.to("cuda")
            vae.trained_vae.decoder = vae.trained_vae.decoder.to("cuda")
        else:
            vae.trained_vae = vae.trained_vae.to(dtype=torch.float32, device="cpu")
            vae.trained_vae.encoder = vae.trained_vae.encoder.to("cpu")
            vae.trained_vae.decoder = vae.trained_vae.decoder.to("cpu")
        return vae
    models_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "trained_models",
    )
    # if architecture is not None and beta is not None:
    model_class = get_model_class(architecture)
    config_path = get_f_config(dataset)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["model"]
    model = model_class(
        latent_dim=config["latent_dim"],
        vocab_size=config["vocab_size"],
        expr_length=config["expression_length"],
        dropout_rate=config["dropout_rate"],
        eps_std=config["eps_std"],
        teacher_forcing=config["teacher_forcing"],
        encoder_size=config["encoder_size"],
    )
    model_path = os.path.join(models_path, f"{dataset}_{architecture}_{beta}")
    weights_file = os.path.join(model_path, "w.pth")
    model.load_state_dict(
        torch.load(weights_file, map_location=model.device, weights_only=True)
    )
    return model


# def get_vae(dataset, architecture=None, beta=None):
#     models_path = "/accounts/campus/omer_ronen/projects/lso_splines/results/vaes"
#     if architecture is not None and beta is not None:
#         # if dataset == "smiles":
#         #     dataset = "molecules"

#         print(f"Loading model for dataset {dataset}, architecture {architecture}, beta {beta}")
#         run_dict = get_run_dict(dataset)
#         run_id = None#run_dict[architecture][beta]["run"]
#         model_id = None#run_dict[architecture][beta]["model"]
#         model_class = get_model_class(architecture)
#         print("Model class", model_class)
#         if beta == 1:
#             beta = int(beta)
#         project = get_project_name(dataset, architecture, 0.001, beta)
#         entity = "energy_splines"
#         model = load_model_from_wandb(run_id, model_class, project, entity, model_id=model_id, ds=dataset)
#         if dataset == "mnist":
#             model.load_cnn()
#         models_path = "trained_models"
#         model_path = os.path.join(models_path, f"{dataset}_{architecture}_{beta}")
#         # save model to the trained_models folder
#         if not os.path.exists(model_path):
#             os.makedirs(model_path)
#         torch.save(model.state_dict(), os.path.join(model_path, "w.pth"))
#         return model, model_path

#     device = get_device(True)

#     if dataset == "expressions":
#         model_name = "gru_expressions"
#         model_path = os.path.join(models_path, model_name)

#         # params_fname = os.path.join(model_path, "params.yaml")
#         params_fname = os.path.join("/accounts/campus/omer_ronen/projects/lso_splines/trained_models", dataset, "params.yaml")

#         with open(params_fname, "r") as f:
#             params = yaml.load(f, Loader=yaml.FullLoader)
#         # params = {"model_name": model_name,
#         #           "batch_size": 600, "eps_std": .01,
#         #           "latent_dim": 25, "models_path": models_path,
#         #           "epochs": 80, "learning_rate": 1e-3, "dataset_name": "expressions",
#         #           "teacher_forcing": False, "hidden_dim_decoder": 100, "encoder_type": "small",
#         #           "property_hidden_dim": 100}
#         params_net = {"latent_dim": params["latent_dim"], "vocab_size": VOCAB_SIZE_EXPRESSION,
#                       "expr_length": EXPR_LENGTH_EXPRESSION,
#                       "dropout_rate": 0.2,
#                       "eps_std": params["eps_std"],
#                       "hidden_size_decoder": params["hidden_dim_decoder"],
#                       "encoder_size": params["encoder_type"],
#                       "property_hidden": params.get("property_hidden_dim", 100),
#                       "teacher_forcing": params["teacher_forcing"]}
#         vae = VAEGRUConv(**params_net)
#         # weights_file = os.path.join(model_path, f'w.pth')
#         weights_file = os.path.join("/accounts/campus/omer_ronen/projects/lso_splines/trained_models", dataset, "w.pth")

#         vae.load_state_dict(torch.load(weights_file, map_location=vae.device))
#     elif dataset == "mnist":
#         model_name = "conv_mnist_property"
#         latent_dim = 2
#         vae = ConvDeconv(latent_dim)
#         model_path = os.path.join(models_path, model_name)
#         # weights_file = os.path.join(model_path, f'w.pth')
#         weights_file = os.path.join("trained_models", dataset, "w.pth")
#         vae.load_state_dict(torch.load(weights_file, map_location=vae.device))
#     elif dataset == "smiles":
#         model_name = "gru_molecules_tf_True_eps_pp1_dim_56_copy_tf_linear"
#         model_path = os.path.join(models_path, model_name)
#         # params_fname = os.path.join(model_path, "params.yaml")
#         params_fname = os.path.join("/accounts/campus/omer_ronen/projects/lso_splines/trained_models", dataset, "params.yaml")

#         with open(params_fname, "r") as f:
#             params = yaml.load(f, Loader=yaml.FullLoader)
#         params_net = {"latent_dim": params["latent_dim"], "vocab_size": VOCAB_SIZE_MOLECULES,
#                       "expr_length": EXPR_LENGTH_MOLECULES,
#                       "dropout_rate": 0.2,
#                       "eps_std": params["eps_std"],
#                       "hidden_size_decoder": params["hidden_dim_decoder"],
#                       "encoder_size": params["encoder_type"],
#                       "property_hidden": params.get("property_hidden_dim", 100),
#                       "teacher_forcing": params["teacher_forcing"]}
#         vae = VAEGRUConv(**params_net)
#         # weights_file = os.path.join(model_path, f'w.pth')
#         weights_file = os.path.join("/accounts/campus/omer_ronen/projects/lso_splines/trained_models", dataset, "w.pth")

#         vae.load_state_dict(torch.load(weights_file, map_location=vae.device))
#     elif dataset == "selfies":
#         print("Loading selfies model")
#         vae = get_trained_selfies_vae()
#         vae = TransformerVAE(vae)
#         vae = vae.to(dtype=torch.float32, device=device)
#         vae.trained_vae = vae.trained_vae.to(dtype=torch.float32, device=device)
#         vae.trained_vae.encoder = vae.trained_vae.encoder.to(device)
#         vae.trained_vae.decoder = vae.trained_vae.decoder.to(device)
#         model_path = os.path.join("/accounts/campus/omer_ronen/projects/lso_splines/results", "ood", "selfies")
#         if not os.path.exists(model_path):
#             os.makedirs(model_path)
#         return vae, model_path
#     else:
#         raise ValueError(f"dataset {dataset} is not supported")
#     vae = vae.to(device=device, dtype=torch.float32)
#     vae.encoder = vae.encoder.to(device)
#     vae.decoder = vae.decoder.to(device)
#     # if hasattr("property_encoder", vae):
#     #     vae.property_encoder.to(device)
#     n_params_decoder = sum(p.numel() for p in vae.decoder.parameters())
#     hidden_dim = vae.latent_dim
#     # output_dim = vae.vocab_size * vae.expr_length
#     x = vae.generate(1)
#     out_dim = x.shape[-1] * x.shape[-2]
#     LOGGER.info(f"Loading VAE for dataset {dataset} with {n_params_decoder} parameters, hidden dim {hidden_dim}, output dim {out_dim}")
#     # save model to the trained_models folder
#     model_name = f"{dataset}_{architecture}_{beta}"
#     model_path = os.path.join("trained_models", model_name)
#     print(f"Saving model to {model_path}")
#     raise ValueError("Stop here")
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)
#     torch.save(vae.state_dict(), os.path.join(model_path, "w.pth"))
#     return vae, model_path

# if __name__ == "__main__":
#     for dataset in ["expressions",  "smiles", "selfies"]:
#         for architecture in ["gru", "lstm", "transformer"]:
#             if dataset == "selfies" and architecture != "transformer":
#                 continue
#             for beta in [0.05, 0.1, 1]:
#                 vae, model_path = get_vae(dataset, architecture, beta)
#                 print(f"Model saved to {model_path}")
