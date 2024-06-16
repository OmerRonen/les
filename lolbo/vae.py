import numpy as np
import torch
from torch.func import jacfwd, hessian, jacrev

from lolbo.utils.mol_utils.load_data import load_molecule_train_data
from .utils.mol_utils.selfies_vae.data import SELFIESDataset, SELFIESDataModule, collate_fn
from .utils.mol_utils.selfies_vae.model_positional_unbounded import InfoTransformerVAE

# from torch.func import jacfwd, hessian, jacrev

def get_trained_selfies_vae():
    vae = InfoTransformerVAE(dataset=SELFIESDataset(load_data=False))
    # load in state dict of trained model:
    # if self.path_to_vae_statedict:
    pth = "lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt"
    state_dict = torch.load(pth, map_location=torch.device('cpu'))
    vae.load_state_dict(state_dict, strict=True)
    return vae

def get_smi_data():
    smiles, selfies, zs, ys = load_molecule_train_data(
        task_id="logp",
        num_initialization_points=20000,
        path_to_vae_statedict="lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt"
    )
    return smiles

def get_selfies_data():
    smiles, selfies, zs, ys = load_molecule_train_data(
        task_id="logp",
        num_initialization_points=10000,
        path_to_vae_statedict="lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt"
    )
    datamodule = SELFIESDataModule(1, train_data_path="data/molecules/250k_rndm_zinc_drugs_clean.slf",
                                   validation_data_path="data/molecules/250k_rndm_zinc_drugs_clean.slf")
    # open file
    encoded = [datamodule.train.encode(d) for d in datamodule.train.tokenize_selfies(selfies)]
    # return collate_fn(encoded)
    return collate_fn(encoded)

if __name__ == '__main__':
    # datamodule = SELFIESDataModule(32, train_data_path="data/molecules/250k_rndm_zinc_drugs_clean.slf",
    #                                validation_data_path="data/molecules/250k_rndm_zinc_drugs_clean_val.slf")
    # encoded = [datamodule.train.encode(d) for d in datamodule.train.data]
    # # arr = np.array()
    # data_tensor = collate_fn(encoded)
    data_tensor = get_selfies_data()
    # datamodule.train
    vae = get_trained_selfies_vae()
    vae.max_string_length = data_tensor.shape[1]
    x = data_tensor[0:5, ...]
    z_encoded = vae.encode(x)[0]
    print(z_encoded.shape)
    x_decoded_2 = vae.decode_z(z_encoded.detach().clone())
    x_decoded = vae.sample(return_logits=True, z=z_encoded.detach().clone())

    # calculate the derivative of vae.decode_z(z_encoded) w.r.t. z_encoded
    # der = jacfwd(vae.decode_z, randomness="same")(z_encoded)
    x_decoded_3 = vae.sample(return_logits=True, z=z_encoded)
    assert torch.allclose(x_decoded[1], x_decoded_3[1])
    assert torch.allclose(x_decoded_2[:,0:x_decoded_3[1].shape[1],: ], x_decoded_3[1], atol=1e-5)
