import numpy as np
import torch

import selfies as sf
from torch import nn

from lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import (
    InfoTransformerVAE,
)
from lolbo.vae import get_trained_selfies_vae
from les.utils.datasets.utils import get_dataset
from les.nets.template import VAE
from ugo.utils.quality_filters import pass_quality_filter


class TransformerDecoder(nn.Module):
    def __init__(self, trained_vae: InfoTransformerVAE):
        super(TransformerDecoder, self).__init__()
        self.trained_vae = trained_vae
        self.trained_vae.eval()

    def forward(self, z, eval=True):
        z = z.reshape(-1, self.trained_vae.bottleneck_size, self.trained_vae.d_model)
        # z = z.to(self.device)
        decoded = self.trained_vae.decode_z(z, eval=eval)
        # decoded = torch.softmax(decoded, dim=-1)
        return decoded

    @property
    def device(self):
        return self.trained_vae.device

    @property
    def dtype(self):
        return self.trained_vae.dtype


class TransformerVAE(VAE):
    def __init__(self, trained_vae: InfoTransformerVAE):
        super(TransformerVAE, self).__init__(
            encoder=trained_vae.encoder,
            decoder=TransformerDecoder(trained_vae),
            latent_dim=trained_vae.d_model * trained_vae.bottleneck_size,
        )
        self.trained_vae = trained_vae.float()
        self.trained_vae.eval()
        # self.full_seq = False

    def decode(self, z, eval=True):
        # z = z.to(self.device)
        z = z.reshape(-1, self.trained_vae.bottleneck_size, self.trained_vae.d_model)
        self.trained_vae.eval()
        self.trained_vae.decoder = self.trained_vae.decoder.to(
            device=self.device, dtype=torch.float32
        )

        decoded = self.trained_vae.decode_z(z, eval=eval)
        # decoded = torch.softmax(decoded, dim=-1)
        return decoded

    def encode(self, x, add_noise=True):
        # x = x
        mu, sigma = self.trained_vae.encode(x)
        if not add_noise:
            sigma = torch.zeros_like(sigma)
        z = self.trained_vae.sample_posterior(mu, sigma)
        z = z.view(-1, self.trained_vae.bottleneck_size * self.trained_vae.d_model)
        return z

    def check_if_valid(self, z, return_smiles=False):
        # reshape z to (batch_size, d_model, bottleneck_size)
        z = z.reshape(-1, self.trained_vae.bottleneck_size, self.trained_vae.d_model)
        # self.full_sequence = False
        x = self.decode(z)
        tokens = x.argmax(dim=-1)
        selfies = [self.trained_vae.dataset.decode(t) for t in tokens]
        smiles = [sf.decoder(s) for s in selfies]
        quality = pass_quality_filter(smiles)
        # if smiles is empty string then quality is 0
        quality = quality * np.array(smiles != "").astype(int)
        # return [verify_smile(s) for s in smiles]
        if return_smiles:
            return quality, smiles

        return quality

    @property
    def device(self):
        return self.trained_vae.device


def test_selfies_vae():
    vae = get_trained_selfies_vae()
    vae = TransformerVAE(vae)
    data = get_dataset("selfies")
    x = data["train"]
    batch_size = 500
    n_batches = len(x) // batch_size
    total_n_high, total_n_low = 0, 0
    total_loss_low, total_loss_high = 0, 0
    total_recon_loss_low, total_recon_loss_high = 0, 0
    for i in range(n_batches):
        x_batch = x[i * batch_size : (i + 1) * batch_size]
        z = vae.encode(x_batch)
        print(z.shape)
        quality = vae.check_if_valid(z)
        x_low_quality = x_batch[quality == 0]
        x_high_quality = x_batch[quality == 1]
        loss_low_quality = vae.trained_vae(x_low_quality)["loss"]
        loss_high_quality = vae.trained_vae(x_high_quality)["loss"]
        recon_loss_low = vae.trained_vae(x_low_quality)["recon_loss"]
        recon_loss_high = vae.trained_vae(x_high_quality)["recon_loss"]

        print(f"Low quality: {loss_low_quality}, High quality: {loss_high_quality}")
        print(
            f"Recon Low quality: {recon_loss_low}, Recon High quality: {recon_loss_high}"
        )
        total_loss_low += loss_low_quality * len(x_low_quality)
        total_loss_high += loss_high_quality * len(x_high_quality)
        total_recon_loss_low += recon_loss_low * len(x_low_quality)
        total_recon_loss_high += recon_loss_high * len(x_high_quality)
        total_n_high += len(x_high_quality)
        total_n_low += len(x_low_quality)

    print(
        f"mean loss low quality: {total_loss_low / total_n_low}\n mean loss high quality: {total_loss_high / total_n_high}"
    )
    print(
        f"mean recon loss low quality: {total_recon_loss_low / total_n_low}\n mean recon loss high quality: {total_recon_loss_high / total_n_high}"
    )


if __name__ == "__main__":
    test_selfies_vae()
