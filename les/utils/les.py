import copy

import torch

import numpy as np

from torch import nn

from les.nets.utils import get_vae
from les.utils.utils import get_mu_sigma


class Net(nn.Module):
    def __init__(self, decoder):
        super(Net, self).__init__()
        self.decoder = decoder

    def forward(self, x):
        return torch.softmax(self.decoder(x), dim=-1)


class SoftMax(nn.Module):
    def __init__(self, is_image=False):
        super(SoftMax, self).__init__()
        self.is_image = is_image

    def forward(self, x):
        seq_len = x.shape[1]
        out_vec = []
        for i in range(seq_len):
            probs = torch.softmax(self.decoder(x[:, i, :]), dim=-1)
            constant = torch.sum(torch.exp(self.decoder(x[:, i, :])), dim=-1)
            out_vec.append(torch.cat([probs, constant], dim=-1))
            # out_vec.append(self.decoder(x[:, i, :]))
        return torch.stack(out_vec, dim=1)
        # return torch.softmax(x, dim=-1)

    def jacobian(self, x):
        """
        Computes the full Jacobian matrix of the softmax function with respect to its input x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, sequence, vocab)

        Returns:
            torch.Tensor: Jacobian tensor of shape (batch, sequence * vocab, sequence * vocab)
        """

        # Compute softmax over the last dimension (vocab dimension)
        if self.is_image:
            # print(x.shape)
            s = torch.softmax(x, dim=2)[..., 0]  # Shape: (batch, pixels)
            jacobian = torch.diag_embed(s * (1 - s))  # Shape: (batch, pixels, pixels)
            # print(jacobian.shape)
        else:
            s = torch.softmax(x, dim=-1)  # Shape: (batch, sequence, vocab)
            batch_size, sequence_length, vocab_size = x.shape
            jacobian = torch.zeros(
                batch_size,
                sequence_length * vocab_size,
                sequence_length * vocab_size,
                device=x.device,
                dtype=x.dtype,
            )

            # Compute the Jacobian for each batch
            for b in range(batch_size):
                # For each position in the sequence
                for t in range(sequence_length):
                    # Extract the softmax output at position t
                    s_t = s[b, t, :]  # Shape: (vocab_size,)

                    # Compute the Jacobian at position t
                    diag_s = torch.diag(s_t)  # Shape: (vocab_size, vocab_size)

                    s_outer = torch.ger(
                        s_t, s_t
                    )  # Outer product, shape: (vocab_size, vocab_size)
                    jacobian_t = diag_s - s_outer  # Shape: (vocab_size, vocab_size)

                    # Place jacobian_t in the block corresponding to position t
                    start = t * vocab_size
                    end = (t + 1) * vocab_size
                    jacobian[b, start:end, start:end] = jacobian_t

        return jacobian

    def jacobian_full(self, x):
        """
        Computes the full Jacobian matrix of the softmax function and the normalizing constant c
        with respect to its input x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, sequence, vocab)

        Returns:
            torch.Tensor: Jacobian tensor of shape (batch, sequence * (vocab + 1), sequence * vocab)
        """
        batch_size, sequence_length, vocab_size = x.shape
        output_size = vocab_size + 1  # Since we include the normalizing constant c

        # Initialize the Jacobian tensor
        jacobian = torch.zeros(
            batch_size,
            sequence_length * output_size,
            sequence_length * vocab_size,
            device=x.device,
            dtype=x.dtype,
        )

        # Compute softmax probabilities and constants
        logits = x  # Apply decoder if necessary
        s = torch.softmax(logits, dim=-1)  # Shape: (batch, sequence, vocab)
        c = torch.sum(
            torch.exp(logits), dim=-1, keepdim=True
        )  # Shape: (batch, sequence, 1)

        # Compute the Jacobian for each sample in the batch
        for b in range(batch_size):
            # For each position in the sequence
            for t in range(sequence_length):
                # Extract the logits, softmax probabilities, and constant at position t
                logits_t = logits[b, t, :]  # Shape: (vocab_size,)
                s_t = s[b, t, :]  # Shape: (vocab_size,)
                # c_t = c[b, t, 0]            # Scalar

                # Compute the Jacobian of softmax probabilities with respect to logits
                diag_s = torch.diag(s_t)  # Shape: (vocab_size, vocab_size)
                s_outer = torch.outer(s_t, s_t)  # Shape: (vocab_size, vocab_size)
                jacobian_probs = diag_s - s_outer  # Shape: (vocab_size, vocab_size)

                # Compute the derivative of c with respect to logits
                exp_logits = torch.exp(logits_t)  # Shape: (vocab_size,)
                jacobian_c = -exp_logits / (
                    exp_logits.sum() ** 2
                )  # Shape: (vocab_size,)
                # jacobian_c *= 0
                # Note: ∂c/∂x_j = e^{x_j}
                # print(jacobian_c)
                # Combine the Jacobian matrices
                jacobian_block = torch.zeros(
                    output_size, vocab_size, device=x.device, dtype=x.dtype
                )
                jacobian_block[:vocab_size, :] = (
                    jacobian_probs  # (vocab_size, vocab_size)
                )
                jacobian_block[vocab_size, :] = jacobian_c  # (1, vocab_size)
                start_row = t * output_size
                end_row = (t + 1) * output_size
                start_col = t * vocab_size
                end_col = (t + 1) * vocab_size
                jacobian[b, start_row:end_row, start_col:end_col] = jacobian_block

        return jacobian


class LES(nn.Module):
    def __init__(self, model, polarity=False) -> None:
        super(LES, self).__init__()
        self.model = copy.deepcopy(model).train()
        self.model.decoder = self.model.decoder.train()
        # self.model.decoder.train()
        self.device = next(self.model.decoder.parameters()).device
        self.polarity = polarity
        # self.inv = inv

    # @timeit
    def forward(self, x, a_omega=None):
        # send x to
        # x = x.requires_grad_(True)
        if torch.cuda.is_available():
            x = x.to(device="cuda")
            decoder = self.model.decoder.to(device="cuda")
            model = self.model.to(device="cuda")
        else:
            model = self.model
            decoder = model.decoder
        decoder = decoder.train()

        pre_sm = decoder(x)
        seq_len = pre_sm.shape[1]
        if a_omega is None:
            with torch.no_grad():
                a_omega = net_derivative(x.clone(), decoder)  # * self.temperature

        a_omega = torch.cat([a_omega[:, :, i, :] for i in range(seq_len)], dim=-1)
        a_omega = a_omega.to(dtype=model.dtype, device=model.device).transpose(1, 2)

        if not self.polarity:
            sm = SoftMax(is_image=False)
            softmax_tag = sm.jacobian_full(pre_sm).to(
                dtype=model.dtype, device=model.device
            )
            c = torch.bmm(softmax_tag, a_omega)
        else:
            c = a_omega

        c = torch.bmm(c.transpose(1, 2), c)
        singular_values = torch.svd(c, compute_uv=False)[1]
        scales = -0.5 * torch.log(singular_values).sum(dim=1)
        return scales


class Likelihood(nn.Module):
    def __init__(self, model, max_length=None):
        super(Likelihood, self).__init__()
        self.model = copy.deepcopy(model).train()
        self.model.decoder = self.model.decoder.train()
        self.max_length = max_length

    def forward(self, X):
        decoder = self.model.decoder.to(
            device=self.model.device, dtype=self.model.dtype
        )  # .train()
        X = X.to(device=self.model.device, dtype=self.model.dtype)
        predictions = decoder(X)
        probs = torch.softmax(predictions, dim=-1)
        # multuply the max prob in each row
        max_probs = probs.max(dim=-1)[0]
        likelihood = torch.log(max_probs).sum(dim=-1)
        return likelihood


class Prior(nn.Module):
    def __init__(self, model, max_length=None):
        super(Prior, self).__init__()
        self.model = model
        self.max_length = max_length
        mu, sigma = get_mu_sigma(self.model)
        self.mu = mu
        self.sigma = sigma

    def forward(self, X):
        mu = self.mu.to(device=X.device, dtype=self.model.dtype)
        sigma = self.sigma.to(device=X.device, dtype=self.model.dtype)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        diff = (X - mu).unsqueeze(1)

        log_density = -1 * (diff @ torch.inverse(sigma) @ diff.transpose(1, 2))

        log_density = torch.squeeze(log_density)  # + det_grad
        return log_density


def check_na(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError("Nan in tensor: {}".format(name))


def net_derivative(x, net):
    if torch.cuda.is_available():
        x = x.to(device="cuda")
        net = net.to(device="cuda")
    eps = 1e-4
    batch_size, latent_dim = x.shape
    pred_x = net(x)
    dxs_list = []
    pred_x = pred_x.repeat(latent_dim, 1, 1)
    for i in range(latent_dim):
        dx = torch.zeros_like(x)
        dx[:, i] = eps
        dxs_list.append(x.clone() + dx)
    del x
    dxs = torch.cat(dxs_list, dim=0)
    _batch_size = 200
    n_batches = int(np.ceil(dxs.shape[0] / _batch_size))
    dys_vec_batches = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * _batch_size
        end_idx = (batch_idx + 1) * _batch_size

        # Get a batch of dxs
        dxs_batch = dxs[start_idx:end_idx]

        dys_batch = net(dxs_batch) - pred_x[start_idx:end_idx]

        dys_vec_batches.append(dys_batch)

        # Concatenate the batches
        dys_vec = torch.cat(dys_vec_batches, dim=0)
    J = dys_vec / eps
    shift = np.arange(0, latent_dim * batch_size, batch_size)
    J = torch.stack([J[j + shift, ...] for j in range(batch_size)], dim=0)
    J = J.detach().cpu()
    return J


def test_les():
    vae, _ = get_vae("smiles", "transformer", 1)
    scales = LES(vae, is_image=False)
    z = torch.randn(20, vae.latent_dim).to(device=vae.device, dtype=vae.dtype)
    scales_z = scales(z)
    print(scales_z)


if __name__ == "__main__":
    test_les()
