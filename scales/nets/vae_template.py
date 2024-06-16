import torch
from torch import nn

from torch.nn import functional as F

from scales import LOGGER

DTYPE = torch.float32


# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim: int, encoder: nn.Module, decoder: nn.Module, eps_std=1.0, teacher_forcing=False):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self.eps_std = eps_std
        self.dtype = DTYPE
        self.teacher_forcing = teacher_forcing

    def set_surrogate_decoder(self, surrogate_decoder):
        self._surrogate_decoder = surrogate_decoder

    @property
    def surrogate_decoder(self):
        if not hasattr(self, '_surrogate_decoder'):
            return None
        return self._surrogate_decoder


    @property
    def device(self):
        raise NotImplementedError

    def normalize(self, x):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        batch_size = mu.size(0)
        epsilon = torch.randn(batch_size, self.latent_dim).to(device=self.device, dtype=self.dtype) * self.eps_std
        z = mu + epsilon * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        property = self.property_net(z)
        reconstructed_x = self.decoder(z, x) if self.teacher_forcing else self.decoder(z)
        reconstructed_x = self.normalize(reconstructed_x)
        return reconstructed_x, mu, logvar, property

    def generate(self, size=1):
        self.eval()
        # samples = []
        z = torch.randn(size, self.latent_dim).to(device=self.device, dtype=self.dtype) * self.eps_std
        decoded = self.decoder(z)
        seq_len = decoded.shape[1]
        # for i in range(seq_len):
        #     print(f"top 5 logits at position {i}: {torch.topk(decoded[0, i, :], 10)}")
        samples = torch.squeeze(self.normalize(decoded))
        return samples
        # for s in range(size):
        #     z = torch.randn(self.latent_dim).to(self.device)
        #     decoded = self.decoder(z.unsqueeze(0))
        #     samples.append(torch.squeeze(self.normalize(decoded)))
        # return torch.stack(samples).cpu().detach().numpy()

    def encode(self, x, add_noise=True):
        x = x.to(self.device, dtype=self.dtype)
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var) * self.eps_std
        # LOGGER.info(f"std: {std}")

        if not add_noise:
            return mu
        z = self.reparameterize(mu, log_var)
        return z.to(self.device, dtype=self.dtype)

    def decode(self, z):
        z = z.to(self.device)
        decoded = self.decoder(z)
        decoded = self.normalize(torch.squeeze(decoded))
        return decoded

    def get_decoding_quality(self, z):
        raise NotImplementedError


class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(FeedforwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class FeedforwardNetworkNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, n_layers=1):
        super(FeedforwardNetworkNew, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # make hidden a list of linear layers with relu activation
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])

        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = sel