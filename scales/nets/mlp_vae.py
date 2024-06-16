import torch

from torch import nn

from .vae_template import VAE, FeedforwardNetwork


class EncoderMLP(nn.Module):
    def __init__(self, latent_space, input_dim, n_classes):
        super(EncoderMLP, self).__init__()
        self.n_classes = n_classes if n_classes > 2 else 1
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * self.n_classes, 512),  # Increased units to 512
            nn.ReLU(),
            nn.Linear(512, 256),  # Added an additional layer with 256 units
            nn.ReLU(),
            nn.Linear(256, 128),  # Added an additional layer with 256 units
            nn.ReLU(),
        )

        self.mu = nn.Linear(128, latent_space)
        self.logvar = nn.Linear(128, latent_space)

    def forward(self, x):
        if len(x.shape) == 4:
            x = torch.squeeze(x)
        x = self.linear(x)
        # z mu is a linear layer with latent_dim outputs
        mu, logvar = self.mu(x), self.logvar(x)
        return mu, logvar


class DecoderMLP(nn.Module):
    def __init__(self, latent_space, out_dim, n_classes, dropout_rate):
        super(DecoderMLP, self).__init__()
        self.out_dim = out_dim
        self.n_classes = n_classes
        self.linear = nn.Sequential(
            nn.Linear(latent_space, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),  # Added an additional layer with 512 units
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),  # Added an additional layer with 512 units
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),  # Added an additional layer with 512 units
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),  # Added an additional layer with 512 units
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),  # Added an additional layer with 512 units
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Added dropout after the second hidden layer
            nn.Linear(512, out_dim * n_classes),
        )

    def pre_sigmoid(self, x):
        x = self.linear(x)
        x = x.view(-1, self.out_dim, self.n_classes)
        # x = self.softmax(x)
        # x = x[..., 0]
        return x

    def forward(self, x):
        x = self.pre_sigmoid(x)
        # x = self.sigmoid(x)
        return x


class VAEMLP(VAE):
    def __init__(self, latent_dim, n_classes, input_dim, dropout_rate=0.2, eps_std=1.0):
        super(VAEMLP, self).__init__(latent_dim=latent_dim,
                                     encoder=EncoderMLP(latent_dim, input_dim, n_classes),
                                     decoder=DecoderMLP(latent_dim,
                                                        dropout_rate=dropout_rate,
                                                        out_dim=input_dim,
                                                        n_classes=n_classes),
                                     eps_std=eps_std)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.property_net = FeedforwardNetwork(latent_dim,
                                               100,
                                               1,
                                               0.2)
        # self.softmax = nn.Softmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def normalize(self, x):
        probs = self.softmax(x)
        # if self.n_classes == 2:
        #     return probs[..., 0]
        return probs

    @property
    def device(self):
        return self.encoder.mu.weight.device


if __name__ == '__main__':
    latent_dim = 10
    n_classes = 15
    input_dim = 12
    dropout_rate = 0.2
    batch_size = 19

    vae = VAEMLP(latent_dim=latent_dim, n_classes=n_classes, input_dim=input_dim, dropout_rate=dropout_rate)
    # get some random data of dim (10, 12, 15)
    x = torch.rand((batch_size, input_dim, n_classes))
    # make the argmax at dim 1 to be one and the rest zero


    # sample z from a gaussian distribution with mean and variance from the encoder
    mu, logvar = vae.encoder(x)
    z = vae.reparameterize(mu, logvar)

    x_reconstructed = vae.decoder(z)

    x_vae = vae(x)[0]
    pass
