import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import Net
from .vae_template import VAE, FeedforwardNetwork

EXPR_LENGTH = 28 * 28
VOCAB_SIZE = 2


class EncoderConvMnist(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderConvMnist, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(28)
        self.relu1 = nn.ReLU()

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(56)
        self.relu2 = nn.ReLU()

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(56)
        self.relu3 = nn.ReLU()

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=56, out_channels=224, kernel_size=4)
        self.bn4 = nn.BatchNorm2d(224)
        self.relu4 = nn.ReLU()
        self.fc_mu = nn.Linear(224 * 11 * 11, latent_dim)
        self.fc_logvar = nn.Linear(224 * 11 * 11, latent_dim)

    def forward(self, x):
        # reshape x to be 28x28
        x = x.view(-1, 1, 28, 28)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Define the Decoder
class DecoderConvMnist(nn.Module):
    def __init__(self, latent_dim):
        super(DecoderConvMnist, self).__init__()
        self.fc = nn.Linear(latent_dim, 224 * 11 * 11)
        self.relu = nn.ReLU()
        # First transposed convolutional layer
        self.deconv1 = nn.ConvTranspose2d(224, 56, kernel_size=4)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(0.2)

        # Second transposed convolutional layer
        self.deconv2 = nn.ConvTranspose2d(56, 56, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(0.2)

        # Third transposed convolutional layer
        self.deconv3 = nn.ConvTranspose2d(56, 28, kernel_size=4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(0.2)

        # Fourth transposed convolutional layer
        self.deconv4 = nn.ConvTranspose2d(28, 2, kernel_size=10)  # Assuming output has 1 channel

    def pre_sigmoid(self, x):
        x = self.relu(self.fc(x))
        x = self.relu1(self.deconv1(x.view(-1, 224, 11, 11)))
        x = self.dropout1(x)

        x = self.relu2(self.deconv2(x))
        x = self.dropout2(x)

        x = self.relu3(self.deconv3(x))
        x = self.dropout3(x)
        x = self.deconv4(x)
        # reshape to be 28x28
        return x

    def forward(self, x):
        x = self.pre_sigmoid(x)
        # x = self.sigmoid(x)
        # flatten the output
        x = x.view(-1, 28 * 28, 2)
        return x


# Define the VAE
class ConvDeconv(VAE):
    def __init__(self, latent_dim):
        super(ConvDeconv, self).__init__(latent_dim=latent_dim,
                                         encoder=EncoderConvMnist(latent_dim),
                                         decoder=DecoderConvMnist(latent_dim))
        self.property_net = FeedforwardNetwork(latent_dim, 100, 1, 0.2)
        self.softmax = nn.Softmax(dim=-1)
        # self.cnn = Net()
        # # load the weights
        # self.cnn.load_state_dict(torch.load("results/models/cnn/w.pth", map_location=self.device))

    @property
    def device(self):
        return self.encoder.fc_mu.weight.device

    def normalize(self, x):
        # x = torch.stack([x, 1 - x], dim=-1)
        # return x
        probs = self.softmax(x)
        # probs = torch.softmax(x, dim=-1)
        return probs

    def get_decoding_quality(self, z):
        cnn = Net()
        cnn.eval()
        cnn.load_state_dict(torch.load("results/models/cnn/w.pth", map_location="cpu"))
        cnn = cnn.to(self.device, dtype=self.dtype)
        self.decoder.eval()
        x = self.decode(z)
        x = x.view(-1, 28, 28, 2)
        x = x[..., 0].reshape(-1, 1, 28, 28)
        return torch.softmax(cnn(x), dim=1)[:, 1]


class EncoderMLPMnist(nn.Module):
    def __init__(self, latent_space):
        super(EncoderMLPMnist, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),  # Increased units to 512
            nn.ReLU(),
            nn.Linear(512, 256),  # Added an additional layer with 256 units
            nn.ReLU(),
            nn.Linear(256, 128),  # Added an additional layer with 256 units
            nn.ReLU(),
        )

        self.mu = nn.Linear(128, latent_space)
        self.logvar = nn.Linear(128, latent_space)

    def forward(self, x):
        x = self.linear(x)
        # z mu is a linear layer with latent_dim outputs
        mu, logvar = self.mu(x), self.logvar(x)
        return mu, logvar


class DecoderMLPMnist(nn.Module):
    def __init__(self, latent_space, dropout_rate):
        super(DecoderMLPMnist, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_space, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),  # Added an additional layer with 512 units
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Added dropout after the second hidden layer
            nn.Linear(512, 28 * 28 * 2),
        )
        self.softmax = nn.Softmax(dim=-1)

    def pre_sigmoid(self, x):
        x = self.linear(x)
        x = x.view(-1, 28 * 28, 2)
        # x = self.softmax(x)
        # x = x[..., 0]
        return x

    def forward(self, x):
        x = self.pre_sigmoid(x)
        # x = self.sigmoid(x)
        return x


class VAEMLPMnist(VAE):
    def __init__(self, latent_dim, dropout_rate=0.2):

        super(VAEMLPMnist, self).__init__(latent_dim=latent_dim,
                                          encoder=EncoderMLPMnist(latent_dim),
                                          decoder=DecoderMLPMnist(latent_dim, dropout_rate))
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.property_net = FeedforwardNetwork(latent_dim, 100, 1, 0.2)
        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Softmax()
        self.cnn = Net()
        # load the weights
        self.cnn.load_state_dict(torch.load("results/models/cnn/w.pth", map_location=self.device))

    def normalize(self, x):
        # return x
        probs = torch.softmax(x, dim=-1)
        return probs

    @property
    def device(self):
        return self.encoder.mu.weight.device

    def get_decoding_quality(self, z):
        cnn = Net()
        cnn.load_state_dict(torch.load("results/models/cnn/w.pth", map_location=self.device))

        x = self.decode(z)
        x = x.view(-1, 28, 28, 2)
        x = x[..., 0]
        return cnn(x.unsqueeze(1))[:, 1].detach().cpu().numpy()
