import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .cnn import Net
from .vae_template import VAE, FeedforwardNetwork


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
        self.sigmoid = nn.Sigmoid()

    def _load_cnn(self):
        if not hasattr(self, "_cnn"):

            self._cnn = Net()

            # load the weights
            w_file = "results/models/cnn/w.pth"
            self._cnn.load_state_dict(torch.load(w_file, map_location=self.device))
            self._cnn.to(dtype=self.dtype, device=self.device)


    @property
    def device(self):
        return self.encoder.fc_mu.weight.device

    def normalize(self, x):
        # x = torch.stack([x, 1 - x], dim=-1)
        # return x
        # probs = self.softmax(x)
        probs = self.sigmoid(x)
        return probs[..., 0]

    def get_decoding_quality(self, z):
        self._load_cnn()
        x = self.decode(z)
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        # Apply the transformations to the image tensor
        # x = transform(x.detach().numpy())
        x = torch.tensor(x.view(-1,1, 28, 28))

        # x = x.view(-1,1, 28, 28)
        # x = x[..., 0]
        return torch.softmax(self._cnn(x), dim=1)[:, 1].detach().cpu().numpy()
