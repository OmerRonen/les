import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from scales.utils.data_utils.expressions import EXPR_LENGTH_EXPRESSION, VOCAB_SIZE_EXPRESSION
from .vae_template import FeedforwardNetwork, VAE, DTYPE

INPUT_DIM = EXPR_LENGTH_EXPRESSION * VOCAB_SIZE_EXPRESSION


# good tutorial on seq to seq - https://d2l.ai/chapter_recurrent-modern/seq2seq.html

class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(CustomGRUCell, self).__init__()
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        # self.out_layer = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.ReLU()
        self.activation_update = nn.ReLU()
        self.activation_reset = nn.ReLU()
        # self.hidden_activation = nn.Tanh()
        # self.activation_update = nn.Sigmoid()
        # self.activation_reset = nn.Sigmoid()

    def forward(self, x, hidden):
        # print(x.device, hidden.device)
        combined = torch.cat((x, hidden), dim=2)
        update_gate = self.activation_update(self.update_gate(combined))
        reset_gate = self.activation_reset(self.reset_gate(combined))

        # Apply ReLU activation to the hidden state
        updated_hidden = self.hidden_activation(self.hidden_gate(torch.cat((x, (reset_gate + hidden) / 2), dim=2)))
        # updated_hidden = self.hidden_activation(self.hidden_gate(torch.cat((x, reset_gate * hidden), dim=2)))

        # new_hidden = (1 - update_gate) * updated_hidden + update_gate * hidden
        # new_hidden =  (updated_hidden +  hidden) / 2
        # output = self.out_layer(new_hidden)

        return updated_hidden  # self.dropout(new_hidden)


# Define a simple GRU model using the custom cell
class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.2):
        super(CustomGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList([CustomGRUCell(input_size, hidden_size, dropout_rate=dropout_rate)] * num_layers)

    def forward(self, input, hidden):
        hidden_list = []
        hidden_i = torch.zeros_like(hidden)[..., :self.hidden_size]
        for i in range(self.num_layers):
            hidden_i = hidden[i, ...]

            hidden_i = self.gru_cells[i](input, hidden_i.unsqueeze(1))
            # hidden[i, ...] = hidden_i.transpose(0, 1)
            hidden_list.append(hidden_i)
        output = hidden_i
        hidden = torch.cat(hidden_list, dim=1).transpose(0, 1)
        return output, hidden


class EncoderMLPExpr(nn.Module):
    def __init__(self, latent_space):
        super(EncoderMLPExpr, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_DIM, 100),  # Increased units to 512
            nn.ReLU(),
            # add batch norm
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),  # Added an additional layer with 256 units
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 50),  # Added an additional layer with 256 units
            nn.ReLU(),
            nn.BatchNorm1d(50),

            nn.Linear(50, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),

            nn.Linear(50, 50),
            nn.ReLU()

        )

        self.mu = nn.Linear(50, latent_space)
        self.logvar = nn.Linear(50, latent_space)

    def forward(self, x):
        x = self.linear(x)
        # z mu is a linear layer with latent_dim outputs
        mu, logvar = self.mu(x), self.logvar(x)
        return mu, logvar


class DecoderMLPExpr(nn.Module):
    def __init__(self, latent_space, dropout_rate):
        super(DecoderMLPExpr, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_space, 50),
            nn.ReLU(),
            # add batch norm
            # nn.BatchNorm1d(50),
            # nn.LayerNorm(50),
            nn.Dropout(dropout_rate),
            nn.Linear(50, 50),  # Added an additional layer with 512 units
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(50, 50),  # Added an additional layer with 512 units
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(50, 100),  # Added an additional layer with 512 units
            nn.ReLU(),
            # # nn.LayerNorm(100),
            # nn.Dropout(dropout_rate),  # Added dropout after the second hidden layer
            # nn.Linear(100, 100),
            # # # nn.LayerNorm(INPUT_DIM),
            # nn.ReLU(),
            # nn.Linear(100, 100),
            # nn.ReLU(),
            nn.Linear(100, INPUT_DIM),
            nn.ReLU()

            # nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, EXPR_LENGTH_EXPRESSION, VOCAB_SIZE_EXPRESSION)
        return x
        # return self.softmax(x)


class VAEMLPExpr(VAE):
    def __init__(self, latent_dim, dropout_rate=0.2):
        super(VAEMLPExpr, self).__init__(latent_dim=latent_dim,
                                         encoder=EncoderMLPExpr(latent_dim),
                                         decoder=DecoderMLPExpr(latent_dim, dropout_rate))
        self.dropout_rate = dropout_rate
        self.property_net = FeedforwardNetwork(latent_dim, 100, 1, 0.2)
        self.softmax = nn.Softmax(dim=-1)

    def normalize(self, x):
        # return x
        return self.softmax(x)

    @property
    def device(self):
        return self.encoder.mu.weight.device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std) * 0.01
        z = mu + epsilon * std
        return z

    def generate(self, size=1):
        self.eval()
        # samples = []
        z = torch.randn(size, self.latent_dim).to(self.device) * 0.01
        decoded = self.decoder(z)
        samples = torch.squeeze(self.normalize(decoded))
        return samples


class EncoderConvExpr(nn.Module):
    def __init__(self, latent_space):
        super(EncoderConvExpr, self).__init__()
        hidden_size = 50
        self.embedding = nn.Embedding(VOCAB_SIZE_EXPRESSION, hidden_size)
        self.conv1 = nn.Conv1d(hidden_size, 3, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm1d(3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(3, 4, kernel_size=5)
        self.batchnorm2 = nn.BatchNorm1d(4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(4, 5, kernel_size=5)
        self.batchnorm3 = nn.BatchNorm1d(5)
        self.relu3 = nn.ReLU()
        # flatten
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25, 25),
            nn.ReLU(),
            # nn.LayerNorm(50),
            #
            # nn.BatchNorm1d(50),
            # nn.Linear(50, 50),
            # nn.ReLU(),
            # nn.BatchNorm1d(50),
            # nn.Linear(50, 50),
            # nn.ReLU(),
            # nn.BatchNorm1d(50),
            # # nn.Linear(50, 50),
            # # nn.ReLU(),
            # # nn.BatchNorm1d(50),
            #
            # # nn.LayerNorm(50),
        )

        self.mu = nn.Linear(25, latent_space)
        self.logvar = nn.Linear(25, latent_space)

    def convnet(self, x):
        x = self.embedding(x.argmax(dim=2).to(torch.long))
        x = self.conv1(x.transpose(1, 2))
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        return x

    def forward(self, x):
        # if x.shape[2] != EXPR_LENGTH:
        #     x = x.transpose(1, 2)
        x = self.convnet(x)
        x = self.linear(x)
        # z mu is a linear layer with latent_dim outputs
        mu, logvar = self.mu(x), self.logvar(x)
        return mu, logvar


class Encoder(nn.Module):
    """Convolutional encoder for Grammar VAE.

    Applies a series of one-dimensional convolutions to a batch
    of one-hot encodings of the sequence of rules that generate
    an artithmetic expression.
    """

    def __init__(self, latent_dim, conv_size='small'):
        super(Encoder, self).__init__()
        hidden_dim = 100
        self.embedding = nn.Embedding(VOCAB_SIZE_EXPRESSION, hidden_dim)
        if conv_size == 'small':
            # 12 rules, so 12 input channels
            self.conv1 = nn.Conv1d(VOCAB_SIZE_EXPRESSION, 2, kernel_size=2)
            self.batch_norm1 = nn.BatchNorm1d(2)
            self.conv2 = nn.Conv1d(2, 3, kernel_size=3)
            self.batch_norm2 = nn.BatchNorm1d(3)
            self.conv3 = nn.Conv1d(3, 4, kernel_size=4)
            self.batch_norm3 = nn.BatchNorm1d(4)
            self.linear = nn.Linear(52, hidden_dim)
        elif conv_size == 'large':
            self.conv1 = nn.Conv1d(VOCAB_SIZE_EXPRESSION, 24, kernel_size=2)
            self.conv2 = nn.Conv1d(24, 12, kernel_size=3)
            self.conv3 = nn.Conv1d(12, 12, kernel_size=4)
            self.linear = nn.Linear(108, hidden_dim)
        else:
            raise ValueError('Invallid value for `conv_size`: {}.'
                             ' Must be in [small, large]'.format(conv_size))

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Encode x into a mean and variance of a Normal"""
        # x = self.embedding(x.argmax(dim=1).to(torch.long))
        # x should be of dim [batch, VOCAB_SIZE, EXPR_LENGTH]
        # x = x.view(-1, VOCAB_SIZE, EXPR_LENGTH)
        # x = self.embedding(x.argmax(dim=2).to(torch.long))
        h = self.conv1(x)
        h = self.relu(h)
        h = self.batch_norm1(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = self.batch_norm2(h)
        h = self.conv3(h)
        h = self.relu(h)
        h = self.batch_norm3(h)
        h = h.view(x.size(0), -1)  # flatten
        h = self.linear(h)
        h = self.relu(h)
        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))
        return mu, sigma


class Decoder(nn.Module):
    """RNN decoder that reconstructs the sequence of rules from laten z"""

    def __init__(self, latent_dim, dropout_rate, output_size=VOCAB_SIZE_EXPRESSION, rnn_type='gru'):
        super(Decoder, self).__init__()
        hidden_size = 100
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = 3

        self.linear_in = nn.Linear(latent_dim, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, dropout=dropout_rate,
                              num_layers=self.num_layers)
            # self.rnn = CustomGRU(hidden_size, hidden_size, num_layers=self.num_layers, dropout_rate=dropout_rate)
        else:
            raise ValueError('Select rnn_type from [lstm, gru]')

        self.relu = nn.ReLU()

    @property
    def device(self):
        return self.linear_out.weight.device

    def forward(self, z, max_length=EXPR_LENGTH_EXPRESSION):
        """The forward pass used for training the Grammar VAE.

        For the rnn we follow the same convention as the official keras
        implementaion: the latent z is the input to the rnn at each timestep.
        See line 138 of
            https://github.com/mkusner/grammarVAE/blob/master/models/model_eq.py
        for reference.
        """
        x = self.linear_in(z)
        x = self.relu(x)

        # The input to the rnn is the same for each timestep: it is z.
        x = x.unsqueeze(1).expand(-1, max_length, -1)
        hx = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        hx = (hx, hx) if self.rnn_type == 'lstm' else hx

        x, _ = self.rnn(x, hx)

        x = self.relu(x)
        x = self.linear_out(x)
        return x


class DecoderRNN(nn.Module):

    def __init__(self, latent_dim, dropout_rate=0.2):
        super(DecoderRNN, self).__init__()
        self.hidden_size = 100

        self.embedding = nn.Linear(VOCAB_SIZE_EXPRESSION, self.hidden_size // 2)
        self.num_layers = 3
        self.project = nn.Linear(latent_dim, self.hidden_size)  # self.hidden_size // 2)
        self.hs = nn.Softmax(dim=1)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, dropout=dropout_rate,
                          num_layers=self.num_layers)
        # self.gru = CustomGRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers, dropout_rate=dropout_rate)
        self.out = nn.Linear(self.hidden_size, VOCAB_SIZE_EXPRESSION)

    def forward(self, encoder_hidden, target_tensor=None):
        batch_size = encoder_hidden.size(0)
        decoder_context = self.project(encoder_hidden).view(batch_size, 1, self.hidden_size)  # self.hidden_size // 2)
        # decoder_token = torch.zeros(batch_size, 1, VOCAB_SIZE, device=self.out.weight.device)
        decoder_hidden = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.out.weight.device,
                                     dtype=DTYPE)

        decoder_outputs = []

        for i in range(EXPR_LENGTH_EXPRESSION):
            decoder_output, decoder_hidden = self.forward_step(decoder_context, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # if target_tensor is not None:
            #     # Teacher forcing: Feed the target as the next input
            #     decoder_token = target_tensor[..., i].unsqueeze(1)  # Teacher forcing
            #     decoder_token = F.one_hot(decoder_token)
            # else:
            #     # Without teacher forcing: use its own predictions as the next input
            #     # _, topi = decoder_output.topk(1)
            #     # decoder_token = topi.squeeze(-1).detach()  # detach from history as input
            #     decoder_token = self.hs(decoder_output)
            #     # decoder_token = decoder_output
            #     # pass
            # decoder_token = decoder_token.to(torch.long)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs  # , decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        # output = self.embedding(input)
        # output = F.relu(output)
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden


class DecoderRNNLinear(nn.Module):

    def __init__(self, latent_dim, dropout_rate=0.2):
        super(DecoderRNNLinear, self).__init__()
        self.hidden_size = 100

        self.embedding = nn.Linear(VOCAB_SIZE_EXPRESSION, self.hidden_size // 2)
        # self.num_layers = 3
        self.project = nn.Linear(latent_dim, self.hidden_size)  # self.hidden_size // 2)
        self.hs = nn.Softmax(dim=1)

        self.gru = FeedforwardNetwork(self.hidden_size, 100, self.hidden_size, dropout_rate=dropout_rate)
        # self.gru = CustomGRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers, dropout_rate=dropout_rate)
        self.out = nn.Linear(self.hidden_size, VOCAB_SIZE_EXPRESSION)
        self.dtype = DTYPE

    def forward(self, encoder_hidden, target_tensor=None):
        batch_size = encoder_hidden.size(0)
        decoder_context = self.project(encoder_hidden).view(batch_size, self.hidden_size)  # self.hidden_size // 2)
        # decoder_token = torch.zeros(batch_size, 1, VOCAB_SIZE, device=self.out.weight.device)
        decoder_hidden = torch.zeros((batch_size, self.hidden_size), device=self.out.weight.device,
                                     dtype=self.dtype)

        decoder_outputs = []

        for i in range(EXPR_LENGTH_EXPRESSION):
            decoder_output, decoder_hidden = self.forward_step(decoder_context, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # if target_tensor is not None:
            #     # Teacher forcing: Feed the target as the next input
            #     decoder_token = target_tensor[..., i].unsqueeze(1)  # Teacher forcing
            #     decoder_token = F.one_hot(decoder_token)
            # else:
            #     # Without teacher forcing: use its own predictions as the next input
            #     # _, topi = decoder_output.topk(1)
            #     # decoder_token = topi.squeeze(-1).detach()  # detach from history as input
            #     decoder_token = self.hs(decoder_output)
            #     # decoder_token = decoder_output
            #     # pass
            # decoder_token = decoder_token.to(torch.long)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs  # , decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        # output = self.embedding(input)
        # output = F.relu(output)
        hidden = self.gru(input + hidden)
        output = self.out(hidden)
        return output, hidden


class VAEGRUConv(VAE):
    def __init__(self, latent_dim, dropout_rate=0.2, eps_std=1):
        super(VAEGRUConv, self).__init__(latent_dim=latent_dim,
                                         encoder=Encoder(latent_dim),
                                         decoder=DecoderRNN(latent_dim, dropout_rate),
                                         eps_std=eps_std)
        self.dropout_rate = dropout_rate
        self.property_net = FeedforwardNetwork(latent_dim, 100, 1, 0.2)
        self.softmax = nn.Softmax(dim=-1)

    def normalize(self, x):
        return x
        # return self.softmax(x)

    @property
    def device(self):
        return self.encoder.mu.weight.device



if __name__ == '__main__':
    encoder_conv = Encoder(10)
    decoder_gru = DecoderRNNLinear(10, 0.2)
    vae = VAEGRUConv(10, 0.2)
    # get some random data of dim (10, 12, 15)
    x = torch.softmax(torch.randn(10, VOCAB_SIZE_EXPRESSION, EXPR_LENGTH_EXPRESSION), dim=1)
    # make the argmax at dim 1 to be one and the rest zero
    x = torch.argmax(x, dim=1)
    x = F.one_hot(x, num_classes=VOCAB_SIZE_EXPRESSION).float()

    # sample z from a gaussian distribution with mean and variance from the encoder
    mu, logvar = encoder_conv(x.transpose(1,2))
    z = vae.reparameterize(mu, logvar)
    # hidden_z = torch.zeros((z.shape[0], 100)).to(z.device)
    # gru = CustomGRUCell(input_size=z.shape[1], hidden_size=100)
    # new_hidden = gru(z, hidden_z)
    decoder_gru.eval()
    x_reconstructed = decoder_gru(z)

    x_vae = vae(x)[0]
    pass
    # z = torch.randn(10, 1, 10)
