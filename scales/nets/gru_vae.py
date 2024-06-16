import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from scales.nets.vae_template import DTYPE, VAE, FeedforwardNetwork, FeedforwardNetworkNew
from scales.utils.data_utils.expressions import VOCAB, is_valid_expression
from scales.utils.data_utils.molecules import VOCAB as VOCAB_MOL 
from scales.utils.data_utils.molecules import verify_smile
from scales.utils.utils import one_hot_to_eq


class DecoderRNN(nn.Module):

    def __init__(self, latent_dim, vocab_size, expr_length,
                 dropout_rate=0.2, hidden_size=100, teacher_forcing=False,
                 num_layers=3):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.project = nn.Linear(latent_dim, self.hidden_size)  # self.hidden_size // 2)
        self.out = nn.Linear(2 * self.hidden_size, vocab_size)
        self.output_projection = nn.Linear(vocab_size, self.hidden_size)
        self.expr_length = expr_length
        self.vocab_size = vocab_size
        self.gru = self._get_gru(dropout_rate)
        self.teacher_forcing = teacher_forcing

    def _get_gru(self, dropout_rate):
        raise NotImplementedError
        # return nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, dropout=dropout_rate,
        #               num_layers=self.num_layers)

    def forward(self, encoder_hidden, target_tensor=None):
        # if len(encoder_hidden.shape) == 1:
        #     encoder_hidden = encoder_hidden.unsqueeze(0)
        batch_size = encoder_hidden.size(0)
        decoder_context = self.project(encoder_hidden).view(batch_size, 1, self.hidden_size)  # self.hidden_size // 2)
        # decoder_token = torch.zeros(batch_size, 1, VOCAB_SIZE, device=self.out.weight.device)
        decoder_hidden = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.out.weight.device,
                                     dtype=DTYPE)
        decoder_output = torch.zeros((batch_size, 1, self.vocab_size), device=self.out.weight.device,
                                     dtype=DTYPE)

        decoder_outputs = []

        for i in range(self.expr_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_context, decoder_hidden, decoder_output)
            decoder_outputs.append(decoder_output)
            if self.teacher_forcing:
                if target_tensor is not None:
                    # Teacher forcing: Feed the target as the next input
                    decoder_output = target_tensor[:, i, :].unsqueeze(1)  # Teacher forcing
                else:
                    decoder_output = F.one_hot(decoder_output.argmax(dim=-1), num_classes=self.vocab_size)
                    decoder_output = decoder_output.to(dtype=DTYPE)
            else:
                decoder_output = torch.zeros_like(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs  # , decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden, output):
        raise NotImplementedError



class DecoderRNNGRU(DecoderRNN):
    def __init__(self, latent_dim, vocab_size, expr_length, dropout_rate=0.2, hidden_size=100, teacher_forcing=False,
                 num_layers=3):
        super(DecoderRNNGRU, self).__init__(latent_dim=latent_dim, vocab_size=vocab_size, expr_length=expr_length,
                                            dropout_rate=dropout_rate, hidden_size=hidden_size,
                                            teacher_forcing=teacher_forcing, num_layers=num_layers)

    def _get_gru(self, dropout_rate):
        return nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, dropout=dropout_rate,
                      num_layers=self.num_layers)

    def forward_step(self, input, hidden, output):
        output_p = self.output_projection(output)
        output, hidden = self.gru(input, hidden)
        output = self.out(torch.cat([output, output_p], dim=-1))
        return output, hidden
    
    @property
    def device(self):
        return self.out.weight.device
    
    @property
    def dtype(self):
        return DTYPE



class Encoder(nn.Module):
    """Convolutional encoder for Grammar VAE.

    Applies a series of one-dimensional convolutions to a batch
    of one-hot encodings of the sequence of rules that generate
    an artithmetic expression.
    """

    def __init__(self, latent_dim, vocab_size, expr_length, conv_size='small'):
        super(Encoder, self).__init__()
        hidden_dim = 100
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        if conv_size == 'small':
            # 12 rules, so 12 input channels
            self.conv1 = nn.Conv1d(vocab_size, 2, kernel_size=2)
            self.batch_norm1 = nn.BatchNorm1d(2)
            self.conv2 = nn.Conv1d(2, 3, kernel_size=3)
            self.batch_norm2 = nn.BatchNorm1d(3)
            self.conv3 = nn.Conv1d(3, 4, kernel_size=4)
            self.batch_norm3 = nn.BatchNorm1d(4)
            # calculate the number of output channels from the convolutions
            out_dim = 4 * (expr_length - 2 - 3 - 4 + 3)
            self.linear = nn.Linear(out_dim, hidden_dim)
        elif conv_size == 'large':
            self.conv1 = nn.Conv1d(vocab_size, 9, kernel_size=9)
            self.batch_norm1 = nn.BatchNorm1d(9)
            self.conv2 = nn.Conv1d(9, 9, kernel_size=9)
            self.batch_norm2 = nn.BatchNorm1d(9)
            self.conv3 = nn.Conv1d(9, 10, kernel_size=11)
            self.batch_norm3 = nn.BatchNorm1d(10)
            out_dim = 10 * (expr_length - 9 - 9 - 11 + 3)
            self.linear = nn.Linear(out_dim, hidden_dim)
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
        x = x.transpose(1, 2)
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


class VAEGRUConv(VAE):
    def __init__(self, latent_dim, vocab_size, expr_length,
                 dropout_rate=0.2, eps_std=1,
                 hidden_size_decoder=100, encoder_size='small',
                 property_hidden = 100,
                 teacher_forcing=False):
        super(VAEGRUConv, self).__init__(latent_dim=latent_dim,
                                         encoder=Encoder(latent_dim=latent_dim,
                                                         vocab_size=vocab_size, expr_length=expr_length,
                                                         conv_size=encoder_size),
                                         decoder=DecoderRNNGRU(latent_dim=latent_dim, vocab_size=vocab_size,
                                                            expr_length=expr_length, dropout_rate=dropout_rate,
                                                            hidden_size=hidden_size_decoder,
                                                               teacher_forcing=teacher_forcing),
                                         eps_std=eps_std,
                                         teacher_forcing=teacher_forcing)
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.property_net = FeedforwardNetwork(latent_dim, property_hidden, 1, 0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dataset = "molecules" if vocab_size == len(VOCAB_MOL) else "expressions"

    def normalize(self, x):
        return x
        # return self.softmax(x)

    @property
    def device(self):
        return self.encoder.mu.weight.device

    def get_decoding_quality(self, z):
        self.decoder.eval()
        decoded_x = self.decode(z)
        oh_decoded_x = F.one_hot(decoded_x.argmax(dim=-1), num_classes=self.decoder.vocab_size)
        if len(oh_decoded_x.shape) == 2:
            oh_decoded_x = oh_decoded_x.unsqueeze(0)
        # print(oh_decoded_x.shape)
        # eq_str = one_hot_to_eq(oh_decoded_z.cpu().numpy(), VOCAB)
        if self.dataset == "molecules":
            smiles = [one_hot_to_eq(x.cpu().numpy(), VOCAB_MOL) for x in oh_decoded_x]
            # print(smiles)
            is_valid = [verify_smile(x) for x in smiles]
        else:
            is_valid = [is_valid_expression(one_hot_to_eq(x.cpu().numpy(), VOCAB)) for x in oh_decoded_x]
        return np.array(is_valid, dtype=np.float32)

