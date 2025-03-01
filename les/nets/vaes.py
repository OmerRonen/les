import os
import math
import numpy as np
import torch
import selfies as sf
from torch import nn
from torch.nn import functional as F
from les.nets.template import (
    DTYPE,
    VAE,
    FeedforwardNetwork,
)
from les.utils.datasets.expressions import VOCAB, is_valid_expression
from les.utils.datasets.molecules import VOCAB as VOCAB_MOL
from les.utils.datasets.molecules import SELFIES_VOCAB
from les.utils.datasets.molecules import verify_smile
from les.utils.utils import one_hot_to_eq
from ugo.utils.quality_filters import pass_quality_filter


def _get_dataset(vocab_size):
    if vocab_size == len(VOCAB):
        return "expressions"
    elif vocab_size == len(VOCAB_MOL):
        return "molecules"
    elif vocab_size == len(SELFIES_VOCAB):
        return "selfies"
    else:
        raise ValueError(f"Unknown vocab size: {vocab_size}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class DecoderRNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        vocab_size,
        expr_length,
        dropout_rate=0.2,
        hidden_size=100,
        teacher_forcing=False,
        num_layers=3,
    ):
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

    def init_hidden(self, batch_size):
        return torch.zeros(
            (self.num_layers, batch_size, self.hidden_size),
            device=self.out.weight.device,
            dtype=DTYPE,
        )

    def forward(self, encoder_hidden, target_tensor=None):
        # if len(encoder_hidden.shape) == 1:
        #     encoder_hidden = encoder_hidden.unsqueeze(0)
        # print("nice")
        self.project = self.project.to(encoder_hidden.device)
        # assert encoder_hidden.requires_grad, "encoder_hidden does not require gradients."
        # print("encoder_hidden.grad_fn:", encoder_hidden.grad_fn)

        batch_size = encoder_hidden.size(0)
        decoder_context = self.project(
            encoder_hidden
        )  # .unsqueeze(1)#.reshape(batch_size, 1, self.hidden_size)  # self.hidden_size // 2)
        # decoder_context = self.project(encoder_hidden)
        # print(self.project)
        # assert decoder_context.grad_fn is not None, "decoder_context lost its grad_fn after projection."
        decoder_context = decoder_context[:, None, :]
        # assert decoder_context.grad_fn is not None, "decoder_context lost its grad_fn after add."

        # decoder_token = torch.zeros(batch_size, 1, VOCAB_SIZE, device=self.out.weight.device)
        decoder_hidden = self.init_hidden(batch_size)
        decoder_output = torch.zeros(
            (batch_size, 1, self.vocab_size), device=self.out.weight.device, dtype=DTYPE
        )

        decoder_outputs = []

        for i in range(self.expr_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_context, decoder_hidden, decoder_output
            )
            # assert decoder_output.grad_fn is not None, f"decoder_output lost its grad_fn at step {i}."

            decoder_outputs.append(decoder_output)
            if self.teacher_forcing and target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_output = target_tensor[:, i, :].unsqueeze(1)
            else:
                # Use softmax instead of hard one-hot encoding to keep gradients
                decoder_output = F.log_softmax(decoder_output, dim=-1)
                decoder_output = torch.exp(
                    decoder_output
                )  # Convert log-softmax back to probabilities
            # assert decoder_output.grad_fn is not None, f"decoder_output lost its grad_fn at step {i} 2."

            # if self.teacher_forcing:
            #     if target_tensor is not None:
            #         # Teacher forcing: Feed the target as the next input
            #         decoder_output = target_tensor[:, i, :].unsqueeze(1)  # Teacher forcing
            #     else:
            #         decoder_output = F.one_hot(decoder_output.argmax(dim=-1), num_classes=self.vocab_size)
            #         decoder_output = decoder_output.to(dtype=DTYPE)
            # else:
            #     decoder_output = torch.zeros_like(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # assert decoder_outputs.grad_fn is not None, f"decoder_outputs at the end."

        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs  # , decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden, output):
        raise NotImplementedError


class DecoderRNNGRU(DecoderRNN):
    def __init__(
        self,
        latent_dim,
        vocab_size,
        expr_length,
        dropout_rate=0.2,
        hidden_size=100,
        teacher_forcing=False,
        num_layers=3,
    ):
        super(DecoderRNNGRU, self).__init__(
            latent_dim=latent_dim,
            vocab_size=vocab_size,
            expr_length=expr_length,
            dropout_rate=dropout_rate,
            hidden_size=hidden_size,
            teacher_forcing=teacher_forcing,
            num_layers=num_layers,
        )

    def _get_gru(self, dropout_rate):
        return nn.GRU(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
            dropout=dropout_rate,
            num_layers=self.num_layers,
        )

    def forward_step(self, input, hidden, output):
        output = output.to(dtype=DTYPE)
        output_p = self.output_projection(output)
        output, hidden = self.gru(input, hidden)
        output = self.out(torch.cat([output, output_p], dim=-1))
        return output, hidden


class DecoderRNNLSTM(DecoderRNN):
    def __init__(
        self,
        latent_dim,
        vocab_size,
        expr_length,
        dropout_rate=0.2,
        hidden_size=100,
        teacher_forcing=False,
        num_layers=3,
    ):
        super(DecoderRNNLSTM, self).__init__(
            latent_dim=latent_dim,
            vocab_size=vocab_size,
            expr_length=expr_length,
            dropout_rate=dropout_rate,
            hidden_size=hidden_size,
            teacher_forcing=teacher_forcing,
            num_layers=num_layers,
        )
        self.gru = self._get_gru(dropout_rate)
        self.hidden_size = hidden_size

    def _get_gru(self, dropout_rate):
        return nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
            dropout=dropout_rate,
            num_layers=self.num_layers,
        )

    def forward_step(self, input, hidden, output):
        output = output.to(dtype=DTYPE)
        output_p = self.output_projection(output)

        # Ensure hidden is a tuple containing both hx and cx, both of which should be 3-D tensors
        output, hidden = self.gru(input, hidden)

        output = self.out(torch.cat([output, output_p], dim=-1))
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state (hx) and cell state (cx) as 3-D tensors
        hx = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.out.weight.device
        )
        cx = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.out.weight.device
        )
        return (hx, cx)


class VAETransformerDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        vocab_size,
        expr_length,
        dropout_rate=0.2,
        hidden_size=100,
        num_layers=3,
        num_heads=4,
        dim_feedforward=2048,
        teacher_forcing=True,
    ):
        super(VAETransformerDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.expr_length = expr_length
        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing

        # Embedding layers
        self.decoder_token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size
        )
        self.decoder_position_encoding = nn.Embedding(
            num_embeddings=expr_length, embedding_dim=hidden_size
        )
        self.decoder_token_unembedding = nn.Parameter(
            torch.randn(hidden_size, vocab_size)
        )

        # Transformer decoder layers
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                activation="relu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Linear layer to expand latent vector z to initial input state
        self.latent_to_initial_input = nn.Linear(latent_dim, hidden_size)

    def forward(self, z, tokens=None, as_probs=False):
        batch_size = z.size(0)
        memory = (
            self.latent_to_initial_input(z)
            .unsqueeze(1)
            .expand(-1, self.expr_length, -1)
        )  # (batch_size, expr_length, hidden_size)

        if tokens is None or True:
            tokens = torch.zeros(batch_size, self.expr_length, device=z.device).long()
        elif tokens.dtype == torch.float64:
            tokens = tokens.argmax(dim=-1).long()

        if as_probs:
            embed = tokens[:, :-1] @ self.decoder_token_embedding.weight
        else:
            embed = self.decoder_token_embedding(tokens[:, :-1])

        # Add the start token embedding at the beginning of the sequence
        start_token_embed = torch.zeros(
            embed.shape[0], 1, embed.shape[-1], device=z.device
        )
        embed = torch.cat([start_token_embed, embed], dim=1)

        # Apply positional encoding
        embed = embed + self.decoder_position_encoding(
            torch.arange(0, embed.size(1), device=z.device).unsqueeze(0)
        )

        # Generate the target mask for sequential decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(embed.shape[1]).to(
            z.device
        )

        # Pass through the Transformer decoder
        decoding = self.decoder(tgt=embed, memory=memory, tgt_mask=tgt_mask)

        # Map the output to the vocabulary size
        logits = decoding @ self.decoder_token_unembedding

        return logits


class Encoder(nn.Module):
    """Convolutional encoder for Grammar VAE.

    Applies a series of one-dimensional convolutions to a batch
    of one-hot encodings of the sequence of rules that generate
    an artithmetic expression.
    """

    def __init__(self, latent_dim, vocab_size, expr_length, conv_size="small"):
        super(Encoder, self).__init__()
        hidden_dim = 100
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        if conv_size == "small":
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
        elif conv_size == "large":
            self.conv1 = nn.Conv1d(vocab_size, 9, kernel_size=9)
            self.batch_norm1 = nn.BatchNorm1d(9)
            self.conv2 = nn.Conv1d(9, 9, kernel_size=9)
            self.batch_norm2 = nn.BatchNorm1d(9)
            self.conv3 = nn.Conv1d(9, 10, kernel_size=11)
            self.batch_norm3 = nn.BatchNorm1d(10)
            out_dim = 10 * (expr_length - 9 - 9 - 11 + 3)
            self.linear = nn.Linear(out_dim, hidden_dim)
        else:
            raise ValueError(
                "Invallid value for `conv_size`: {}. Must be in [small, large]".format(
                    conv_size
                )
            )

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
        conv1_dtype = self.conv1.weight.dtype
        x = x.to(dtype=conv1_dtype)
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
    def __init__(
        self,
        latent_dim,
        vocab_size,
        expr_length,
        dropout_rate=0.2,
        eps_std=1,
        hidden_size_decoder=100,
        encoder_size="small",
        property_hidden=100,
        teacher_forcing=False,
    ):
        super(VAEGRUConv, self).__init__(
            latent_dim=latent_dim,
            encoder=Encoder(
                latent_dim=latent_dim,
                vocab_size=vocab_size,
                expr_length=expr_length,
                conv_size=encoder_size,
            ),
            decoder=DecoderRNNGRU(
                latent_dim=latent_dim,
                vocab_size=vocab_size,
                expr_length=expr_length,
                dropout_rate=dropout_rate,
                hidden_size=hidden_size_decoder,
                teacher_forcing=teacher_forcing,
            ),
            eps_std=eps_std,
            teacher_forcing=teacher_forcing,
        )
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.property_net = FeedforwardNetwork(latent_dim, property_hidden, 1, 0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dataset = _get_dataset(vocab_size)

    def normalize(self, x):
        return x
        # return self.softmax(x)

    @property
    def device(self):
        return self.encoder.mu.weight.device

    def get_output_str(self, z):
        self.decoder.eval()
        decoded_x = self.decode(z)
        oh_decoded_x = F.one_hot(
            decoded_x.argmax(dim=-1), num_classes=self.decoder.vocab_size
        )
        if len(oh_decoded_x.shape) == 2:
            oh_decoded_x = oh_decoded_x.unsqueeze(0)
        if self.dataset == "molecules":
            vocab = VOCAB_MOL
        elif self.dataset == "selfies":
            vocab = SELFIES_VOCAB
        else:
            vocab = VOCAB
        expressions = []
        for x in oh_decoded_x:
            eq_str = one_hot_to_eq(x.cpu().numpy(), vocab)
            expressions.append(eq_str)
        return expressions
        # LOGGER.info("Decoded expression: %s", eq_str)

    def check_if_valid(self, z):
        self.decoder.eval()
        decoded_x = self.decode(z)
        oh_decoded_x = F.one_hot(
            decoded_x.argmax(dim=-1), num_classes=self.decoder.vocab_size
        )
        if len(oh_decoded_x.shape) == 2:
            oh_decoded_x = oh_decoded_x.unsqueeze(0)
        # print(oh_decoded_x.shape)
        # eq_str = one_hot_to_eq(oh_decoded_z.cpu().numpy(), VOCAB)
        if self.dataset == "molecules":
            smiles = []
            blanks = []
            for x in oh_decoded_x:
                s, b = one_hot_to_eq(x.cpu().numpy(), VOCAB_MOL, return_blanks=True)
                smiles.append(s)
                blanks.append(b)
            # print(f"mean blanks: {np.mean(blanks)}")
            # print(smiles)
            is_valid = [verify_smile(x) for x in smiles]
            # smiles_txt = "\n".join(smiles)
            # LOGGER.info("SMILES: %s", smiles_txt)
        elif self.dataset == "selfies":
            smiles = []
            blanks = []
            for x in oh_decoded_x:
                s, b = one_hot_to_eq(x.cpu().numpy(), SELFIES_VOCAB, return_blanks=True)
                s = s.replace(" ", "")
                smile = sf.decoder(s)
                smiles.append(smile)
                blanks.append(b)
            is_valid = pass_quality_filter(smiles)

        else:
            is_valid = [
                is_valid_expression(one_hot_to_eq(x.cpu().numpy(), VOCAB))
                for x in oh_decoded_x
            ]
        return np.array(is_valid, dtype=np.float32)


class VAELSTMConv(VAE):
    def __init__(
        self,
        latent_dim,
        vocab_size,
        expr_length,
        dropout_rate=0.2,
        eps_std=1,
        hidden_size_decoder=100,
        encoder_size="small",
        property_hidden=100,
        teacher_forcing=False,
    ):
        super(VAELSTMConv, self).__init__(
            latent_dim=latent_dim,
            encoder=Encoder(
                latent_dim=latent_dim,
                vocab_size=vocab_size,
                expr_length=expr_length,
                conv_size=encoder_size,
            ),
            decoder=DecoderRNNLSTM(
                latent_dim=latent_dim,
                vocab_size=vocab_size,
                expr_length=expr_length,
                dropout_rate=dropout_rate,
                hidden_size=hidden_size_decoder,
                teacher_forcing=teacher_forcing,
            ),
            eps_std=eps_std,
            teacher_forcing=teacher_forcing,
        )
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.property_net = FeedforwardNetwork(latent_dim, property_hidden, 1, 0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dataset = _get_dataset(vocab_size)

    def normalize(self, x):
        return x
        # return self.softmax(x)

    @property
    def device(self):
        return self.encoder.mu.weight.device

    def get_output_str(self, z):
        self.decoder.eval()
        decoded_x = self.decode(z)
        oh_decoded_x = F.one_hot(
            decoded_x.argmax(dim=-1), num_classes=self.decoder.vocab_size
        )
        if len(oh_decoded_x.shape) == 2:
            oh_decoded_x = oh_decoded_x.unsqueeze(0)
        if self.dataset == "molecules":
            vocab = VOCAB_MOL
        elif self.dataset == "selfies":
            vocab = SELFIES_VOCAB
        else:
            vocab = VOCAB
        expressions = []
        for x in oh_decoded_x:
            eq_str = one_hot_to_eq(x.cpu().numpy(), vocab)
            expressions.append(eq_str)
        return expressions

    def check_if_valid(self, z):
        self.decoder.eval()
        z = z.to(self.device)
        decoded_x = self.decode(z)
        oh_decoded_x = F.one_hot(
            decoded_x.argmax(dim=-1), num_classes=self.decoder.vocab_size
        )
        if len(oh_decoded_x.shape) == 2:
            oh_decoded_x = oh_decoded_x.unsqueeze(0)
        # print(oh_decoded_x.shape)
        # eq_str = one_hot_to_eq(oh_decoded_z.cpu().numpy(), VOCAB)
        if self.dataset == "molecules":
            smiles = []
            blanks = []
            for x in oh_decoded_x:
                s, b = one_hot_to_eq(x.cpu().numpy(), VOCAB_MOL, return_blanks=True)
                smiles.append(s)
                blanks.append(b)
            # print(f"mean blanks: {np.mean(blanks)}")
            # print(smiles)
            is_valid = [verify_smile(x) for x in smiles]
        elif self.dataset == "selfies":
            smiles = []
            blanks = []
            for x in oh_decoded_x:
                s, b = one_hot_to_eq(x.cpu().numpy(), SELFIES_VOCAB, return_blanks=True)
                s = s.replace(" ", "")
                smile = sf.decoder(s)
                smiles.append(smile)
                blanks.append(b)
            is_valid = pass_quality_filter(smiles)
        else:
            is_valid = [
                is_valid_expression(one_hot_to_eq(x.cpu().numpy(), VOCAB))
                for x in oh_decoded_x
            ]
        return np.array(is_valid, dtype=np.float32)


class VAETransformerConv(VAE):
    def __init__(
        self,
        latent_dim,
        vocab_size,
        expr_length,
        dropout_rate=0.2,
        eps_std=1,
        hidden_size_decoder=100,
        encoder_size="small",
        property_hidden=100,
        teacher_forcing=False,
    ):
        super(VAETransformerConv, self).__init__(
            latent_dim=latent_dim,
            encoder=Encoder(
                latent_dim=latent_dim,
                vocab_size=vocab_size,
                expr_length=expr_length,
                conv_size=encoder_size,
            ),
            decoder=VAETransformerDecoder(
                latent_dim=latent_dim,
                vocab_size=vocab_size,
                expr_length=expr_length,
                dropout_rate=dropout_rate,
                hidden_size=hidden_size_decoder,
                teacher_forcing=teacher_forcing,
            ),
            eps_std=eps_std,
            teacher_forcing=teacher_forcing,
        )
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.property_net = FeedforwardNetwork(latent_dim, property_hidden, 1, 0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dataset = _get_dataset(vocab_size)

    def normalize(self, x):
        return x
        # return self.softmax(x)

    @property
    def device(self):
        return self.encoder.mu.weight.device

    def get_output_str(self, z):
        self.decoder.eval()
        decoded_x = self.decode(z)
        oh_decoded_x = F.one_hot(
            decoded_x.argmax(dim=-1), num_classes=self.decoder.vocab_size
        )
        if len(oh_decoded_x.shape) == 2:
            oh_decoded_x = oh_decoded_x.unsqueeze(0)
        if self.dataset == "molecules":
            vocab = VOCAB_MOL
        elif self.dataset == "selfies":
            vocab = SELFIES_VOCAB
        else:
            vocab = VOCAB
        expressions = []
        for x in oh_decoded_x:
            eq_str = one_hot_to_eq(x.cpu().numpy(), vocab)
            expressions.append(eq_str)
        return expressions

    def encode(self, x):
        mu, sigma = self.encoder(x)
        return mu

    def check_if_valid(self, z):
        self.decoder.eval()
        decoded_x = self.decode(z)
        oh_decoded_x = F.one_hot(
            decoded_x.argmax(dim=-1), num_classes=self.decoder.vocab_size
        )
        if len(oh_decoded_x.shape) == 2:
            oh_decoded_x = oh_decoded_x.unsqueeze(0)
        # print(oh_decoded_x.shape)
        # eq_str = one_hot_to_eq(oh_decoded_z.cpu().numpy(), VOCAB)
        if self.dataset == "molecules":
            smiles = []
            blanks = []
            for x in oh_decoded_x:
                s, b = one_hot_to_eq(x.cpu().numpy(), VOCAB_MOL, return_blanks=True)
                smiles.append(s)
                blanks.append(b)
            # print(f"mean blanks: {np.mean(blanks)}")
            # print(smiles)
            is_valid = [verify_smile(x) for x in smiles]
        elif self.dataset == "selfies":
            smiles = []
            blanks = []
            for x in oh_decoded_x:
                s, b = one_hot_to_eq(x.cpu().numpy(), SELFIES_VOCAB, return_blanks=True)
                # remove spaces from s
                s = s.replace(" ", "")
                smile = sf.decoder(s)
                smiles.append(smile)
                blanks.append(b)
            is_valid = pass_quality_filter(smiles)
        else:
            is_valid = [
                is_valid_expression(one_hot_to_eq(x.cpu().numpy(), VOCAB))
                for x in oh_decoded_x
            ]
        return np.array(is_valid, dtype=np.float32)
