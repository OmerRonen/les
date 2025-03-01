import time
import torch

import numpy as np

from les import LOGGER

torch.set_default_dtype(torch.float32)


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        LOGGER.debug(f"{func.__name__} execution time: {execution_time} seconds")
        return result

    return wrapper


def get_device(print_info=False):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    try:
        if torch.backends.mps.is_available():
            # device = "mps"
            device = "cpu"
    except AttributeError:
        pass
    if print_info:
        LOGGER.debug(f"Using device: {device}")
    return device


DEVICE = get_device(False)


def array_to_tensor(array, device):
    # if the array is already a tensor, return it
    if isinstance(array, torch.Tensor):
        return array.to(device)
    array = torch.from_numpy(array).float().to(device)
    return array.to(device=device)


def get_arg_max_probs(prediction):
    arg_max_y_s = torch.argmax(prediction, dim=1)
    expr_length = prediction.shape[-1]
    # create a vector with 1 at the argmax and zero everywhere else
    prediction_binary = torch.zeros_like(prediction)
    # eye = torch.zeros_like(y_s[0, ...])
    for i in range(prediction.shape[0]):
        prediction_binary[i, arg_max_y_s[i], torch.arange(expr_length)] = 1
    return prediction_binary


def probabilities_to_one_hot(matrix):
    """
    Converts a PyTorch tensor of probabilities into a one-hot encoding.

    Args:
        matrix (torch.Tensor): Input tensor of probabilities with shape (l, d).

    Returns:
        torch.Tensor: One-hot encoding of the input tensor with shape (l, d).
    """
    # Find the column index with the maximum probability in each row
    max_indices = torch.argmax(matrix, dim=1)

    # Create a one-hot encoding tensor
    one_hot_matrix = torch.zeros_like(matrix)
    one_hot_matrix.scatter_(1, max_indices.unsqueeze(1), 1)

    return one_hot_matrix


def get_mu_sigma(model):
    # dataset = get_data()
    mu = torch.zeros(model.latent_dim, dtype=model.dtype)
    sigma = torch.ones_like(mu)
    sigma = torch.diag(sigma).detach()  # * ((model.eps_std* 10) ** 2)
    return mu.to(device=model.device, dtype=model.dtype), sigma.to(
        device=model.device, dtype=model.dtype
    )


def one_hot_encode(eq, vocab, max_length):
    # eq = "SOS " + eq + " EOF"
    eq = list(eq)
    one_hot = np.zeros((max_length, len(vocab)))
    one_hot[:, vocab[" "]] = 1
    for i, char in enumerate(eq):
        one_hot[i, vocab[char]] = 1
        one_hot[i, vocab[" "]] = 0
    return one_hot


def one_hot_to_eq(one_hot, vocab, return_blanks=False):
    seq = []
    inverse_vocab = {v: k for k, v in vocab.items()}
    # make sure the first dimension is the tokens
    if one_hot.shape[1] == len(vocab):
        one_hot = one_hot.T
    for i in range(one_hot.shape[1]):
        # print(one_hot.shape)
        seq.append(inverse_vocab[np.argmax(one_hot[:, i])])
    eq = "".join(seq)
    l_blank = len(eq) - len(eq.rstrip())
    eq = eq.rstrip()
    # remove all "<stop>" tokens
    eq = eq.replace("<stop>", "")
    # remove space anywhere in the equation
    eq = eq.replace(" ", "")
    if return_blanks:
        return eq, l_blank
    return eq


def split_data(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset = train_dataset.dataset[train_dataset.indices, :]
    test_dataset.dataset = test_dataset.dataset[test_dataset.indices, :]
    return train_dataset.dataset, test_dataset.dataset
