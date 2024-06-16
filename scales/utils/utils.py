import os
import time
import torch

import numpy as np

from collections import namedtuple
from matplotlib import pyplot as plt

from scales import LOGGER

torch.set_default_dtype(torch.float32)
# DTYPE = torch.float32

# create a namedtuple with dataset names
Datasets = namedtuple("Dataset", ["expression", "molecules"])

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

def freeze_batch_norm(model):
    for module in model.modules():
        # if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

def get_mu_sigma(model):
    # dataset = get_data()
    mu = torch.zeros(model.latent_dim, dtype=model.dtype)
    sigma = torch.ones_like(mu)
    sigma = torch.diag(sigma).detach() #* ((model.eps_std* 10) ** 2)
    # n_batches = int(np.ceil(len(dataset) / batch_size))
    # mu_list = []
    # sigma_list = []
    # for batch in range(n_batches):
    #     dataset_b = dataset[batch * batch_size:(batch + 1) * batch_size, :]
    #     dataset_b = dataset_b.to(device=model.device, dtype=model.dtype)
    #
    #     mu_b, logvar_b = model.encoder(dataset_b)
    #     mu_list.append(mu_b.detach().cpu())
    #     sigma_list.append(torch.exp(logvar_b).detach().cpu())
    #
    # mu_full = torch.cat(mu_list, axis=0)
    # sigma_full = torch.cat(sigma_list, axis=0)
    # mu = torch.mean(mu_full, axis=0)
    # sigma = mu_full.var(dim=0) + torch.mean(sigma_full, axis=0) * (model.eps_std ** 2)
    # # # # sigma = np.diag(sigma)
    # sigma = torch.diag(torch.ones_like(sigma)).detach()
    return mu.to(device=model.device, dtype=model.dtype), sigma.to(device=model.device, dtype=model.dtype)


def one_hot_encode(eq, vocab, max_length):
    # eq = "SOS " + eq + " EOF"
    eq = list(eq)
    one_hot = np.zeros((max_length, len(vocab)))
    one_hot[:, vocab[' ']] = 1
    for i, char in enumerate(eq):
        one_hot[i, vocab[char]] = 1
        one_hot[i, vocab[' ']] = 0
    return one_hot


def one_hot_to_eq(one_hot, vocab):
    seq = []
    inverse_vocab = {v: k for k, v in vocab.items()}
    # print(inverse_vocab)
    # print(one_hot.shape)
    # reshape one_hot to (vocab_size, max_length)
    if np.allclose(one_hot.sum(axis=1), np.ones_like(one_hot.sum(axis=1))):
        one_hot = one_hot.T
    for i in range(one_hot.shape[1]):
        seq.append(inverse_vocab[np.argmax(one_hot[:, i])])
    eq = ''.join(seq)
    # remove spaces at the end
    eq = eq.rstrip()
    return eq


def split_data(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))

    train_dataset.dataset = train_dataset.dataset[train_dataset.indices, :]
    test_dataset.dataset = test_dataset.dataset[test_dataset.indices, :]
    return train_dataset.dataset, test_dataset.dataset


def accuracy(logits, y):
    y_ = logits.argmax(-1)
    a = (y == y_).float().mean()
    return 100 * a.cpu().data.numpy()


def test_property_network(vae, dataset, model_path, black_box_function, n=1000):
    # get n points from the dataset
    sample = dataset[np.random.randint(0, len(dataset), n)]
    sample = sample.to(device=vae.device, dtype=vae.dtype)
    true_values = black_box_function(sample).detach().cpu().numpy()
    vae.property_net.to(device=vae.device, dtype=vae.dtype)
    # vae = vae.to(device=vae.device, dtype=vae.dtype)
    predicted_values = np.squeeze(vae(sample)[-1].detach().cpu().numpy())
    # plot true vs predicted
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(true_values, predicted_values)
    # add the mean of the true values
    # ax.axvline(true_values.mean(), color="red")
    # calculate mse of the mean and the mse of the predicted values
    mse_mean = np.round(float(np.mean((true_values - true_values.mean()) ** 2).round(2)), 2)
    mse_predicted = np.round(float(np.mean((true_values - predicted_values) ** 2).round(2)), 2)
    ax.set_title(f"MSE of mean: {mse_mean}, MSE of predicted: {mse_predicted}")
    print(f"MSE of mean: {mse_mean}, MSE of predicted: {mse_predicted}")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    fig.savefig(os.path.join(model_path, "property.png"))
