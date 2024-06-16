import numpy as np
import seaborn as sns
from scales.nets.gru_vae import DecoderRNNGRU


import torch

from scales import LOGGER
from torch.func import jacfwd


from scales.utils.vae import get_vae
from scales.nets.vae_template import DTYPE, VAE
from scales.utils.data_utils.molecules import VOCAB_SIZE
from scales.utils.data_utils.molecules import EOS_TOKEN_IDX as EOS_TOKEN_IDX_MOLECULES
# from lso_splines.utils.data_utils.expressions import EOS_TOKEN_IDX as EOS_TOKEN_IDX_EXPRESSIONS
from scales.utils.utils import DEVICE, freeze_batch_norm, get_mu_sigma, timeit

sns.set_palette("Set1")  # You can replace "Set1" with other Seaborn palettes
# set ticks sizes to be 14
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False})

EFFECTIVE_SEQ_LENGTH_RNN = 30
EFFECTIVE_SEQ_LENGTH_TRANSFORMER = 60


def get_eos_index(hidden_dim):
    if hidden_dim == 56:
        return EOS_TOKEN_IDX_MOLECULES
    elif hidden_dim == 25:
        return 14
    elif hidden_dim == 256:
        return 0

def get_eos_mask(pre_softmax):
# len(VOCAB_SIZE) - 1

    eos = VOCAB_SIZE - 1
    # find the first occurrence of the eos token
    sm = torch.softmax(pre_softmax, dim=-1)
    eos_binary_array = sm.argmax(dim=-1) == eos
    # print(eos_binary_array)
    # find the first True index in each row
    eos_token_idx = torch.where(eos_binary_array)
    seq_length = pre_softmax.shape[1]
    eos_token_idx_full = torch.ones(pre_softmax.shape[0], dtype=torch.long) * seq_length
    eos_token_idx_full = eos_token_idx_full.to(device=pre_softmax.device)
    eos_token_idx_full[eos_token_idx[0]] = eos_token_idx[1]

    # print(eos_token_idx_full)
    # expr_length = eos_token_idx[0] if len(eos_token_idx) > 0 else sm.shape[-1]
    batch = pre_softmax.shape[0]
    hidden_dim = 56
    seq_length = pre_softmax.shape[1]
    dict_size = pre_softmax.shape[2]
    shp = (batch, hidden_dim, seq_length, dict_size)
    arr_dim = torch.ones(size=shp)
    # zero out the values after the eos token
    for i in range(batch):
        if eos_token_idx_full[i] < seq_length:
            arr_dim[i, :, eos_token_idx_full[i]:, :] = 0
    arr_dim = arr_dim[:, :, :EFFECTIVE_SEQ_LENGTH_RNN, :]
    return arr_dim

@timeit
def calculate_scales(X: torch.Tensor, model: torch.nn.Module, mu:torch.Tensor = None, sigma: torch.Tensor = None, batch_size=5, use_probs=True, tau=1):
    # calculate gaussian density of input
    model.eval()
    batch_size = min(batch_size, X.shape[0])
    X = X.to(device="cpu", dtype=model.dtype)
    latent_dim = X.shape[1]
    if mu is None:
        mu = torch.zeros(X.shape[0], latent_dim).to(device="cpu")
    if sigma is None:
        sigma = torch.eye(latent_dim).repeat(X.shape[0], 1, 1).to(device="cpu")
    mu = mu.to(device="cpu")
    if len(X.shape) == 3:
        # flatten the input except for the batch dimension
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    diff = (X - mu).unsqueeze(1).to(device="cpu")
    sigma = sigma.to(device="cpu")
    log_density = - 0.5 * (
            diff @ torch.inverse(sigma) @ diff.transpose(1, 2))  # - 0.5 * torch.sum(torch.log(torch.diag(sigma)))
    # log_density *=  1/(model.eps_std ** 2) 
    # log_density = -0.5 * (diff  @ diff.transpose(1, 2))
    n_batches = int(np.ceil(X.shape[0] / batch_size))
    det_grad_list = []
    for b in range(n_batches):
        X_b_gpu = X[b * batch_size: (b + 1) * batch_size, ...].to(device=model.device)
        det_grad_list.append(calculate_det_grad(X_b_gpu, model, use_probs=use_probs).detach().cpu())
    det_grad = torch.cat(det_grad_list, dim=0)
    LOGGER.debug(f"det grad: {det_grad}, log density: {log_density}")
    log_density = tau * det_grad + torch.squeeze(log_density)
    return log_density


def calculate_sum_of_entropy(X, model, max_length=None):
    predictions = model.decoder(X)
    probs = torch.softmax(predictions, dim=-1)
    if max_length is not None:
        probs = probs[:, :max_length, :]
    entropy = -1 * (probs * torch.log(probs + 1e-6)).sum(dim=-1)
    return -1 * torch.squeeze(entropy.sum(dim=-1))


def get_jth_det(j, dx, a_omega, a_omega_dagger, c_vec, pre_softmax, inv_gtg, expr_length, probs):
    # print(dx)
    vocab_size = pre_softmax.shape[2]
    # print(f"vocab size: {vocab_size}")
    dx_j = dx[:, j, :, :]
    c_tag_vec = torch.sum(a_omega[:, j, :, :] * torch.exp(pre_softmax), dim=2)
    sign_j_i = torch.sign(c_tag_vec)
    log_diff = torch.log(torch.abs(c_tag_vec)) - torch.log(c_vec)
    j_term_vec = sign_j_i * torch.exp(log_diff)
    j_term_vec = j_term_vec.unsqueeze(2).unsqueeze(3).repeat(1, 1, vocab_size, vocab_size)
    d_term_vec = torch.diag_embed((1 / probs**3) * dx_j)
    dgdz = 0
    for i in range(expr_length):
        a_omega_dagger_i = a_omega_dagger[..., i * vocab_size: (i + 1) * vocab_size]
        i_th_term = d_term_vec[:, i, :, :] + j_term_vec[:, i, :, :]
        dgdz += -2 * a_omega_dagger_i @ i_th_term @ a_omega_dagger_i.transpose(1, 2)

    grads = inv_gtg @ dgdz / 1
    return torch.diagonal(grads, dim1=1, dim2=2).sum(dim=1).detach().cpu()

@timeit
def calculate_scales_derivative(X, model, mu, sigma, option=4, thres=12, tau=1, fast=True):
    # thres = 6
    derivative_fn = net_derivative
    a_omega = derivative_fn(X, model.decoder, derivative=True, option=option,  fast=fast).to(dtype=model.dtype, device=model.device)
    with torch.no_grad():
        pre_softmax = model.decoder(X)
        latent_dim = a_omega.shape[1]
        if type(model.decoder) == DecoderRNNGRU and latent_dim == 56:
            expr_length = EFFECTIVE_SEQ_LENGTH_RNN
        else:
            expr_length = EFFECTIVE_SEQ_LENGTH_TRANSFORMER
        if pre_softmax.shape[1] > expr_length:
            # expr_length = seq_length_effective
            pre_softmax = pre_softmax[:, :expr_length, :]
        pre_softmax = torch.clamp(pre_softmax, min=-thres, max=thres)

    probs = torch.softmax(pre_softmax, dim=-1)
    model.decoder.eval()

    def _get_sm_j(p):
        m = (torch.diag_embed(torch.ones_like(p)) - p.unsqueeze(2)) * torch.cat(p.shape[1] * [p.unsqueeze(1)], dim=1)
        return m
        # return (torch.diag(p) - p) @ p.T
    expr_length = pre_softmax.shape[1]
    # if type(model.decoder) == DecoderRNNGRU:
    #     expr_length = EFFECTIVE_SEQ_LENGTH_RNN

    mats = [_get_sm_j(probs[:, i, :]) for i in range(expr_length)]
    # print("step 2")
    # create a block diagonal matrix of the mats
    # expr_length = probs.shape[1]
    vocab_size = probs.shape[2]
    batch_size = probs.shape[0]
    latent_dim = a_omega.shape[1]
    mat = torch.zeros(batch_size, vocab_size * expr_length, vocab_size * expr_length).to(
        dtype=model.dtype, device=model.device)
    for i in range(expr_length):
        mat[:, i * vocab_size: (i + 1) * vocab_size, i * vocab_size: (i + 1) * vocab_size] = mats[i]
    dx = a_omega.reshape(batch_size, latent_dim, expr_length * vocab_size) @ mat
    dx = dx.reshape(batch_size, latent_dim, expr_length, vocab_size).to(dtype=model.dtype, device=model.device)

    seq_length = a_omega.shape[2]
    if seq_length == 1:
        a_omega_dagger = torch.pinverse(a_omega).transpose(1, 2)
    else:
        a_omega_dagger = torch.pinverse(torch.cat([a_omega[:, :, i, :].detach() for i in range(seq_length)], dim=-1)).transpose(1, 2)
    p = get_probs(model, X, pre_softmax=pre_softmax, thres=thres, derivative=True).transpose(1, 2)
    # p = get_probs(model, X, pre_softmax=pre_softmax, thres=30, derivative=True).transpose(1, 2)

    a_omega_dagger = a_omega_dagger.to(dtype=model.dtype, device=model.device)

    g = a_omega_dagger @ p.to(dtype=model.dtype, device=model.device)
    gtg = g @ g.transpose(1, 2)
    # print(gtg.shape)
    gtg_np = gtg.cpu().detach()
    # s = time.time()
    inv_gtg_np = np.linalg.inv(gtg_np)
    
    inv_gtg = torch.from_numpy(inv_gtg_np).to(dtype=model.dtype, device=model.device)

    # @timeit
    def _get_jth_det(j):
        # print(dx)
        vocab_size = pre_softmax.shape[2]
        # print(f"vocab size: {vocab_size}")
        dx_j = dx[:, j, :, :]
        c_tag_vec = torch.sum(a_omega[:, j, :, :] * torch.exp(pre_softmax), dim=2)
        sign_j_i = torch.sign(c_tag_vec)
        log_diff = torch.log(torch.abs(c_tag_vec)) - torch.log(c_vec)
        j_term_vec = sign_j_i * torch.exp(log_diff)
        j_term_vec = j_term_vec.unsqueeze(2).unsqueeze(3).repeat(1, 1, vocab_size, vocab_size)
        d_term_vec = torch.diag_embed((1 / probs**3) * dx_j)
        dgdz = 0
        
        for i in range(expr_length):
            a_omega_dagger_i = a_omega_dagger[..., i * vocab_size: (i + 1) * vocab_size]
            i_th_term = d_term_vec[:, i, :, :] + j_term_vec[:, i, :, :]
            # print(f"omega dagger shape: {a_omega_dagger_i.shape}, i_th_term shape: {i_th_term.shape}")
            dgdz += -2 * a_omega_dagger_i @ i_th_term @ a_omega_dagger_i.transpose(1, 2)
    
        grads = inv_gtg @ dgdz / 1
        return torch.diagonal(grads, dim1=1, dim2=2).sum(dim=1)
    
    c_vec = torch.exp(pre_softmax).sum(dim=2) ** 2

    # s = time.time()
    det_grad_der = 0.5 * torch.stack([_get_jth_det(j) for j in range(latent_dim)], dim=-1)
    # det_grad_der = 0.5 * torch.stack(jth_det, dim=-1)

    if vocab_size == 2:
        det_grad_der = -1 * det_grad_der

    gaussian_der = -1 * (X - mu)#* (1/ model.eps_std ** 2) * tau
    # else:
    der = tau * det_grad_der + gaussian_der
    LOGGER.debug(f"gaussian norm: {torch.norm(gaussian_der, dim=1).cpu().mean()} det_grad_der norm: {torch.norm(det_grad_der, dim=1).cpu().mean()}")

    LOGGER.debug(f"det grad der: {det_grad_der.mean()} gaussian der: {gaussian_der.mean()}")
    return der.detach().cpu()


def check_na(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError("Nan in tensor: {}".format(name))

@timeit
def calculate_nll_derivative(X, mu, sigma):
    gaussian_der = -0.5 * (X - mu) #@ torch.inverse(sigma)

    return gaussian_der


def calculate_nll(X, model, mu, sigma):
    # calculate gaussian density of input
    model.eval()
    if len(X.shape) == 3:
        # flatten the input except for the batch dimension
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    diff = (X - mu).unsqueeze(1)

    log_density = -1 * (diff @ torch.inverse(sigma) @ diff.transpose(1, 2))

    log_density = torch.squeeze(log_density)  # + det_grad
    return log_density  # (torch.sign(log_density) * torch.log(torch.abs(log_density)))

    # return det_grad
    # return -1 * (torch.sign(log_density) * torch.log(torch.abs(log_density)))

# @timeit
def get_probs(model, X, thres=12, pre_softmax=None, derivative=False):
    decoder = model
    with torch.no_grad():
        pre_softmax = decoder(X).detach().cpu() if pre_softmax is None else pre_softmax
    pre_softmax = pre_softmax.detach().cpu()

    latent_dim = pre_softmax.shape[2]
    if type(model) == DecoderRNNGRU and latent_dim == 56:
        seq_length = EFFECTIVE_SEQ_LENGTH_RNN
    else:
        seq_length = EFFECTIVE_SEQ_LENGTH_TRANSFORMER
    
    pre_softmax = pre_softmax[:, :seq_length, :]
    is_binary = pre_softmax.shape[-1] == 2

    if is_binary and not derivative:
        idx = 0
        pre_softmax = pre_softmax[..., idx]
        pre_softmax = torch.clamp(pre_softmax, min=-15, max=15)
        probs = torch.sigmoid(pre_softmax)
        probs = 1 / (probs * (1 - probs))
        return probs.unsqueeze(2).transpose(1, 2)

    n_preds = pre_softmax.shape[0]
    seq_length = pre_softmax.shape[1]
    dict_size = pre_softmax.shape[2]

    temperature = torch.tensor(1) * 1

    # if type(model.decoder) == DecoderRNNGRU:
    #     seq_length = EFFECTIVE_SEQ_LENGTH_RNN

    out_size = seq_length * (dict_size)
    n_params = seq_length * (dict_size + 1)
    # probs_arr = torch.zeros(pre_softmax.shape[0], pre_softmax.shape[1], pre_softmax.shape[2] + 1).to(DEVICE)
    probs_arr = torch.zeros(pre_softmax.shape[0], n_params, out_size).to(dtype=model.dtype,
                                                                         device="cpu")  # .to(DEVICE)

    for i in range(seq_length):
        token_pre_softmax = pre_softmax[:, i, :]
        probs = torch.softmax(temperature * token_pre_softmax, dim=1)

        log_c = -1 * token_pre_softmax.logsumexp(dim=1)
        c_vec =  (torch.exp(log_c).view(-1, 1) * torch.ones(n_preds, dict_size).to(dtype=model.dtype, device="cpu")).unsqueeze(1)
        one_over_probs = 1 / probs
        p_diag = torch.diag_embed(one_over_probs).to(dtype=model.dtype, device="cpu")
        probs_arr[:, i * (dict_size + 1): (i + 1) * (dict_size + 1),
        i * dict_size: (i + 1) * dict_size] = torch.cat([p_diag, c_vec], dim=1)


    return probs_arr

    # return probs


def calculate_det_grad(X, model, use_probs, fast=False):
    derivative_function = net_derivative  #if not is_image else net_derivative_images
    decoder = model.decoder if hasattr(model, "decoder") else model
    decoder = decoder.to(device=model.device, dtype=model.dtype)
    probs = get_probs(decoder, X).to(device=model.device, dtype=model.dtype)  # [n_preds, seq_length, dict_size]
    a_omega = derivative_function(X.detach().clone(), decoder, fast=True)

    seq_length = a_omega.shape[2]

    if len(a_omega.shape) == 3:
        a_omega_dagger = torch.pinverse(a_omega).transpose(1, 2)
    else:
        a_omega_dagger = torch.pinverse(torch.cat([a_omega[:, :, i, :].detach().cpu() for i in range(seq_length)], dim=-1)).transpose(1, 2)
    a_omega_dagger = a_omega_dagger.to(dtype=model.dtype, device=model.device)

    g = a_omega_dagger @ probs.transpose(1, 2) if use_probs else a_omega_dagger
    if g.shape[-1] == 1:
        eigenvalues = torch.sqrt(torch.svd(g @ g.transpose(1, 2))[1])
    else:
        eigenvalues = torch.svd(g)[1]
    dets = torch.log(eigenvalues).sum(dim=1)
    return dets


def net_derivative_images(x, net, fast=False):
    x.requires_grad = True
    preds = net(x)
    # flatten the preds except for the batch dimension
    preds = preds.view(preds.shape[0], -1)
    out_dim = preds.shape[1]
    n_preds = preds.shape[0]
    in_dim = x.shape[1]
    J = torch.zeros((n_preds, in_dim, out_dim))  # loop will fill in Jacobian

    for i in range(out_dim):
        grd = torch.zeros_like(preds).to(DEVICE)  # same shape as preds
        grd[:, i] = 1  # column of Jacobian to compute
        preds.backward(gradient=grd, retain_graph=True)
        J[:, :, i] = x.grad  # fill in one column of Jacobian
        # raise ValueError
        x.grad.zero_()  # .backward() accumulates gradients, so reset to zero
    return J


# @timeit
def net_derivative(x, net, option=4, derivative=False, fast=False):
    net.eval()
    freeze_batch_norm(net)
    # net.decoder.eval()
    # LOGGER.info(f"model device: {x.device}")
    # option = 5
    if option == 1:

        def _predict(z):
            return net(z).sum(dim=0)

        J = jacfwd(_predict)(x)
        J = J.transpose(2, 0).transpose(1, 3)
    elif option == 2:
        net.train()
        n_preds = x.shape[0]
        xp = x.clone().requires_grad_()
        preds = net(xp)
        latent_dim = x.shape[1]
        length_seq, dict_size = preds.shape[1:]
        # # # preds =preds.view(n_preds, length_seq * dict_size)
        J = torch.zeros((n_preds, latent_dim,
                         length_seq, dict_size)).to(dtype=DTYPE).cpu()  # loop will fill in Jacobian
        for i in range(length_seq):
            for j in range(dict_size):
                # s = time.time()
                grd = torch.autograd.grad(preds[:, i, j].sum(), xp, create_graph=True, retain_graph=True)[0].to(
                    dtype=DTYPE)
                # print(grd.shape)
                # print(time.time() - s)
                J[..., i, j] = grd.detach().cpu()

        J = J.reshape(n_preds, latent_dim, length_seq, dict_size)
    elif option == 3:
        eps = 1e-4
        # net = net.to(dtype=torch.float64)
        # x = x.to(dtype=torch.float64)
        # J = torch.zeros()
        latent_dim = x.shape[1]
        pred_x = net(x).detach().cpu()
        n_preds, length_seq, dict_size = pred_x.shape
        J = torch.zeros((n_preds, latent_dim,
                         length_seq, dict_size)).cpu()  # loop will fill in Jacobian
        # dys = []
        for i in range(latent_dim):
            dx = torch.zeros_like(x)
            dx[:, i] = eps
            dy = net(x + dx).detach().cpu() - pred_x
            # dys.append(dy)

            J[:, i, :, :] = dy / eps
        # return torch.squeeze(torch.stack(dys, dim=1))
    elif option == 4:
        # torch.cuda.empty_cache()
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
        # ys = net(dxs)
        _n_b = 4
        _b_s = dxs.shape[0] // _n_b + dxs.shape[0] % _n_b
        ys_list =[]
        for i in range(_n_b):
            start_idx = i * _b_s
            end_idx = min((i + 1) * _b_s, dxs.shape[0])
            ys_list.append(net(dxs[start_idx:end_idx]))
        ys = torch.cat(ys_list, dim=0).to(device = pred_x.device, dtype=pred_x.dtype)

        _batch_size = 256
        n_batches = int(np.ceil(dxs.shape[0] / _batch_size))
        # print(n_batches)
        dys_vec_batches = []
            
        for batch_idx in range(n_batches):

            start_idx = batch_idx * _batch_size
            end_idx = (batch_idx + 1) * _batch_size

            # Get a batch of dxs
            dxs_batch = dxs[start_idx:end_idx]

            # Calculate dys for the batch
            if fast:
                # assert torch.allclose(ys[start_idx:end_idx], net(dxs_batch), atol=1e-4, rtol=1e-4)
                dys_batch = ys[start_idx:end_idx] - pred_x[start_idx:end_idx]
            else:
                dys_batch = net(dxs_batch) - pred_x[start_idx:end_idx]

            dys_vec_batches.append(dys_batch)

        # Concatenate the batches
            dys_vec = torch.cat(dys_vec_batches, dim=0)
        J = dys_vec / eps
        shift = np.arange(0, latent_dim * batch_size, batch_size)
        J = torch.stack([J[j + shift, ...] for j in range(batch_size)], dim=0)
        J = J.detach().cpu()
    else:
        raise ValueError(f"option {option} is not supported")
    is_binary = J.shape[-1] == 2
    if is_binary and not derivative:
        J = J[..., 0]
    # if net is RNN decoeer take first 20 elements of the sequence
    latent_dim = J.shape[1]
    if type(net) == DecoderRNNGRU and latent_dim == 56:
        max_length = EFFECTIVE_SEQ_LENGTH_RNN
    else:
        max_length = EFFECTIVE_SEQ_LENGTH_TRANSFORMER
    if J.shape[2] > max_length and not is_binary:
        J = J[:, :, :max_length, :]
    return J

