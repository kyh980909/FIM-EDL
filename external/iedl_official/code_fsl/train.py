import wandb
import torch
import torch.nn
from torch.optim import LBFGS
from torch.optim import Adam

from classifier import ExpBatchLinNet
from utils.io_utils import logger


def compute_fisher_loss(labels_1hot_, evi_alp_):
    """Return the I-EDL fit terms and Fisher log-det regularizer."""
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)
    gamma1_alp = torch.polygamma(1, evi_alp_)
    gamma1_alp0 = torch.polygamma(1, evi_alp0_)

    gap = labels_1hot_ - evi_alp_ / evi_alp0_
    loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1)
    loss_var_ = (
        evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))
    ).sum(-1)
    loss_det_fisher_ = -(
        torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))
    )
    loss_det_fisher_ = torch.where(torch.isfinite(loss_det_fisher_), loss_det_fisher_, torch.zeros_like(loss_det_fisher_))
    return loss_mse_.mean(), loss_var_.mean(), loss_det_fisher_.mean()


def compute_kl_loss(alphas, labels=None, target_concentration=1.0, concentration=1.0, reverse=True):
    """KL between predicted and target Dirichlet distributions."""
    target_alphas = torch.ones_like(alphas) * concentration
    if labels is not None:
        target_alphas += torch.zeros_like(alphas).scatter_(-1, labels.unsqueeze(-1), target_concentration - 1)

    if reverse:
        loss = dirichlet_kl_divergence(alphas, target_alphas)
    else:
        loss = dirichlet_kl_divergence(target_alphas, alphas)

    return loss


def compute_fisher_trace(alpha):
    alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
    trigamma_alpha = torch.polygamma(1, alpha)
    trigamma_alpha0 = torch.polygamma(1, alpha0)
    return (trigamma_alpha - trigamma_alpha0).sum(dim=-1)


def dirichlet_kl_divergence(alphas, target_alphas):
    epsilon = alphas.new_tensor(1e-8)

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                          torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = torch.squeeze(alp0_term + alphas_term).mean()

    return loss


def build_evidence(outputs_, act_type):
    if act_type == 'exp':
        return torch.exp(outputs_) + 1.0
    if act_type == 'relu':
        return torch.relu(outputs_) + 1.0
    if act_type == 'softplus':
        return torch.nn.functional.softplus(outputs_) + 1.0
    raise NotImplementedError(f'act_type:{act_type} is not supported.')


def compute_edl_fit_terms(labels_1hot_, evi_alp_):
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)
    loss_mse_ = (labels_1hot_ - evi_alp_ / evi_alp0_).pow(2).sum(-1).mean()
    loss_var_ = (
        evi_alp_ * (evi_alp0_ - evi_alp_) / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))
    ).sum(-1).mean()
    return loss_mse_, loss_var_


def build_optimizer(net, optimizer_name, max_corr, max_eval_per_epoch, tolerance_grad,
                    tolerance_change, history_size, line_search_fn, lbfgs_lr, adam_lr):
    if optimizer_name == 'lbfgs':
        return LBFGS(
            net.parameters(),
            lr=lbfgs_lr,
            max_iter=max_corr,
            max_eval=max_eval_per_epoch,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
    if optimizer_name == 'adam':
        return Adam(net.parameters(), lr=adam_lr)
    raise ValueError(f'optimizer_name:{optimizer_name} is not supported.')


def train_iedl(X, Y, loss_type='EDL', act_type='softplus', fisher_c=0.0, kl_c=-1.0, target_c=1.0,
               info_beta=1.0, info_gamma=1.0,
               max_iter=1000, verbose=True, use_wandb=False, n_ep=1,
               optimizer_name='lbfgs', lbfgs_lr=1.0, lbfgs_line_search_fn=None,
               adam_lr=1e-2, grad_clip_norm=10.0):
    """Train an episodic evidential linear classifier on frozen support features."""
    batch_dim, n_samps, n_dim = X.shape
    assert Y.shape == (batch_dim, n_samps)
    num_classes = Y.unique().numel()

    device = X.device
    tch_dtype = X.dtype

    # default value from https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    # from scipy.minimize.lbfgsb. In pytorch, it is the equivalent "max_iter"
    # (note that "max_iter" in torch.optim.LBFGS is defined per epoch and a step function call!)
    max_corr = 10
    tolerance_grad = 1e-05
    tolerance_change = 1e-09
    line_search_fn = lbfgs_line_search_fn

    # According to https://github.com/scipy/scipy/blob/master/scipy/optimize/_lbfgsb_py.py#L339
    # wa (i.e., the equivalenet of history_size) is 2 * m * n (where m is max_corrections and n is the dimensions).
    history_size = max_corr * 2  # since wa is O(2*m*n) in size

    num_epochs = max_iter // max_corr  # number of optimization steps
    max_eval_per_epoch = None  # int(max_corr * max_evals / max_iter) matches the 15000 default limit in scipy!

    net = ExpBatchLinNet(
        exp_bs=batch_dim,
        in_dim=n_dim,
        out_dim=num_classes,
        device=device,
        tch_dtype=tch_dtype,
    ).to(device)
    optimizer = build_optimizer(
        net=net,
        optimizer_name=optimizer_name,
        max_corr=max_corr,
        max_eval_per_epoch=max_eval_per_epoch,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn=line_search_fn,
        lbfgs_lr=lbfgs_lr,
        adam_lr=adam_lr,
    )

    Y_i64 = Y.to(device=device, dtype=torch.int64)
    inputs_ = X
    labels_ = Y_i64

    for epoch in range(num_epochs):
        if verbose:
            running_loss = 0.0

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            outputs_ = net(inputs_)
            labels_1hot_ = torch.zeros_like(outputs_).scatter_(-1, labels_.unsqueeze(-1), 1)
            evi_alp_ = build_evidence(outputs_, act_type)
            loss_fisher_ = 0.0

            if loss_type == 'EDL':
                loss_mse_, loss_var_ = compute_edl_fit_terms(labels_1hot_, evi_alp_)
            elif loss_type == 'IEDL':
                loss_mse_, loss_var_, loss_fisher_ = compute_fisher_loss(labels_1hot_, evi_alp_)
            elif loss_type == 'INFO_EDL':
                loss_mse_, loss_var_ = compute_edl_fit_terms(labels_1hot_, evi_alp_)
            elif loss_type == 'DEDL':
                loss_mse_, loss_var_ = compute_edl_fit_terms(labels_1hot_, evi_alp_)
                _, _, loss_fisher_ = compute_fisher_loss(labels_1hot_, evi_alp_)
            else:
                raise ValueError(f'loss_type:{loss_type} is not supported.')

            evi_alp_ = (evi_alp_ - target_c) * (1 - labels_1hot_) + target_c

            loss_kl = compute_kl_loss(evi_alp_, labels_, target_c)

            if loss_type == 'INFO_EDL':
                alpha_for_weight = evi_alp_.detach()
                fisher_trace = compute_fisher_trace(alpha_for_weight)
                lambda_ps = info_beta * torch.exp(-info_gamma * fisher_trace)
                loss = loss_mse_ + loss_var_ + (lambda_ps * loss_kl).mean()
            else:
                if kl_c == -1.0:
                    loss = loss_mse_ + loss_var_ + fisher_c * loss_fisher_ + (epoch / num_epochs) * loss_kl
                else:
                    loss = loss_mse_ + loss_var_ + fisher_c * loss_fisher_ + kl_c * loss_kl

            if use_wandb:
                if (n_ep < 10) and ((epoch == num_epochs - 1) or (epoch % 5 == 0)):
                    payload = {'Train/total_loss': loss, 'Train/loss_kl': loss_kl,
                               'Train/loss_mse_': loss_mse_.sum(-1).mean(), 'Train/loss_var_': loss_var_.sum(-1).mean(),
                               'Train/loss_fisher_': loss_fisher_,
                               'Train/iter': (n_ep * num_epochs + epoch + 1) * max_corr}
                    if loss_type == 'INFO_EDL':
                        payload['Train/info_lambda_mean'] = lambda_ps.mean()
                        payload['Train/info_fisher_trace_mean'] = fisher_trace.mean()
                    wandb.log(payload)

            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

            if loss.requires_grad:
                loss.backward()

            return loss

        if optimizer_name == 'lbfgs':
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            loss = closure()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        if verbose:
            loss = closure()
            running_loss += loss.item()
            logger(f"Epoch: {epoch + 1:02}/{num_epochs} Loss: {running_loss:.5e}")

    return net
