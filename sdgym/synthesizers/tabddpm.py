"""TabDDPMSynthesizer -- SDGym synthesizer for TabDDPM.

Paper: "TabDDPM: Modelling Tabular Data with Diffusion Models (2022)"
https://arxiv.org/abs/2209.15421

Original implementation is provided:
https://github.com/yandex-research/tab-ddpm/tree/main/tab_ddpm.
"""

import logging
import math
import sys
from copy import deepcopy
from typing import Callable, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sdv.metadata import Metadata
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from torch import Tensor

from sdgym.synthesizers.base import BaselineSynthesizer

CAT_MISSING_VALUE = '__nan__'


class FoundNANsError(BaseException):
    """Found NANs during sampling."""

    def __init__(self, message='Found NANs during sampling.'):
        super(FoundNANsError, self).__init__(message)


def sum_except_batch(x, num_dims=1):
    """Sum all dimensions except the first ``num_dims`` batch dimensions."""
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def ohe_to_categories(ohe, K):
    """Apply one hot encoding."""
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i] : indices[i + 1]].argmax(dim=1))
    return torch.stack(res, dim=1)


def _log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def _log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def _extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def _log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def _index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))

    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot


def _log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m


def _sliced_logsumexp(x, slices):
    lse = torch.logcumsumexp(torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')), dim=-1)

    slice_starts = slices[:-1]
    slice_ends = slices[1:]

    slice_lse = _log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(slice_lse, slice_ends - slice_starts, dim=-1)
    return slice_lse_repeated


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Get a pre-defined beta schedule for the given name."""
    if schedule_name == 'linear':
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f'unknown beta schedule: {schedule_name}')


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Beta schedule.

    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def update_ema(target_params, source_params, rate=0.999):
    """Update parametes using EMA.

    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLP(nn.Module):
    """The MLP model from "Revisiting Deep Learning Models for Tabular Data".

    MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
    Block: (in) -> Linear -> Activation -> Dropout -> (out)
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: Union[str, Callable[..., nn.Module]],
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = (
                getattr(nn, activation)() if isinstance(activation, str) else activation()
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass."""
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)

        self.blocks = nn.ModuleList([
            MLP.Block(
                d_in=d_layers[i - 1] if i else d_in,
                d_out=d,
                bias=True,
                activation=activation,
                dropout=dropout,
            )
            for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
        ])
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls,
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        """Create a "baseline" MLP: ReLU activations, uniform dropout."""
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class MLPDiffusion(nn.Module):
    """MLP denoiser with sinusoidal timestep embedding and optional label conditioning."""

    def __init__(self, d_in, num_classes, is_y_cond, rtdl_params, dim_t=128):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond

        rtdl_params = dict(rtdl_params)
        rtdl_params['d_in'] = dim_t
        rtdl_params['d_out'] = d_in

        self.mlp = MLP.make_baseline(**rtdl_params)

        if self.num_classes > 0 and is_y_cond:
            self.label_emb = nn.Embedding(self.num_classes, dim_t)
        elif self.num_classes == 0 and is_y_cond:
            self.label_emb = nn.Linear(1, dim_t)

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, timesteps, y=None):
        """Forward pass."""
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_y_cond and y is not None:
            if self.num_classes > 0:
                y = y.squeeze()
            else:
                y = y.resize(y.size(0), 1).float()
            emb += F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)


class GaussianMultinomialDiffusion(torch.nn.Module):
    """Joint diffusion: Gaussian over numerical features, multinomial over categorical."""

    def __init__(
        self,
        num_classes: np.array,
        num_numerical_features: int,
        denoise_fn,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        gaussian_parametrization='eps',
        multinomial_loss_type='vb_stochastic',
        parametrization='x0',
        scheduler='cosine',
        device=torch.device('cpu'),
    ):
        super(GaussianMultinomialDiffusion, self).__init__()
        assert multinomial_loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if multinomial_loss_type == 'vb_all':
            sys.stdout.write(
                'Computing the loss using the bound on _all_ timesteps. '
                'This is expensive both in terms of memory and computation.\n'
            )

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes  # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device)

        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        alphas = 1.0 - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1.0 - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = _log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = _log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = (
            torch
            .from_numpy(np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])))
            .float()
            .to(device)
        )
        self.posterior_mean_coef1 = (
            (betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float().to(device)
        )
        self.posterior_mean_coef2 = (
            ((1.0 - alphas_cumprod_prev) * np.sqrt(alphas.numpy()) / (1.0 - alphas_cumprod))
            .float()
            .to(device)
        )

        assert _log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.0e-5
        assert _log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.0e-5

        # Convert to float32 and register buffers.
        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            'sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device)
        )

        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))

    def _gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _gaussian_p_mean_variance(self, model_output, x, t):
        B = x.shape[0]
        assert t.shape == (B,)

        model_variance = torch.cat(
            [self.posterior_variance[1].unsqueeze(0).to(x.device), (1.0 - self.alphas)[1:]], dim=0
        )
        model_log_variance = torch.log(model_variance)

        model_variance = _extract(model_variance, t, x.shape)
        model_log_variance = _extract(model_log_variance, t, x.shape)

        # 'eps' parametrization: the network predicts the noise
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

        model_mean, _, _ = self._gaussian_q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart,
        }

    def _gaussian_loss(self, model_out, noise):
        # 'mse' loss: the network predicts the added noise
        return mean_flat((noise - model_out) ** 2)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _gaussian_p_sample(self, model_out, x, t):
        out = self._gaussian_p_mean_variance(model_out, x, t)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

    def _multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def _q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = _extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = _extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = _log_add_exp(
            log_x_t + log_alpha_t, log_1_min_alpha_t - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def _q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = _extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = _extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = _log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded),
        )

        return log_probs

    def _predict_start(self, model_out, log_x_t):
        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes.sum(), f'{model_out.size()}'

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def _q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self._q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(
            t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32)
        )

        # Note: the formula uses log q_pred_one_timestep(x_t, t), _NOT_ x_tmin1.
        unnormed_logprobs = log_EV_qxtmin_x0 + self._q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - _sliced_logsumexp(
            unnormed_logprobs, self.offsets
        )

        return log_EV_xtmin_given_xt_given_xstart

    def _p_pred(self, model_out, log_x, t):
        # 'x0' parametrization
        log_x_recon = self._predict_start(model_out, log_x)
        log_model_pred = self._q_posterior(log_x_start=log_x_recon, log_x_t=log_x, t=t)
        return log_model_pred

    @torch.no_grad()
    def _p_sample(self, model_out, log_x, t):
        model_log_prob = self._p_pred(model_out, log_x=log_x, t=t)
        out = self._log_sample_categorical(model_log_prob)
        return out

    def _log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = _index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def _q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self._q_pred(log_x_start, t)
        log_sample = self._log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def _kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self._q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = self._multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def _compute_Lt(self, model_out, log_x_start, log_x_t, t, detach_mean=False):
        log_true_prob = self._q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self._p_pred(model_out, log_x=log_x_t, t=t)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self._multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -_log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1.0 - mask) * kl

        return loss

    def _sample_time(self, b, device):
        # uniform time sampling
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        pt = torch.ones_like(t).float() / self.num_timesteps
        return t, pt

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt):
        # 'vb_stochastic' loss
        kl = self._compute_Lt(model_out, log_x_start, log_x_t, t)
        kl_prior = self._kl_prior(log_x_start)
        # Upweigh loss term of the kl
        vb_loss = kl / pt + kl_prior
        return vb_loss

    def _mixed_loss(self, x, out_dict):
        b = x.shape[0]
        device = x.device
        t, pt = self._sample_time(b, device)

        x_num = x[:, : self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features :]

        x_num_t = x_num
        log_x_cat_t = x_cat
        noise = None
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self._gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = _index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self._q_sample(log_x_start=log_x_cat, t=t)

        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)

        model_out = self._denoise_fn(x_in, t, **out_dict)

        model_out_num = model_out[:, : self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features :]

        loss_multi = torch.zeros((1,), device=device).float()
        loss_gauss = torch.zeros((1,), device=device).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt) / len(
                self.num_classes
            )

        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, noise)

        return loss_multi.mean(), loss_gauss.mean()

    @torch.no_grad()
    def _sample(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self._log_sample_categorical(uniform_logits)

        y = torch.multinomial(y_dist, num_samples=b, replacement=True)
        out_dict = {'y': y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]
            if self.num_numerical_features > 0:
                z_norm = self._gaussian_p_sample(model_out_num, z_norm, t)['sample']
            if has_cat:
                log_z = self._p_sample(model_out_cat, log_z, t)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    def _sample_all(self, num_samples, batch_size, y_dist, verbose=False):
        all_y = []
        all_samples = []
        num_generated = 0
        max_attempts = 10 * math.ceil(num_samples / batch_size) + 10
        attempts = 0
        while num_generated < num_samples:
            if attempts >= max_attempts:
                raise FoundNANsError(
                    'Sampling keeps producing NaNs; the model may be undertrained '
                    'or the learning rate too high.'
                )
            attempts += 1

            b = min(batch_size, num_samples - num_generated)
            sample, out_dict = self._sample(b, y_dist)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            y = out_dict['y'][~mask_nan]

            all_samples.append(sample)
            all_y.append(y.cpu())
            num_generated += sample.shape[0]
            if verbose:
                sys.stdout.write(f'Sampled {min(num_generated, num_samples)}/{num_samples} rows\n')

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]
        y_gen = torch.cat(all_y, dim=0)[:num_samples]

        return x_gen, y_gen


class _DataTransformer:
    """Converts a DataFrame into (numerical block, categorical index block) and back."""

    _NUMERICAL_ROLES = ('numerical', 'datetime')

    def __init__(self, columns_metadata, normalization='quantile', seed=0):
        self._columns_metadata = columns_metadata
        self._normalization = normalization
        self._seed = seed

        self._roles = {}
        for column, spec in columns_metadata.items():
            sdtype = spec.get('sdtype', 'categorical')
            role = sdtype
            if spec.get('pii', False):
                # PII columns are not modelled; placeholders are regenerated
                # at sampling time (like SDV's anonymization behavior).
                role = 'id'
            elif role not in ('numerical', 'datetime', 'boolean', 'id'):
                # 'categorical' and any unrecognized sdtype
                role = 'categorical'
            self._roles[column] = role

        self.num_columns: List[str] = []
        self.cat_columns: List[str] = []
        self.id_columns: List[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        for column in df.columns:
            role = self._roles[column]
            if role in self._NUMERICAL_ROLES:
                self.num_columns.append(column)
            elif role == 'id':
                self.id_columns.append(column)
            else:
                self.cat_columns.append(column)

        self._dtypes = {column: df[column].dtype for column in df.columns}
        self._id_is_numeric = {
            column: pd.api.types.is_numeric_dtype(df[column]) for column in self.id_columns
        }

        # Numerical block: raw floats (datetimes as epoch nanoseconds)
        X_num = self._to_numeric_block(df)

        self._num_means = None
        self._num_transform = None
        self._disc_uniques = {}
        if X_num.shape[1] > 0:
            col_means = np.nanmean(X_num, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            self._num_means = col_means
            inds = np.where(np.isnan(X_num))
            X_num[inds] = np.take(col_means, inds[1])

            # Columns holding few unique integer values are snapped back to the
            # observed values after sampling.
            for j, column in enumerate(self.num_columns):
                if self._roles[column] != 'numerical':
                    continue
                uniq = np.unique(X_num[:, j])
                if len(uniq) <= 32 and np.allclose(uniq, np.round(uniq)):
                    self._disc_uniques[j] = uniq

            if self._normalization is not None:
                self._num_transform = QuantileTransformer(
                    output_distribution='normal',
                    n_quantiles=max(min(X_num.shape[0] // 30, 1000), 10),
                    subsample=int(1e9),
                    random_state=self._seed,
                )
                self._num_transform.fit(X_num)

        # Categorical block: ordinal codes ('__nan__' is its own category)
        self._cat_transform = None
        self.category_sizes = np.array([0])
        if self.cat_columns:
            X_cat = self._to_categorical_block(df)
            self._cat_transform = OrdinalEncoder(dtype='int64')
            self._cat_transform.fit(X_cat)
            self.category_sizes = np.array([
                len(categories) for categories in self._cat_transform.categories_
            ])

    def transform(self, df: pd.DataFrame):
        X_num = self._to_numeric_block(df)
        if X_num.shape[1] > 0:
            inds = np.where(np.isnan(X_num))
            X_num[inds] = np.take(self._num_means, inds[1])
            if self._num_transform is not None:
                X_num = self._num_transform.transform(X_num)

        X_cat = np.empty((len(df), 0), dtype='int64')
        if self.cat_columns:
            X_cat = self._cat_transform.transform(self._to_categorical_block(df))

        return X_num.astype('float32'), X_cat

    def inverse_transform(self, X_num: np.ndarray, X_cat: np.ndarray) -> pd.DataFrame:
        n_rows = max(X_num.shape[0], X_cat.shape[0])
        columns = {}

        if self.num_columns:
            if self._num_transform is not None:
                X_num = self._num_transform.inverse_transform(X_num)
            for j, uniq in self._disc_uniques.items():
                dist = np.abs(X_num[:, j][:, None] - uniq[None, :])
                X_num[:, j] = uniq[dist.argmin(axis=1)]
            for j, column in enumerate(self.num_columns):
                columns[column] = self._from_numeric_column(column, X_num[:, j])

        if self.cat_columns:
            decoded = self._cat_transform.inverse_transform(np.round(X_cat).astype('int64'))
            for j, column in enumerate(self.cat_columns):
                columns[column] = self._from_categorical_column(column, decoded[:, j])

        for column in self.id_columns:
            if self._id_is_numeric[column]:
                columns[column] = np.arange(n_rows)
            else:
                columns[column] = np.array([f'{column}_{i}' for i in range(n_rows)])

        return pd.DataFrame(columns)

    def _to_numeric_block(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for column in self.num_columns:
            if self._roles[column] == 'datetime':
                fmt = self._columns_metadata[column].get('datetime_format')
                series = pd.to_datetime(df[column], format=fmt, errors='coerce')
                values = series.to_numpy().astype('int64').astype('float64')
                values[series.isna().to_numpy()] = np.nan
            else:
                values = pd.to_numeric(df[column], errors='coerce').astype('float64').to_numpy()
            parts.append(values)
        if not parts:
            return np.empty((len(df), 0), dtype='float64')
        return np.column_stack(parts)

    def _to_categorical_block(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for column in self.cat_columns:
            series = df[column]
            values = series.astype(object).where(series.notna(), CAT_MISSING_VALUE).astype(str)
            parts.append(values.to_numpy())
        return np.column_stack(parts)

    def _from_numeric_column(self, column: str, values: np.ndarray):
        if self._roles[column] == 'datetime':
            stamps = pd.to_datetime(np.round(values).astype('int64'))
            fmt = self._columns_metadata[column].get('datetime_format')
            if fmt is not None:
                return stamps.strftime(fmt)
            return stamps
        dtype = self._dtypes[column]
        if pd.api.types.is_integer_dtype(dtype):
            return np.round(values).astype(dtype)
        return values.astype('float64')

    def _from_categorical_column(self, column: str, values: np.ndarray):
        series = pd.Series(values, dtype=object)
        series = series.where(series != CAT_MISSING_VALUE, np.nan)
        if self._roles[column] == 'boolean':
            return series.map({'True': True, 'False': False})
        try:
            return series.astype(self._dtypes[column])
        except (ValueError, TypeError):
            return series


class _FastTensorDataLoader:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = min(batch_size, X.shape[0])

    def __iter__(self):
        while True:
            perm = torch.randperm(self.X.shape[0])
            X, y = self.X[perm], self.y[perm]
            for i in range(0, X.shape[0], self.batch_size):
                yield X[i : i + self.batch_size], y[i : i + self.batch_size]


class Trainer:
    """Diffusion trainer."""

    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device, verbose=True):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = iter(train_iter)
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        self.verbose = verbose

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion._mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()
        return loss_multi, loss_gauss

    def run_loop(self):
        """Training loop."""
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if self.verbose and (step + 1) % self.print_every == 0:
                    sys.stdout.write(
                        f'Step {(step + 1)}/{self.steps} '
                        f'MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}\n'
                    )
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    mloss,
                    gloss,
                    mloss + gloss,
                ]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1


class TabDDPM:
    """Single table synthesizer based on TabDDPM.

    Args:
        metadata (sdv.metadata.Metadata):
            The metadata describing the data.
        target_column (str or None):
            Categorical/boolean column to condition the diffusion on (the paper's
            class-conditional setup for classification datasets). ``None`` trains
            unconditionally.
        d_layers (List[int]):
            Hidden layer sizes of the MLP denoiser.
        dropout (float):
            Dropout of the MLP denoiser.
        dim_t (int):
            Timestep/label embedding dimension.
        num_timesteps (int):
            Diffusion timesteps T.
        scheduler (str):
            ``'cosine'`` or ``'linear'`` beta schedule.
        steps (int):
            Training iterations.
        lr (float):
            Learning rate for the optimizer.
        weight_decay (float):
            Weight decay for the optimizer.
        batch_size (int):
            Batch size.
        normalization (str or None):
            ``'quantile'`` (paper default) or ``None`` to skip normalizing numerical features.
        sample_batch_size (int):
            Batch size used during sampling.
        device (str or None):
            Whether to use ``'cuda'`` or ``'cpu'``. If None, auto-select is used.
        seed (int):
            Random seed used for fitting.
        verbose (bool):
            Print training / sampling progress.
    """

    def __init__(
        self,
        metadata,
        target_column=None,
        d_layers=None,
        dropout=0.0,
        dim_t=128,
        num_timesteps=1000,
        scheduler='cosine',
        steps=1000,
        lr=0.001,
        weight_decay=1e-5,
        batch_size=4096,
        normalization='quantile',
        sample_batch_size=10000,
        device=None,
        seed=0,
        verbose=False,
    ):
        self.target_column = target_column
        self.d_layers = d_layers or [256, 256]
        self.dropout = dropout
        self.dim_t = dim_t
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.normalization = normalization
        self.sample_batch_size = sample_batch_size
        self.seed = seed
        self.verbose = verbose

        if isinstance(metadata, dict):
            metadata = Metadata.load_from_dict(metadata)

        metadata.validate()
        self._metadata = metadata
        self._table_name = list(metadata.tables)[0]
        self._table_metadata = metadata.tables[self._table_name].to_dict()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)
        self._fitted = False

    def fit(self, data):
        """Fit the synthesizer on real data.

        Args:
            data (pandas.DataFrame):
                The data to fit the synthesizer to.
        """
        data = data.copy()
        if isinstance(data, pd.DataFrame):
            data_dict = {self._table_name: data}

        self._metadata.validate_data(data_dict)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Target column for class-conditional training (paper's setup for
        # classification datasets); everything else goes through the transformer.
        target_column = self.target_column
        columns_metadata = dict(self._table_metadata['columns'])
        self._n_target_classes = 0
        self._target_encoder = None
        if target_column is not None:
            if target_column not in data.columns:
                raise ValueError(f"target_column '{target_column}' not found in the data.")
            sdtype = columns_metadata[target_column].get('sdtype', 'categorical')
            if sdtype not in ('categorical', 'boolean'):
                raise ValueError(
                    f"target_column '{target_column}' must be categorical or boolean "
                    f'(got sdtype={sdtype!r}). Numerical columns are modelled '
                    'unconditionally; leave target_column unset.'
                )
            target_series = data[target_column]
            target_values = (
                target_series
                .astype(object)
                .where(target_series.notna(), CAT_MISSING_VALUE)
                .astype(str)
                .to_numpy()
                .reshape(-1, 1)
            )
            self._target_encoder = OrdinalEncoder(dtype='int64')
            y = self._target_encoder.fit_transform(target_values).reshape(-1)
            self._n_target_classes = len(self._target_encoder.categories_[0])
            self._target_dtype = target_series.dtype
            self._target_is_boolean = sdtype == 'boolean'
            columns_metadata.pop(target_column)
            data = data.drop(columns=[target_column])
        else:
            y = np.zeros(len(data), dtype='int64')

        # Preprocess: quantile-normalized numerical block + ordinal categorical block
        self._transformer = _DataTransformer(
            columns_metadata, normalization=self.normalization, seed=self.seed
        )
        self._transformer.fit(data)
        X_num, X_cat = self._transformer.transform(data)

        n_num = X_num.shape[1]
        K = self._transformer.category_sizes
        d_in = int(n_num + K.sum())
        if d_in == 0:
            raise ValueError('The table has no modelable columns (only id columns?).')

        # Empirical distribution of the conditioning label (uniform over the
        # single dummy class when training unconditionally).
        y_tensor = torch.from_numpy(y)
        self._y_dist = torch.bincount(y_tensor).float()

        # Build the denoiser and the joint Gaussian/multinomial diffusion
        model = MLPDiffusion(
            d_in=d_in,
            num_classes=self._n_target_classes,
            is_y_cond=target_column is not None,
            rtdl_params={'d_layers': list(self.d_layers), 'dropout': self.dropout},
            dim_t=self.dim_t,
        ).to(self._device)

        self._diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=n_num,
            denoise_fn=model,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self._device,
        ).to(self._device)
        self._diffusion.train()

        X = torch.from_numpy(np.concatenate([X_num, X_cat.astype('float32')], axis=1)).float()
        train_loader = _FastTensorDataLoader(X, y_tensor, batch_size=self.batch_size)

        trainer = Trainer(
            self._diffusion,
            train_loader,
            lr=self.lr,
            weight_decay=self.weight_decay,
            steps=self.steps,
            device=self._device,
            verbose=self.verbose,
        )
        trainer.run_loop()
        self.loss_history = pd.DataFrame(trainer.loss_history)

        self._column_order = list(self._table_metadata['columns'])
        self._fitted = True

    def sample(self, num_rows):
        """Sample synthetic rows from the fitted synthesizer.

        Args:
            num_rows (int):
                Amount of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if not self._fitted:
            raise RuntimeError('The synthesizer has not been fitted; call fit() first.')
        if num_rows <= 0:
            raise ValueError('num_rows must be a positive integer.')

        self._diffusion.eval()
        x_gen, y_gen = self._diffusion._sample_all(
            num_rows,
            batch_size=self.sample_batch_size,
            y_dist=self._y_dist,
            verbose=self.verbose,
        )

        X_gen = x_gen.numpy()
        n_num = self._diffusion.num_numerical_features
        X_num = X_gen[:, :n_num]
        has_cat = self._transformer.category_sizes[0] != 0
        X_cat = X_gen[:, n_num:] if has_cat else np.empty((num_rows, 0), dtype='int64')

        df = self._transformer.inverse_transform(X_num, X_cat)

        if self._target_encoder is not None:
            decoded = self._target_encoder.inverse_transform(y_gen.numpy().reshape(-1, 1)).reshape(
                -1
            )
            series = pd.Series(decoded, dtype=object)
            series = series.where(series != CAT_MISSING_VALUE, np.nan)
            if self._target_is_boolean:
                series = series.map({'True': True, 'False': False})
            else:
                try:
                    series = series.astype(self._target_dtype)
                except (ValueError, TypeError):
                    pass
            df[self.target_column] = series

        df = df[[column for column in self._column_order if column in df.columns]]
        return df


class TabDDPMSynthesizer(BaselineSynthesizer):
    """Custom wrapper for the TabDDPM synthesizer to make it work with SDGym."""

    LOGGER = logging.getLogger(__name__)
    _MODEL_KWARGS = None
    _MODALITY_FLAG = 'single_table'

    def _fit(self, data, metadata):
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = TabDDPM(metadata, **model_kwargs)
        model.fit(data)

        self._internal_synthesizer = model

    def _sample_from_synthesizer(self, synthesizer, n_sample):
        """Sample synthetic data with specified sample count."""
        return synthesizer._internal_synthesizer.sample(n_sample)
