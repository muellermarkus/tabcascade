import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class TimeStepEmbedding(nn.Module):
    """
    Layer that embeds diffusion timesteps.

     Args:
        - dim (int): the dimension of the output.
        - max_period (int): controls the minimum frequency of the embeddings.
        - n_layers (int): number of dense layers
        - fourer (bool): whether to use random fourier features as embeddings
    """

    def __init__(
        self,
        dim: int,
        max_period: int = 10000,
        n_layers: int = 2,
        fourier: bool = False,
        scale=16,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.n_layers = n_layers
        self.fourier = fourier

        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")

        if fourier:
            self.register_buffer("freqs", torch.randn(dim // 2) * scale)

        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.SiLU())
        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim))

    def forward(self, timesteps):
        if not self.fourier:
            d, T = self.dim, self.max_period
            mid = d // 2
            fs = torch.exp(-math.log(T) / mid * torch.arange(mid, dtype=torch.float32))
            fs = fs.to(timesteps.device)
            args = timesteps[:, None].float() * fs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            x = timesteps.ger((2 * torch.pi * self.freqs).to(timesteps.dtype))
            emb = torch.cat([x.cos(), x.sin()], dim=1)

        return self.fc(emb)


def normalize_emb(emb, dim):
    return F.normalize(emb, dim=dim, eps=1e-20)


class CatEmbedding(nn.Module):
    """
    Feature-specific embedding layer for categorical features.
    bias = True adds a learnable bias term to each feature, which is is same across categories.
    """

    def __init__(self, dim, categories, cat_emb_init_sigma=0.001, bias=False, normalize_emb=False, norm_dim=None):
        super().__init__()

        self.categories = torch.tensor(categories)
        categories_offset = self.categories.cumsum(dim=-1)[:-1]
        categories_offset = torch.cat((torch.zeros((1,), dtype=torch.long), categories_offset))
        self.register_buffer("categories_offset", categories_offset)

        if norm_dim is None:
            self.dim = torch.tensor(dim)
        else:
            self.dim = torch.tensor(norm_dim).pow(2)

        self.normalize_emb = normalize_emb

        self.cat_emb = nn.Embedding(sum(categories), dim)
        nn.init.normal_(self.cat_emb.weight, std=cat_emb_init_sigma)

        self.bias = bias
        if self.bias:
            self.cat_bias = nn.Parameter(torch.zeros(len(categories), dim))

    def forward(self, x):
        x = self.cat_emb(x + self.categories_offset)
        if self.bias:
            x += self.cat_bias
        if self.normalize_emb:
            x = normalize_emb(x, dim=2) * self.dim.sqrt()
        return x

    def get_all_feat_emb(self, feat_idx):
        emb_idx = (
            torch.arange(self.categories[feat_idx], device=self.cat_emb.weight.device)
            + self.categories_offset[feat_idx]
        )
        x = self.cat_emb(emb_idx)
        if self.bias:
            x += self.cat_bias[feat_idx]

        if self.normalize_emb:
            x = normalize_emb(x, dim=1) * self.dim.sqrt()
        return x


class FourierFeatures(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        assert (emb_dim % 2) == 0
        self.half_dim = emb_dim // 2
        self.register_buffer("weights", torch.randn(1, self.half_dim))

    def forward(self, x):
        freqs = x.unsqueeze(1) * self.weights * 2 * np.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class WeightNetwork(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.fourier = FourierFeatures(emb_dim)
        self.fc = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, u):
        x = self.fourier(u)
        return self.fc(x).squeeze()

    def loss_fn(self, preds, avg_loss):
        # learn to fit expected average loss
        return (preds - avg_loss) ** 2


class NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings.

    From: https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py

    In other words,
    each feature embedding is transformed by its own dedicated linear layer.
    """

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming uniform, replicating from nn.Linear
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]
        x = x.transpose(0, 1)  # (n_features, batch_size, dim)
        # use broadcasting, it automatically does self.weight.unsqueeze(1)
        x = x @ self.weight  # (n, B, in_dim) x (n, in_dim, out_dim)
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x

    def forward_single(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Forward pass for a single feature."""
        assert x.shape[-1] == self.weight.shape[-2]
        x = x @ self.weight[i]  # (B, in_dim) x (in_dim, out_dim)
        if self.bias is not None:
            x = x + self.bias[i]
        return x


class Timewarp(nn.Module):
    """
    Timewarping based on a piece-wise linear learnable function
    as proposed in the CDCD paper (Dieleman et al., 2022).

    timewarp_type selects the type of timewarping:
        - single (single noise schedule, like CDCD)
        - bytype (per type noise schedule)
        - all (per feature noise schedule)
    """

    def __init__(
        self,
        sigma_min,
        sigma_max,
        num_bins=100,
        decay=0.1,
    ):
        super(Timewarp, self).__init__()
        self.num_bins = num_bins
        self.register_buffer("sigma_min", torch.tensor(sigma_min))
        self.register_buffer("sigma_max", torch.tensor(sigma_max))

        self.logits_t = nn.Parameter(torch.full((1, num_bins), -torch.tensor(num_bins).log()))
        self.logits_u = nn.Parameter(torch.full((1, num_bins), -torch.tensor(num_bins).log()))

        # copy parameters to keep EMA
        self.decay = decay
        logits_t_shadow = torch.clone(self.logits_t).detach()
        logits_u_shadow = torch.clone(self.logits_u).detach()
        self.register_buffer("logits_t_shadow", logits_t_shadow)
        self.register_buffer("logits_u_shadow", logits_u_shadow)

    def update_ema(self):
        with torch.no_grad():
            self.logits_t.copy_(self.decay * self.logits_t_shadow + (1 - self.decay) * self.logits_t.detach())
            self.logits_u.copy_(self.decay * self.logits_u_shadow + (1 - self.decay) * self.logits_u.detach())
            self.logits_t_shadow.copy_(self.logits_t)
            self.logits_u_shadow.copy_(self.logits_u)

    def get_bins(self, invert, normalize):
        if not invert:
            logits_t = self.logits_t
            logits_u = self.logits_u
        else:
            normalize = True
            # we can invert by simply switching the roles of the logits
            logits_t = self.logits_u
            logits_u = self.logits_t

        if normalize:
            weights_u = F.softmax(logits_u, dim=1)
        else:
            weights_u = logits_u.exp()
        weights_t = F.softmax(logits_t, dim=1)

        # add small constant to each bin size and renormalize
        weights_u = weights_u + 1e-7
        if normalize:
            weights_u = weights_u / weights_u.sum(dim=1, keepdims=True)
        weights_t = weights_t + 1e-7
        weights_t = weights_t / weights_t.sum(dim=1, keepdims=True)

        # get edge values and slopes
        edges_t_right = torch.cumsum(weights_t, dim=1)
        edges_u_right = torch.cumsum(weights_u, dim=1)
        edges_t_left = F.pad(edges_t_right[:, :-1], (1, 0), value=0)
        edges_u_left = F.pad(edges_u_right[:, :-1], (1, 0), value=0)
        slopes = weights_u / weights_t

        return edges_t_left, edges_t_right, edges_u_left, edges_u_right, slopes

    def forward(self, x, invert=False, normalize=False, return_pdf=False):
        edges_t_left, edges_t_right, edges_u_left, _, slopes = self.get_bins(invert=invert, normalize=normalize)

        if not invert:
            # scale sigmas to [0,1]
            x = (x - self.sigma_min) / (self.sigma_max - self.sigma_min)

        bin_idx = torch.searchsorted(edges_t_right, x.unsqueeze(0).contiguous(), right=False)
        bin_idx[bin_idx > self.num_bins - 1] = self.num_bins - 1

        slope = slopes.gather(dim=1, index=bin_idx)  # num_cdfs, batch
        left_t = edges_t_left.gather(dim=1, index=bin_idx)
        left_u = edges_u_left.gather(dim=1, index=bin_idx)

        if return_pdf:
            return slope.T.squeeze(1).detach()

        # linearly interpolate bin edges
        interpolation = (left_u + (x - left_t) * slope).T.squeeze(1)

        if normalize:
            interpolation = torch.clamp(interpolation, 0, 1)

        if invert:
            interpolation = interpolation * (self.sigma_max - self.sigma_min) + self.sigma_min

        return interpolation

    def get_sigmas(self, u):
        return self.forward(u, invert=True, normalize=True).to(torch.float32)

    def get_t(self, sigma):
        return self.forward(sigma, invert=False, normalize=True).to(torch.float32)

    def loss_fn(self, sigmas, losses):
        losses_estimated = self.forward(sigmas)

        with torch.no_grad():
            pdf = self.forward(sigmas, return_pdf=True, normalize=True).detach()

        return ((losses_estimated - losses) ** 2) / (pdf + 1e-8)


class Timewarp_Logistic(nn.Module):
    """
    Our version of timewarping with exact cdfs instead of p.w.l. functions.
    We use a domain-adapted cdf of the logistic distribution.

    timewarp_type selects the type of timewarping:
        - single (single noise schedule, like CDCD)
        - bytype (per type noise schedule)
        - all (per feature noise schedule)
    """

    def __init__(
        self,
        timewarp_type,
        num_cat_features,
        num_cont_features,
        sigma_min,
        sigma_max,
        weight_low_noise=1.0,
        decay=0.0,
    ):
        super(Timewarp_Logistic, self).__init__()

        self.timewarp_type = timewarp_type
        self.num_cat_features = num_cat_features
        self.num_cont_features = num_cont_features
        self.num_features = num_cat_features + num_cont_features

        # save bounds for min max scaling
        self.register_buffer("sigma_min", sigma_min)
        self.register_buffer("sigma_max", sigma_max)

        if timewarp_type == "single":
            self.num_funcs = 1
        elif timewarp_type == "bytype":
            self.num_funcs = 2
        elif timewarp_type == "all":
            self.num_funcs = self.num_cat_features + self.num_cont_features

        # init parameters
        v = torch.tensor(1.01)
        logit_v = torch.log(torch.exp(v - 1) - 1)
        self.logits_v = nn.Parameter(torch.full((self.num_funcs,), fill_value=logit_v))
        self.register_buffer("init_v", self.logits_v.clone())

        p_large_noise = torch.tensor(1 / (weight_low_noise + 1))
        logit_mu = torch.log(((1 / (1 - p_large_noise)) - 1)) / v
        self.logits_mu = nn.Parameter(torch.full((self.num_funcs,), fill_value=logit_mu))
        self.register_buffer("init_mu", self.logits_mu.clone())

        # init gamma, scaling parameter to 1
        self.logits_gamma = nn.Parameter((torch.ones((self.num_funcs, 1)).exp() - 1).log())

        # for ema
        self.decay = decay
        logits_v_shadow = torch.clone(self.logits_v).detach()
        logits_mu_shadow = torch.clone(self.logits_mu).detach()
        logits_gamma_shadow = torch.clone(self.logits_gamma).detach()
        self.register_buffer("logits_v_shadow", logits_v_shadow)
        self.register_buffer("logits_mu_shadow", logits_mu_shadow)
        self.register_buffer("logits_gamma_shadow", logits_gamma_shadow)

    def update_ema(self):
        with torch.no_grad():
            self.logits_v.copy_(self.decay * self.logits_v_shadow + (1 - self.decay) * self.logits_v.detach())
            self.logits_mu.copy_(self.decay * self.logits_mu_shadow + (1 - self.decay) * self.logits_mu.detach())
            self.logits_gamma.copy_(
                self.decay * self.logits_gamma_shadow + (1 - self.decay) * self.logits_gamma.detach()
            )
            self.logits_v_shadow.copy_(self.logits_v)
            self.logits_mu_shadow.copy_(self.logits_mu)
            self.logits_gamma_shadow.copy_(self.logits_gamma)

    def get_params(self):
        logit_mu = self.logits_mu  # let underlying parameter be ln(mu / (1-mu))
        v = 1 + F.softplus(self.logits_v)  # v > 1
        scale = F.softplus(self.logits_gamma)
        return logit_mu, v, scale

    def cdf_fn(self, x, logit_mu, v):
        "mu in (0,1), v >= 1"
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v)
        return 1 / (1 + Z)

    def pdf_fn(self, x, logit_mu, v):
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v)
        return (v / (x * (1 - x))) * (Z / ((1 + Z) ** 2))

    def quantile_fn(self, u, logit_mu, v):
        c = logit_mu + 1 / v * torch.special.logit(u, eps=1e-7)
        return F.sigmoid(c)

    def forward(self, x, invert=False, normalize=False, return_pdf=False):
        logit_mu, v, scale = self.get_params()

        if not invert:
            if normalize:
                scale = 1.0

            # can have more sigmas than cdfs
            x = (x - self.sigma_min) / (self.sigma_max - self.sigma_min)

            # ensure x is never 0 or 1 to ensure robustness
            x = torch.clamp(x, 1e-7, 1 - 1e-7)

            if self.timewarp_type == "single":
                # all sigmas are the same so just take first one
                input = x[:, 0].unsqueeze(0)

            elif self.timewarp_type == "bytype":
                # first sigma belongs to categorical feature, last to continuous feature
                input = torch.stack((x[:, 0], x[:, -1]), dim=0)

            elif self.timewarp_type == "all":
                input = x.T  # (num_features, batch)

            if return_pdf:
                output = (torch.vmap(self.pdf_fn, in_dims=0)(input, logit_mu, v)).T
            else:
                output = (torch.vmap(self.cdf_fn, in_dims=0)(input, logit_mu, v) * scale).T

        else:
            # have single u, need to repeat u
            input = repeat(x, "b -> f b", f=self.num_funcs)
            output = (torch.vmap(self.quantile_fn, in_dims=0)(input, logit_mu, v)).T

            if self.timewarp_type == "single":
                output = repeat(output, "b 1 -> b f", f=self.num_features)
            elif self.timewarp_type == "bytype":
                output = torch.column_stack(
                    (
                        repeat(output[:, 0], "b -> b f", f=self.num_cat_features),
                        repeat(output[:, 1], "b -> b f", f=self.num_cont_features),
                    )
                )

            zero_mask = x == 0.0
            one_mask = x == 1.0
            output = output.masked_fill(zero_mask.unsqueeze(-1), 0.0)
            output = output.masked_fill(one_mask.unsqueeze(-1), 1.0)

            output = output * (self.sigma_max - self.sigma_min) + self.sigma_min

        return output

    def get_sigmas(self, t):
        return self.forward(t, invert=True, normalize=True).to(torch.float32)

    def loss_fn(self, sigmas, losses):
        # losses and sigmas have shape (B, num_features)

        if self.timewarp_type == "single":
            # fit average loss (over all feature)
            losses = losses.mean(1, keepdim=True)  # (B,1)
        elif self.timewarp_type == "bytype":
            # fit average loss over cat and over cont features separately
            losses_cat = losses[:, : self.num_cat_features].mean(1, keepdim=True)  # (B,1)
            losses_cont = losses[:, self.num_cat_features :].mean(1, keepdim=True)  # (B,1)
            losses = torch.cat((losses_cat, losses_cont), dim=1)

        losses_estimated = self.forward(sigmas)

        with torch.no_grad():
            pdf = self.forward(sigmas, return_pdf=True).detach()

        return ((losses_estimated - losses) ** 2) / (pdf + 1e-7)


class FinalLayer(nn.Module):
    """
    Final layer that predicts logits for each category for categorical features
    and scalers for continuous features.
    """

    def __init__(self, dim_in, categories, num_cont_features, bias_init=None):
        super().__init__()
        self.num_cont_features = num_cont_features
        self.num_cat_features = len(categories)
        dim_out = sum(categories) + self.num_cont_features
        self.linear = nn.Linear(dim_in, dim_out)
        nn.init.zeros_(self.linear.weight)
        if bias_init is None:
            nn.init.zeros_(self.linear.bias)
        else:
            self.linear.bias = nn.Parameter(bias_init)
        self.split_chunks = [self.num_cont_features, *categories]

    def forward(self, x):
        x = self.linear(x)
        out = torch.split(x, self.split_chunks, dim=-1)

        if self.num_cont_features > 0:
            cont_logits = out[0]
        else:
            cont_logits = None
        if self.num_cat_features > 0:
            cat_logits = out[1:]
        else:
            cat_logits = None

        return cat_logits, cont_logits


class LowResMLP(nn.Module):
    def __init__(self, num_classes, cat_emb_dim, emb_dim, n_layers, n_units, proportions, act="relu"):
        super().__init__()

        self.num_features = len(num_classes)
        self.num_classes = num_classes
        self.time_emb = TimeStepEmbedding(emb_dim)
        self.proj = nn.Linear(self.num_features * cat_emb_dim, emb_dim)

        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = (n_layers - 1) * [n_units] + [emb_dim]
        layers = nn.ModuleList()
        for i in range(len(in_dims)):
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        self.mlp = nn.Sequential(*layers)

        # init final layer
        if proportions is not None:
            bias_init = torch.cat(proportions).log()
        else:
            bias_init = None

        self.final_layer = FinalLayer(out_dims[-1], num_classes, 0, bias_init=bias_init)

    def forward(self, x_emb_t, t):

        t_emb = self.time_emb(t)
        x = rearrange(x_emb_t, "B F D -> B (F D)")
        x = self.proj(x) + t_emb
        x = self.mlp(x)
        logits, _ = self.final_layer(x)

        return logits
