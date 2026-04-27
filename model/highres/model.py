import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .utils import low_discrepancy_sampler, set_seeds


class PolyNoiseSchedule(nn.Module):
    def __init__(self, emb_dim, num_features, gamma_min=0.0, gamma_max=1.0, grad_min_epsilon=0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_range = self.gamma_max - self.gamma_min
        self.grad_min_epsilon = grad_min_epsilon

        self.h_net = nn.Sequential(
            nn.Linear(emb_dim, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
            nn.SiLU(),
        )
        self.l_a = nn.Linear(num_features, num_features)
        nn.init.zeros_(self.l_a.weight)
        nn.init.zeros_(self.l_a.bias)
        self.l_b = nn.Linear(num_features, num_features)
        self.l_c = nn.Linear(num_features, num_features)

    def forward(self, emb, t):
        if t.numel() == 1:
            # scalar
            t = t * torch.ones((emb.shape[0], 1), device=emb.device)
        else:
            t = t.unsqueeze(-1)

        assert len(emb.shape) == 2
        assert emb.shape[0] == t.shape[0]

        a, b, c = self.get_params(emb)
        return self._eval_poly(t, a, b, c)

    def get_grads(self, emb, t):
        t = t.unsqueeze(-1)
        assert len(emb.shape) == 2
        assert emb.shape[0] == t.shape[0]
        a, b, c = self.get_params(emb)
        return self._grad_t(t, a, b, c)

    def get_params(self, emb):
        h = self.h_net(emb)
        a = self.l_a(h)
        b = self.l_b(h)
        c = 1e-3 + F.softplus(self.l_c(h))
        return a, b, c

    def _eval_poly(self, t, a, b, c):
        polynomial = (
            (a**2) * (t**5) / 5.0
            + (b**2 + 2 * a * c) * (t**3) / 3.0
            + a * b * (t**4) / 2.0
            + b * c * (t**2)
            + (c**2 + self.grad_min_epsilon) * t
        )
        scale = (a**2) / 5.0 + (b**2 + 2 * a * c) / 3.0 + a * b / 2.0 + b * c + (c**2 + self.grad_min_epsilon)
        return self.gamma_min + self.gamma_range * polynomial / scale

    def _grad_t(self, t, a, b, c):
        polynomial = (a**2) * (t**4) + (b**2 + 2 * a * c) * (t**2) + a * b * (t**3) * 2.0 + b * c * t * 2 + (c**2)
        scale = (a**2) / 5.0 + (b**2 + 2 * a * c) / 3.0 + a * b / 2.0 + b * c + (c**2)
        return self.gamma_range * polynomial / scale


class CatEmbedding(nn.Module):
    """
    Feature-specific embedding layer for categorical features.
    bias = True adds a learnable bias term to each feature, which is the same across categories.
    """

    def __init__(self, dim, categories, cat_emb_init_sigma=0.001, bias=False):
        super().__init__()

        self.categories = torch.tensor(categories)
        categories_offset = self.categories.cumsum(dim=-1)[:-1]
        categories_offset = torch.cat((torch.zeros((1,), dtype=torch.long), categories_offset))
        self.register_buffer("categories_offset", categories_offset)

        self.cat_emb = nn.Embedding(sum(categories), dim)
        nn.init.normal_(self.cat_emb.weight, std=cat_emb_init_sigma)

        self.bias = bias
        if self.bias:
            self.cat_bias = nn.Parameter(torch.zeros(len(categories), dim))

    def forward(self, x):
        x = self.cat_emb(x + self.categories_offset)
        if self.bias:
            x += self.cat_bias
        return x


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


class HighResMLP(nn.Module):
    def __init__(self, num_features, x_low_dim, emb_dim, n_layers, n_units, act="relu"):
        super().__init__()

        self.num_features = num_features
        self.time_emb = TimeStepEmbedding(emb_dim)
        self.proj = nn.Linear(self.num_features, emb_dim)
        # self.proj_low = nn.Linear(x_low_dim, emb_dim)
        self.proj_low = nn.Sequential(
            nn.Linear(x_low_dim, 2 * emb_dim),
            nn.SiLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )

        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = (n_layers - 1) * [n_units] + [emb_dim]
        layers = nn.ModuleList()
        for i in range(len(in_dims)):
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())

        self.mlp = nn.Sequential(*layers)
        self.final_layer = nn.Linear(out_dims[-1], self.num_features)

    def forward(self, x_t, x_low, t):
        c_noise = torch.log(t + 1e-8) * 0.25 * 1000  # from CDTD / EDM
        t_emb = self.time_emb(c_noise)
        x_low_cond_1 = self.proj_low(x_low)
        x = self.proj(x_t) + t_emb + x_low_cond_1
        x = self.mlp(x)
        return self.final_layer(x)


class HighResFlowModel(nn.Module):
    def __init__(
        self,
        group_means,
        group_stds,
        categories,
        emb_dim,
        n_layers,
        n_units,
        gamma_input_dim,
        cat_emb_dim,
    ):
        super().__init__()
        self.num_features = len(group_means)

        # init embeddings that allow for efficient retrieval of group moments
        n_groups = torch.tensor([len(m) for m in group_means])
        group_offset = n_groups.cumsum(dim=-1)[:-1]
        group_offset = torch.cat((torch.zeros((1,), dtype=torch.long), group_offset))
        self.register_buffer("group_offset", group_offset)

        self.get_group_means = nn.Embedding.from_pretrained(torch.cat(group_means).unsqueeze(-1), freeze=True)
        self.get_group_stds = nn.Embedding.from_pretrained(torch.cat(group_stds).unsqueeze(-1), freeze=True)

        self.emb = CatEmbedding(cat_emb_dim, categories, cat_emb_init_sigma=1)
        self.proj_to_gamma = nn.Sequential(
            nn.Linear(len(categories) * cat_emb_dim, 2 * gamma_input_dim),
            nn.SiLU(),
            nn.Linear(2 * gamma_input_dim, gamma_input_dim),
            nn.SiLU(),
        )
        self.gamma = PolyNoiseSchedule(gamma_input_dim, self.num_features)

        # init parameterized vector field
        x_low_dim = len(categories) * cat_emb_dim
        self.mlp = HighResMLP(self.num_features, x_low_dim, emb_dim=emb_dim, n_layers=n_layers, n_units=n_units)

    @property
    def device(self):
        return next(self.mlp.parameters()).device

    def loss_fn(self, x_num, x_cat, z_num, mask):
        assert x_num.shape == z_num.shape

        # retrieve groups and sample from feature-specific source distributions
        means = self.get_group_means(z_num + self.group_offset).squeeze(-1)
        stds = self.get_group_stds(z_num + self.group_offset).squeeze(-1)

        # coupling
        x_1 = x_num
        x_0 = means + stds * torch.randn_like(x_num)

        # derive gamma_t and its time-derivative
        d_cat = torch.column_stack((x_cat, z_num)) if x_cat is not None else z_num
        x_low = self.emb(d_cat).flatten(1)
        e_gamma = self.proj_to_gamma(x_low)
        t = low_discrepancy_sampler(x_num.shape[0], device=x_num.device).to(torch.float32)

        gamma_t = self.gamma(e_gamma, t)
        gamma_t_grad = self.gamma.get_grads(e_gamma, t)

        x_t = gamma_t * x_1 + (1 - gamma_t) * x_0
        target = x_1 - x_0
        pred = self.mlp(x_t, x_low, t)
        loss = gamma_t_grad.pow(2) * (target - pred).pow(2)

        obs_mask = ~mask
        loss = (loss * obs_mask).sum(1) / (obs_mask.sum(1) + 1e-8)
        loss = loss.mean()

        return loss

    def u_t(self, x_t, x_low, t, gamma_t_grad=None):
        if gamma_t_grad is None:
            e_gamma = self.proj_to_gamma(x_low)
            gamma_t_grad = self.gamma.get_grads(e_gamma, t)

        u = gamma_t_grad * self.mlp(x_t, x_low, t)
        return u

    @torch.inference_mode()
    def sampler(self, x_cat, z_num, num_steps=200):
        B = x_cat.shape[0]
        x_low = self.emb(torch.column_stack((x_cat, z_num))).flatten(1)

        # construct time steps
        t_steps = torch.linspace(0, 1, num_steps + 1, device=self.device, dtype=torch.float32)
        # initialize latents
        means = self.get_group_means(z_num + self.group_offset).squeeze(-1)
        stds = self.get_group_stds(z_num + self.group_offset).squeeze(-1)
        x_next = means + stds * torch.randn_like(means)

        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:], strict=True):
            t_cur = t_cur.repeat((B,))
            t_next = t_next.repeat((B,))
            u_t = self.u_t(x_next, x_low, t_cur)
            h = t_next - t_cur
            x_next = x_next + h.unsqueeze(1) * u_t

        return x_next.cpu()

    @torch.inference_mode()
    def sample_data(self, x_cat, z_num, num_steps=200, batch_size=4096, seed=42, verbose=True):
        set_seeds(seed, cuda_deterministic=True)
        n_batches, remainder = divmod(x_cat.shape[0], batch_size)
        sample_sizes = n_batches * [batch_size] + [remainder] if remainder != 0 else n_batches * [batch_size]
        x_cat_parts = torch.split(x_cat, sample_sizes, dim=0)
        z_num_parts = torch.split(z_num, sample_sizes, dim=0)

        x = []
        for i in tqdm(range(len(sample_sizes)), disable=(not verbose)):
            x_cat_part = x_cat_parts[i].to(self.device)
            z_num_part = z_num_parts[i].to(self.device)
            x_gen = self.sampler(x_cat_part, z_num_part, num_steps=num_steps)
            x.append(x_gen)
        x = torch.cat(x).cpu()

        return x

    @torch.inference_mode()
    def sample_path(self, z_num, num_steps=200, batch_size=4096, seed=42, verbose=True):
        set_seeds(seed, cuda_deterministic=True)
        n_batches, remainder = divmod(z_num.shape[0], batch_size)
        sample_sizes = n_batches * [batch_size] + [remainder] if remainder != 0 else n_batches * [batch_size]
        z_num_parts = torch.split(z_num, sample_sizes, dim=0)

        # construct time steps
        t_steps = torch.linspace(0, 1, num_steps + 1, device=self.device, dtype=torch.float32)

        paths = []
        for i in tqdm(range(len(sample_sizes)), disable=(not verbose)):
            z_num_part = z_num_parts[i].to(self.device)
            B = z_num_part.shape[0]
            x_low = self.emb(z_num_part).flatten(1)

            # initialize latents
            means = self.get_group_means(z_num_part + self.group_offset).squeeze(-1)
            stds = self.get_group_stds(z_num_part + self.group_offset).squeeze(-1)
            x_next = means + stds * torch.randn_like(means)

            path = [x_next.cpu()]
            for t_cur, t_next in zip(t_steps[:-1], t_steps[1:], strict=True):
                t_cur = t_cur.repeat((B,))
                t_next = t_next.repeat((B,))
                u_t = self.u_t(x_next, x_low, t_cur)
                h = t_next - t_cur
                x_next = x_next + h.unsqueeze(1) * u_t
                path.append(x_next.cpu())

            path = torch.stack(path)  # (num_steps+1, batch_size, num_features)
            paths.append(path)
        paths = torch.cat(paths, dim=1)  # (num_steps+1, B, num_features)

        return paths, t_steps.cpu()

    @torch.inference_mode()
    def plot_gamma(self, x_cat, z_num, num_points=100):
        x_low = self.emb(torch.column_stack((x_cat, z_num))).flatten(1)
        e_gamma = self.proj_to_gamma(x_low)
        t_grid = torch.linspace(0, 1, num_points, device=self.device).to(torch.float32)

        gamma_t = []
        for t in t_grid:
            t = t.repeat((x_cat.shape[0],))
            gamma_t.append(self.gamma(e_gamma, t))
        gamma_t = torch.stack(gamma_t, dim=0)  # (num_points, batch_size, num_features)

        return t_grid.cpu().numpy(), gamma_t.cpu().numpy()
