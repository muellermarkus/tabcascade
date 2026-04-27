import random

import numpy as np
import torch


def set_seeds(seed, cuda_deterministic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


def low_discrepancy_sampler(num_samples, device):
    """
    Inspired from the Variational Diffusion Paper (Kingma et al., 2022)
    """
    single_u = torch.rand((1,), device=device, requires_grad=False, dtype=torch.float64)
    return (single_u + torch.arange(0.0, 1.0, step=1.0 / num_samples, device=device, requires_grad=False)) % 1
