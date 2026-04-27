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


def cycle(dl):
    while True:
        for data in dl:
            yield data


def low_discrepancy_sampler(num_samples, device):
    """
    Inspired from the Variational Diffusion Paper (Kingma et al., 2022)
    """
    single_u = torch.rand((1,), device=device, requires_grad=False, dtype=torch.float64)
    return (single_u + torch.arange(0.0, 1.0, step=1.0 / num_samples, device=device, requires_grad=False)) % 1


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Adapted from: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *data, batch_size=256, shuffle=False, drop_last=False):
        assert all(d.shape[0] == data[0].shape[0] for d in data), (
            "All tensors must have the same size in the first dimension"
        )
        self.dataset_len = data[0].shape[0]
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        if drop_last:
            self.dataset_len = (self.dataset_len // self.batch_size) * self.batch_size

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration

        if self.indices is not None:
            idx = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(torch.index_select(d, 0, idx) for d in self.data)
        else:
            batch = tuple(d[self.i : self.i + self.batch_size] for d in self.data)

        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
