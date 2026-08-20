import torch


class FastTensorDataLoader:
    """A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).

    Adapted from: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *data, batch_size=32, shuffle=False, drop_last=False):
        self.dataset_len = next(t.shape[0] for t in data if t is not None)
        assert all(t.shape[0] == self.dataset_len for t in data if t is not None)
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        self.n_batches = n_batches if drop_last else n_batches + (remainder > 0)
        self.iter_len = self.n_batches * self.batch_size if drop_last else self.dataset_len

    def __iter__(self):
        self.indices = torch.randperm(self.dataset_len) if self.shuffle else None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.iter_len:
            raise StopIteration

        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) if t is not None else None for t in self.data)
        else:
            batch = tuple(t[self.i : self.i + self.batch_size] if t is not None else None for t in self.data)

        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
