import torch

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Adapted from: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *data, batch_size=256, shuffle=False, drop_last=False):
        assert all(d.shape[0] == data[0].shape[0] for d in data), "All tensors must have the same size in the first dimension"
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


