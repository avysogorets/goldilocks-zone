from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, data_X, data_y):
        self.data = data_X.float()
        self.targets = data_y.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.targets[idx]
        return X, y