from data.data_base import ClassificationDataBase
from data.utils import TorchDataset
import torch
import os


class Circles(ClassificationDataBase):
    def __init__(self, path, device, **kwargs):
        super().__init__()
        train_dataset = TorchDataset(
                torch.load(os.path.join(path, 'circles_train_data_X.pt')),
                torch.load(os.path.join(path, 'circles_train_data_y.pt')))
        train_dataset.data = train_dataset.data.to(device)
        train_dataset.targets = train_dataset.targets.to(device)
        dev_dataset = TorchDataset(
                torch.load(os.path.join(path, 'circles_dev_data_X.pt')),
                torch.load(os.path.join(path, 'circles_dev_data_y.pt')))
        dev_dataset.data = dev_dataset.data.to(device)
        dev_dataset.targets = dev_dataset.targets.to(device)
        self.datasets = {'train': train_dataset.to(device),
                        'dev': dev_dataset.to(device)}
        self.num_classes = 3
        self.in_shape = (2,)