from .data_base import ClassificationDataBase
from .utils import TorchDataset
import torch
import os


class Circles(ClassificationDataBase):
    def __init__(self, path, **kwargs):
        super().__init__()
        train_dataset = TorchDataset(
                torch.load(os.path.join(path, 'circles_train_data_X.pt')),
                torch.load(os.path.join(path, 'circles_train_data_y.pt')))
        dev_dataset = TorchDataset(
                torch.load(os.path.join(path, 'circles_dev_data_X.pt')),
                torch.load(os.path.join(path, 'circles_dev_data_y.pt')))
        self.datasets = {'train': train_dataset,
                        'dev': dev_dataset}
        self.num_classes = 3
        self.in_shape = (2,)