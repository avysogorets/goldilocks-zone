from torch.utils.data import Dataset
from typing import Dict
import torch


class ClassificationDataBase:

    def __init__(self,
                path: str,
                device: torch.device,
                **kwargs):
        self.num_classes: int
        self.datasets: Dict[str, Dataset] = {'train': None, 'dev': None}