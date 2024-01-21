from torch.utils.data import Dataset
from typing import Dict


class ClassificationDataBase:

    def __init__(self, **kwargs):
        self.num_classes: int
        self.datasets: Dict[str, Dataset] = {'train': None, 'dev': None}