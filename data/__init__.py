from .utils import get_all_subclasses
from .data_base import ClassificationDataBase
from .fmnist import FMNIST
from .cifar10 import CIFAR10
from .circles import Circles


def DataFactory(dataset_name, **kwargs):
    available_nlp_datasets = {}
    for _class_ in get_all_subclasses(ClassificationDataBase):
        available_nlp_datasets[_class_.__name__] = _class_
    if dataset_name in available_nlp_datasets:
        return available_nlp_datasets[dataset_name](**kwargs)
    else:
        raise NotImplementedError(f"undefined dataset <{dataset_name}>")