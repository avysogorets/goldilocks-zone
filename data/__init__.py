from data.utils import get_all_subclasses
from data.data_base import ClassificationDataBase
from data.fmnist import FMNIST
from data.cifar10 import CIFAR10
from data.circles import Circles


def DataFactory(dataset_name, **kwargs):
    available_nlp_datasets = {}
    for _class_ in get_all_subclasses(ClassificationDataBase):
        available_nlp_datasets[_class_.__name__] = _class_
    if dataset_name in available_nlp_datasets:
        return available_nlp_datasets[dataset_name](**kwargs)
    else:
        raise NotImplementedError(f"undefined dataset <{dataset_name}>")