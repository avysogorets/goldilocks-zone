from models.utils import get_all_subclasses
from models.model_base import ClassificationModelBase
from models.lenet300100 import LeNet300100
from models.lenet5 import LeNet5
from models.demo import Demo
from models.utils import *


def ModelFactory(model_name, **kwargs):
    available_models = {}
    for _class_ in get_all_subclasses(ClassificationModelBase):
        available_models[_class_.__name__] = _class_
    if model_name in available_models:
        return available_models[model_name](**kwargs)
    else:
        raise NotImplementedError(f"undefined model {model_name}")