from .model_base import ClassificationModelBase
import torch


class Demo(ClassificationModelBase):

    def __init__(self,
                in_shape,
                num_classes,
                **kwargs):
        super().__init__(
                in_shape=in_shape,
                **kwargs)
        in_neurons = 1
        for i in in_shape:
                in_neurons*=i
        self.module_list = torch.nn.ModuleList([
                torch.nn.Flatten(),
                torch.nn.Linear(in_neurons, 32, bias=False),
                self.activation(),
                torch.nn.Linear(32, 32, bias=False),
                self.activation(),
                torch.nn.Linear(32, num_classes, bias=False)])
        self.initialize()