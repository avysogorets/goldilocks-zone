from .model_base import ClassificationModelBase
import torch


class LeNet5(ClassificationModelBase):

    def __init__(self,
                in_shape,
                num_classes,
                **kwargs):
        super().__init__(
                in_shape=in_shape,
                **kwargs)
        num_at_flat = int(16*(((in_shape[1]-4)/2-4)/2)**2)
        self.module_list = torch.nn.ModuleList([
            torch.nn.Conv2d(in_shape[0], 6, padding=0, kernel_size=(5,5), stride=1, bias=False),
            self.activation(),
            torch.nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            torch.nn.Conv2d(6, 16, padding=0, kernel_size=(5,5), stride=1, bias=False),
            self.activation(),
            torch.nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            torch.nn.Flatten(),
            torch.nn.Linear(num_at_flat, 120, bias=False),
            self.activation(),
            torch.nn.Linear(120, 84, bias=False),
            self.activation(),
            torch.nn.Linear(84, num_classes, bias=False)])
        self.initialize()