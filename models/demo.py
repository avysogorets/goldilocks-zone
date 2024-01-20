from models.model_base import ClassificationModelBase
import torch


class Demo(ClassificationModelBase):

    def __init__(self,
                device,
                in_shape,
                num_classes,
                activation='ReLU',
                temperature=1.,
                dtype=torch.float32,
                **kwargs):
        super().__init__(device, activation)
        self.in_shape = in_shape
        self.dtype = dtype
        self.temperature = temperature
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
        
    def forward(self, x):
        x = x.to(self.dtype)
        for module in self.module_list:
            for m in module.modules():
                x = m(x)
        x = x/self.temperature
        return x