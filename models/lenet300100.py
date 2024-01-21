from models.model_base import ClassificationModelBase
import torch


class LeNet300100(ClassificationModelBase):

    def __init__(self,
                device,
                in_shape,
                num_classes,
                activation='ReLU',
                temperature=1.,
                dtype=torch.float32,
                **kwargs):
        super().__init__(
                device=device,
                dtype=dtype,
                in_shape=in_shape,
                temperature=temperature,
                activation=activation)
        in_neurons = 1
        for i in in_shape:
                in_neurons*=i
        self.module_list = torch.nn.ModuleList([
                torch.nn.Flatten(),
                torch.nn.Linear(in_neurons, 300, bias=False),
                self.activation(),
                torch.nn.Linear(300, 100, bias=False),
                self.activation(),
                torch.nn.Linear(100, num_classes, bias=False)])
        self.initialize()
        
    def forward(self, x):
        x = x.to(self.dtype).to(self.device)
        for module in self.module_list:
            for m in module.modules():
                x = m(x)
        x = x/self.temperature
        return x