from models.model_base import ClassificationModelBase
import torch


class LeNet5(ClassificationModelBase):

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

    def forward(self,x):
        x = x.to(self.dtype)
        for module in self.module_list:
            for m in module.modules():
                x = m(x)
        x = x/self.temperature
        return x