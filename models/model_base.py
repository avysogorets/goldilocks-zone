from torch.utils.data import DataLoader
import torch


class ClassificationModelBase(torch.nn.Module):

    def __init__(self,
                in_shape,
                device=torch.device('cpu'),
                dtype=torch.float32,
                temperature=1.,
                activation='ReLU'):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.dtype = dtype
        self.temperature = temperature
        self.module_list = None
        if activation == 'ReLU':
            self.activation = torch.nn.ReLU
        elif activation == 'Tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'Sigmoid':
            self.activation = torch.nn.Sigmoid
        elif activation == 'LeakyReLU':
            self.activation = torch.nn.LeakyReLU
        elif activation == 'Identity':
            self.activation = torch.nn.Identity
        else:
            raise NotImplementedError(f"activation function <{activation}> is unknown")
        
    def set_temperature(self, temp):
        self.temperature = temp

    def initialize(self):
        self.num_layers = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                self.num_layers += 1
                torch.nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.to(self.device)
        self.to(self.dtype)  

    def forward(self, x):
        x = x.to(self.dtype).to(self.device)
        for module in self.module_list:
            for m in module.modules():
                x = m(x)
        x = x/self.temperature
        return x      

    def predict(self, dataset, batch_size=32):

        # Calling even on a small dataset outside of torch.no_grad()
        # environment will likely result in cuda OOM (graph resources
        # are freed with backward() call, which happens after every
        # batch during training but not in this predict function). If 
        # gradients needed, write a custom validation loop that computes
        # gradients and releases graph resources after every batch. 

        batch_size = min(len(dataset), batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        outputs = []
        for batch in dataloader:
            X,_ = batch
            X = X.to(self.device)
            output = self(X)
            outputs.append(output)
        outputs = torch.vstack(outputs)
        return outputs