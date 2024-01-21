from data.data_base import ClassificationDataBase
import torchvision


class FMNIST(ClassificationDataBase):
    def __init__(self, path, device, to_transform=False, **kwargs):
        super().__init__()
        optional_transform = [torchvision.transforms.RandomRotation(degrees=4)]
        train_transform = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,))]
        if to_transform:
                train_transform = optional_transform+train_transform
        train_transform = torchvision.transforms.Compose(train_transform)
        train_dataset = torchvision.datasets.FashionMNIST(path,
                train=True,
                download=True,
                transform=train_transform)
        dev_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,))])
        dev_dataset = torchvision.datasets.FashionMNIST(path,
                train=False,
                download=True,
                transform=dev_transform)
        self.datasets = {'train': train_dataset.to(device),
                        'dev': dev_dataset.to(device)}
        self.num_classes = 10
        self.in_shape = (1,28,28)