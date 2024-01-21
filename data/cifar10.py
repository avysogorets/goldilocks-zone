from data.data_base import ClassificationDataBase
import torchvision


class CIFAR10(ClassificationDataBase):
    def __init__(self, path, device, to_transform=True, **kwargs):
        super().__init__()
        optional_transform = [
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomAffine(
                        degrees=0,  
                        translate=(0.15,0.15))]
        train_transform = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                        mean=(0.4914,0.4822,0.4465),
                        std=(0.2470,0.2435,0.2616))]
        if to_transform:
                train_transform = optional_transform+train_transform
        train_transform = torchvision.transforms.Compose(train_transform)
        train_dataset = torchvision.datasets.CIFAR10(path,
                train=True,
                download=True,
                transform=train_transform)
        train_dataset.data = train_dataset.data.to(device)
        train_dataset.targets = train_dataset.targets.to(device)
        dev_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                        mean=(0.4914,0.4822,0.4465),
                        std=(0.2470,0.2435,0.2616))])
        dev_dataset = torchvision.datasets.CIFAR10(path,
                train=False,
                download=True,
                transform=dev_transform)
        dev_dataset.data = dev_dataset.data.to(device)
        dev_dataset.targets = dev_dataset.targets.to(device)
        self.datasets = {'train': train_dataset,
                        'dev': dev_dataset}
        self.num_classes = 10
        self.in_shape = (3,32,32)