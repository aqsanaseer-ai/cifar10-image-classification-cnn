import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomCrop, RandomHorizontalFlip, RandomRotation
from torch.utils.data import DataLoader

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)

train_transform = Compose([
    Resize(64),
    RandomCrop(64, padding=4),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ToTensor(),
    Normalize(cifar_mean, cifar_std)
])

val_transform = Compose([
    Resize(64),
    ToTensor(),
    Normalize(cifar_mean, cifar_std)
])

def get_loaders(batch_size=64):
    train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )

    val = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader