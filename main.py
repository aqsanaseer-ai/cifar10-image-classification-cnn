import torch
from src.dataset import get_loaders
from src.model import CIFAR10CNN
from src.train import train
from src.evaluate import evaluate
from utils.metrics import save_accuracy_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

train_loader, val_loader = get_loaders()

model = CIFAR10CNN().to(device)

train_acc, val_acc = train(model, train_loader, val_loader, device, epochs=20)

save_accuracy_plot(train_acc, val_acc)

evaluate(model, val_loader, device, classes)