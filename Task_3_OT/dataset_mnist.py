import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_subset(digit1=0, digit2=4, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    idx = (train_data.targets == digit1) | (train_data.targets == digit2)
    subset = Subset(train_data, torch.where(idx)[0])
    
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    return loader