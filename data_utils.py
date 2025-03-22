import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader, trainset

def perform_eda(trainset):
    class_counts = np.bincount(trainset.targets.numpy())
    print("Class distribution:", class_counts)
    plt.bar(range(10), class_counts)
    plt.title("Class Distribution in FashionMNIST")
    plt.savefig("eda_class_distribution.png")
    plt.close()