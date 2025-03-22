import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


device = torch.cuda.is_available() and torch.device("cuda") or torch.device("cpu")

# 1. Dataset Preparation
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

# 2. Define Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                layers.append(activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        return self.network(x)


def train_model(model, trainloader, testloader, optimizer, criterion, epochs=20, patience=3):
    best_acc = 0.0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        acc = evaluate_model(model, testloader)
        print(f"Epoch {epoch+1}, Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    return best_acc


def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    acc = 100 * correct / total
    return acc

# 3. Test Different Topologies
def test_topologies(trainloader, testloader):
    topologies = [
        [784, 128, 10],           # 1 hidden layer
        [784, 256, 128, 10],      # 2 hidden layers
        [784, 512, 256, 10],      # 2 hidden layers, wider
        [784, 256, 128, 64, 10],  # 3 hidden layers
        [784, 1024, 512, 10]      # 2 hidden layers, very wide
    ]
    results = {}
    criterion = nn.CrossEntropyLoss()
    for i, topology in enumerate(topologies):
        print(f"\nTesting Topology {i+1}: {topology}")
        model = NeuralNet(topology).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        acc = train_model(model, trainloader, testloader, optimizer, criterion)
        results[str(topology)] = acc
    best_topology = max(results, key=results.get)
    print(f"Best topology: {best_topology} with accuracy {results[best_topology]:.4f}")
    return eval(best_topology)

# 4. Test Different Optimizers
def test_optimizers(trainloader, testloader, best_topology):
    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "RMSprop": optim.RMSprop
    }
    results = {}
    criterion = nn.CrossEntropyLoss()
    for name, opt in optimizers.items():
        print(f"\nTesting Optimizer: {name}")
        model = NeuralNet(best_topology).to(device)
        optimizer = opt(model.parameters(), lr=0.01)
        acc = train_model(model, trainloader, testloader, optimizer, criterion)
        results[name] = acc
    best_optimizer = max(results, key=results.get)
    print(f"Best optimizer: {best_optimizer} with accuracy {results[best_optimizer]:.4f}")
    return optimizers[best_optimizer]

# 5. Test Different Learning Rates
def test_learning_rates(trainloader, testloader, best_topology, best_optimizer_class):
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    results = {}
    criterion = nn.CrossEntropyLoss()
    for lr in learning_rates:
        print(f"\nTesting Learning Rate: {lr}")
        model = NeuralNet(best_topology).to(device)
        optimizer = best_optimizer_class(model.parameters(), lr=lr)
        acc = train_model(model, trainloader, testloader, optimizer, criterion)
        results[lr] = acc
    best_lr = max(results, key=results.get)
    print(f"Best learning rate: {best_lr} with accuracy {results[best_lr]:.4f}")
    return best_lr

# 6. Test Different Activation Functions
def test_activations(trainloader, testloader, best_topology, best_optimizer_class, best_lr):
    activations = {
        "ReLU": nn.ReLU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "LeakyReLU": nn.LeakyReLU(0.01),
        "ELU": nn.ELU()
    }
    results = {}
    criterion = nn.CrossEntropyLoss()
    for name, act in activations.items():
        print(f"\nTesting Activation: {name}")
        model = NeuralNet(best_topology, activation=act).to(device)
        optimizer = best_optimizer_class(model.parameters(), lr=best_lr)
        acc = train_model(model, trainloader, testloader, optimizer, criterion)
        results[name] = acc
    best_activation = max(results, key=results.get)
    print(f"Best activation: {best_activation} with accuracy {results[best_activation]:.4f}")
    return activations[best_activation]


def get_confusion_matrix(model, testloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of Best Model')
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")
    # plt.show()


if __name__ == "__main__":

    trainloader, testloader, trainset = load_data()
    perform_eda(trainset)

    # Test hyperparameters
    best_topology = test_topologies(trainloader, testloader)
    best_optimizer_class = test_optimizers(trainloader, testloader, best_topology)
    best_lr = test_learning_rates(trainloader, testloader, best_topology, best_optimizer_class)
    best_activation = test_activations(trainloader, testloader, best_topology, best_optimizer_class, best_lr)

    final_model = NeuralNet(best_topology, activation=best_activation).to(device)
    optimizer = best_optimizer_class(final_model.parameters(), lr=best_lr)
    criterion = nn.CrossEntropyLoss()
    final_acc = train_model(final_model, trainloader, testloader, optimizer, criterion)
    torch.save(final_model.state_dict(), "final_best_model.pth")
    print(f"Final model accuracy: {final_acc:.4f}")

    # Plot confusion matrix for best model
    get_confusion_matrix(final_model, testloader)
