import torch.nn as nn
import torch.optim as optim
from model import NeuralNet
from train_utils import train_model, device

def test_topologies(trainloader, testloader):
    topologies = [
        [784, 128, 10],
        [784, 256, 128, 10],
        [784, 512, 256, 10],
        [784, 256, 128, 64, 10],
        [784, 1024, 512, 10]
    ]
    results = {}
    criterion = nn.CrossEntropyLoss()
    for i, topology in enumerate(topologies):
        print(f"\nTesting Topology {i + 1}: {topology}")
        model = NeuralNet(topology).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        acc = train_model(model, trainloader, testloader, optimizer, criterion)
        results[str(topology)] = acc
    best_topology = max(results, key=results.get)
    print(f"Best topology: {best_topology} with accuracy {results[best_topology]:.4f}")
    return eval(best_topology)

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

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values())
    plt.xlabel("Activation Function")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison by Activation Function")
    plt.savefig("activation_comparison.png")
    plt.close()

    best_activation = max(results, key=results.get)
    print(f"Best activation: {best_activation} with accuracy {results[best_activation]:.4f}")
    return activations[best_activation]