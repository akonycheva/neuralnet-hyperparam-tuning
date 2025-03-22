import torch
import torch.nn as nn
import random
import numpy as np
from data_utils import load_data, perform_eda
from experiments import test_topologies, test_optimizers, test_learning_rates, test_activations
from model import NeuralNet
from train_utils import train_model, device
from visualize import get_confusion_matrix, plot_sample_predictions

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if __name__ == "__main__":
    # Load data
    trainloader, testloader, trainset = load_data()
    perform_eda(trainset)

    # Run experiments
    best_topology = test_topologies(trainloader, testloader)
    best_optimizer_class = test_optimizers(trainloader, testloader, best_topology)
    best_lr = test_learning_rates(trainloader, testloader, best_topology, best_optimizer_class)
    best_activation = test_activations(trainloader, testloader, best_topology, best_optimizer_class, best_lr)

    # Train final model
    final_model = NeuralNet(best_topology, activation=best_activation).to(device)
    optimizer = best_optimizer_class(final_model.parameters(), lr=best_lr)
    criterion = nn.CrossEntropyLoss()
    final_acc = train_model(final_model, trainloader, testloader, optimizer, criterion)
    torch.save(final_model.state_dict(), "final_best_model.pth")
    print(f"Final model accuracy: {final_acc:.4f}")

    # Visualize results
    get_confusion_matrix(final_model, testloader)
    plot_sample_predictions(final_model, testloader)