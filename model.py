import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                layers.append(activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        return self.network(x)