import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class TwoLayerFCN(nn.Module):
    """
    The usage of this class is to return a neural network that computes the probability of each
    node belonging to any class. It is a standard feedforward neural network.
    """
    def __init__(self, num_classes, num_features, num_nodes, seed=42):
        super(TwoLayerFCN, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_nodes = num_nodes

        self.FCN1 = FullyConnectedLayer(num_features // 4, num_features, num_nodes, seed=seed)
        self.FCN2 = FullyConnectedLayer(num_classes, num_features // 4, num_nodes, seed=seed)
        self.softmax = nn.Softmax(dim=1)
        return

    def forward(self, X):
        """
        Applies two affine transformations, each of which is followed by an activation, and a 
        softmax (for turning outputs into probabilities).
        """
        X = X.detach().clone()
        z1 = self.FCN1.forward(X)
        z2 = self.FCN2.forward(z1)
        return self.softmax(z2)

class FullyConnectedLayer(nn.Module):
    """
    Applies a single affine transformation followed by ReLU.
    """
    def __init__(self, num_classes, num_features, num_nodes, seed=42):
        super(FullyConnectedLayer, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_nodes = num_nodes

        torch.manual_seed(seed)
        self.W = Parameter(
            torch.normal(
                mean=torch.Tensor(1), 
                std=torch.ones(num_classes * num_features) / (num_classes * num_features)
            )
            .reshape(num_classes, num_features)
        )
        self.b = Parameter(
            torch.normal(
                mean=torch.Tensor(1),
                std=torch.ones(num_classes * num_nodes) / (num_classes * num_nodes)
            )
            .reshape(num_classes, num_nodes)
        )
        return

    def forward(self, X):
        """
        Applies a linear transformation and then ReLU.
        """
        raw = F.relu((self.W @ X.transpose(1,0)) + self.b).transpose(1,0)
        return raw