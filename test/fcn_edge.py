import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class FullyConnectedLayerEdge(nn.Module):
    """
    There is little difference between FullyConnectedLayerEdge and FullyConnectedLayer in fcn.py -
    I just wanted to be able to change the components of TwoLayerFCNEdge without changing anything
    in TwoLayerFCN.
    """
    def __init__(self, num_nodes, num_features, hidden_dim):
        super(FullyConnectedLayerEdge, self).__init__()
        self.W = Parameter(torch.normal(
            mean=torch.zeros(size=(num_features, hidden_dim)),
            std=torch.ones(size=(num_features, hidden_dim)) / (hidden_dim) # row normalization
        ))
        self.b = Parameter(torch.normal(
            mean=torch.zeros(size=(num_nodes, hidden_dim)),
            std=torch.ones(size=(num_nodes, hidden_dim)) / (hidden_dim)
        ))
        return
    
    def forward(self, X):
        """
        Applies a linear transformation and adds a bias, after which the ReLU activation is
        applied. As one can see from the following computations, the first dimension of the
        input is preserved, as intended. The effect of the forward method is merely to
        project the input's feature dimension into a latent space.

        dim(X) == (num_nodes, num_features)
        dim(W) == (num_features, hidden_dim)
        dim(X @ W) == (num_nodes, hidden_dim)
        dim(b) == (num_nodes, hidden_dim)
        """
        out = F.relu((X @ self.W) + self.b)
        return out

class TwoLayerFCNEdge(nn.Module):
    """
    Returns probabilities of whether node i is adjacent to node j. See TwoLayerGCNEdge for more
    details. In essence, this is an encoder followed by an inner product followed by a sigmoid.
    """
    def __init__(self, num_nodes, num_features, hidden_dim1, hidden_dim2):
        super(TwoLayerFCNEdge, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = FullyConnectedLayerEdge(num_nodes, num_features, hidden_dim1)
        self.layer2 = FullyConnectedLayerEdge(num_nodes, hidden_dim1, hidden_dim2)
        return 

    def forward(self, X):
        """
        Applies FullyConnectedLayerEdge twice, an inner product for each node against every other
        node, and then a sigmoid function to finally return probabilities.
        """
        input = X.detach().clone()
        input = self.layer1(input)
        input = self.layer2(input)
        return self.sigmoid(input @ input.t())