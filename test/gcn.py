from scipy.linalg import fractional_matrix_power
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class TwoLayerGCN(nn.Module):
    def __init__(self, A, num_classes, num_features, p=0, seed=42):
        """
        The two layer graph convolutional network as in Kipf and Welling's original paper, linked
        here: https://arxiv.org/abs/1609.02907. It projects the design matrix into a latent space
        of dimension (num_nodes, 16), after which it returns probabilities of belonging to each 
        class for each node.

        A: adjacency matrix of the graph.
        num_features: number of features per node.
        num_classes: number of classes that a node can belong to.
        p: parameter for controlling dropout.
        """
        super(TwoLayerGCN, self).__init__()
        self.conv1 = GraphConvLayer(A, num_features, 16, seed=seed)
        self.conv2 = GraphConvLayer(A, 16, num_classes, seed=seed)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p)
        return

    def forward(self, X):
        """
        Applies two convolutional layers and then a softmax.
        """
        input = X.detach().clone()
        z1 = self.dropout(input)
        z2 = self.conv1(z1)
        z3 = self.dropout(z2)
        z4 = self.conv2(z3)
        return self.softmax(z4)

class GraphConvLayer(nn.Module):
    def __init__(self, A, num_features, num_classes, seed=42):
        """
        A standard graph convolutional layer.

        A: adjacency matrix of graph
        num_classes: number of classes.
        num_features: number of features for each node.
        seed: for initializing the weight matrix.
        """
        super(GraphConvLayer, self).__init__()
        self.num_nodes = A.shape[0]
        self.num_classes = num_classes
        self.num_features = num_features
        
        self.A_hat = torch.add(torch.eye(self.num_nodes), A)
        self.D = torch.diag(self.A_hat.sum(axis=1))
        self.D_inv_root = torch.Tensor(fractional_matrix_power(self.D, -0.5))
        self.norm_A = self.D_inv_root @ self.A_hat @ self.D_inv_root
        
        torch.manual_seed(seed)
        self.W = Parameter(torch.normal(
                mean=torch.Tensor(1), 
                std=torch.ones(num_features, num_classes) / (num_features * num_classes)
            ))
        return
    
    def forward(self, H):
        """
        dim(self.norm_A) == (num_nodes, num_nodes)
        dim(H) == (num_nodes, num_features)
        dim(W) == (num_features, num_classes)

        dim(out) == (num_classes, num_nodes)
        """
        out = F.relu(self.norm_A @ H @ self.W)
        return out