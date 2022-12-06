from scipy.linalg import fractional_matrix_power
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class TwoLayerGCNEdge(nn.Module):
    """
    The two layer graph convolutional network as in Kipf and Welling's original paper, linked
    here: https://arxiv.org/abs/1609.02907. It has been modified to give link predictions.
    Two graph convolutional layers project the node features into a latent space, after which
    the features are decoded into a matrix of the same dimension of the adjacency matrix via
    an inner product. We apply a sigmoid so that each entry is between 0 and 1. If the (i,j)-th
    entry is closer to 1 than 0, then we say that an edge exists between nodes i and j.

    A: the adjacency matrix of the graph.
    num_features: number of features per node.
    hidden_dim1: the dimension of the first latent space.
    hidden_dim2: the dimension of the second latent space.
    p: percent of neurons to drop out.
    seed: random seed for initializing weight matrices.
    """
    def __init__(self, A, num_features, hidden_dim1, hidden_dim2, p=0, seed=42):
        super(TwoLayerGCNEdge, self).__init__()
        self.conv1 = GraphConvLayer(A, num_features, hidden_dim1, seed=seed)
        self.conv2 = GraphConvLayer(A, hidden_dim1, hidden_dim2, seed=seed)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p)
        return

    def forward(self, X):
        input = X.detach().clone()
        input = self.dropout(input)
        z = self.conv1(input)
        z = self.dropout(z)
        Z = self.conv2(z)
        probs = self.sigmoid(Z @ Z.t())
        return probs

class GraphConvLayer(nn.Module):
    def __init__(self, A, num_features, hidden_dim, seed=42):
        """
        A: the adjacency matrix of the graph.
        num_features: number of features per node.
        hidden_dim: the dimensionality of the latent space to which \mathbb{R}^{num_features}
        will be projected.
        seed: for initializing the weight matrix.
        """
        super(GraphConvLayer, self).__init__()
        self.num_nodes = A.shape[0]
        self.num_features = num_features
        
        self.A_hat = torch.add(torch.eye(self.num_nodes), A)
        self.D = torch.diag(self.A_hat.sum(axis=1))
        self.D_inv_root = torch.Tensor(fractional_matrix_power(self.D, -0.5))
        self.A_tilde = self.D_inv_root @ self.A_hat @ self.D_inv_root
        
        torch.manual_seed(seed)
        self.W = Parameter(torch.normal(
                mean=torch.Tensor(1), 
                std=torch.ones(num_features, hidden_dim) / (hidden_dim)
        ))
        return
    
    def forward(self, H):
        """
        dim(A_tilde) == (num_nodes, num_nodes)
        dim(H) == (num_nodes, num_features)
        dim(W) == (num_features, hidden_dim)

        dim(out) == (num_nodes, hidden_dim)
        """
        out = F.relu(self.A_tilde @ H @ self.W)
        return out