import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFrameGNN(nn.Module):
    """Graph neural network for multi-frame association."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Propagate features through the graph and predict association scores."""
        x = node_feats
        for layer in self.layers:
            x = torch.relu(layer(torch.matmul(adj, x)))
        scores = torch.sigmoid(self.classifier(x))
        return scores
