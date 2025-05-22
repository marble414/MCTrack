import torch
import torch.nn as nn
import torch.nn.functional as F

class ImplicitObject(nn.Module):
    """Implicit neural field for object representation."""

    def __init__(self, repr_dim: int = 128, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = 3  # xyz input
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, repr_dim))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("latent", torch.zeros(repr_dim))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(xyz) + self.latent

    def update_representation(self, xyz: torch.Tensor, momentum: float = 0.1):
        """Update latent representation with new observations."""
        new_feat = self.mlp(xyz).mean(dim=0)
        self.latent.mul_(1 - momentum).add_(momentum * new_feat)
