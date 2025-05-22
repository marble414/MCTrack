import torch
import torch.nn as nn

class CrossModalTracker(nn.Module):
    """End-to-end tracker with transformer encoder."""

    def __init__(self, feature_dim: int = 256, num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(feature_dim, feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Process concatenated multi-modal features."""
        x = self.encoder(features)
        return self.head(x)
