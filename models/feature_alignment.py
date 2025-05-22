import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAlignment(nn.Module):
    """Align LiDAR and image features via contrastive learning."""

    def __init__(self, feature_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.lidar_proj = nn.Linear(feature_dim, proj_dim)
        self.image_proj = nn.Linear(feature_dim, proj_dim)

    def forward(self, lidar_feat: torch.Tensor, image_feat: torch.Tensor):
        """Project features to a shared space."""
        lidar_emb = F.normalize(self.lidar_proj(lidar_feat), dim=-1)
        image_emb = F.normalize(self.image_proj(image_feat), dim=-1)
        return lidar_emb, image_emb

    def contrastive_loss(self, lidar_emb: torch.Tensor, image_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """Compute a symmetric contrastive loss."""
        logits = torch.matmul(lidar_emb, image_emb.t()) / temperature
        labels = torch.arange(lidar_emb.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i + loss_j)
