import torch
from torch import nn
import torch.nn.functional as F

class MultiModalAugmentor:
    def __init__(self, lidar_augs=None, image_augs=None):
        self.lidar_augs = lidar_augs or []
        self.image_augs = image_augs or []
        # TODO: implement augmentation functions

    def augment_lidar(self, batch):
        # TODO: apply lidar augmentations
        raise NotImplementedError

    def augment_image(self, batch):
        # TODO: apply image augmentations
        raise NotImplementedError

class ContrastiveSelfSupervisedPretraining:
    """Contrastive self-supervised pretraining."""
    def __init__(self, feature_dim=256, projection_dim=128):
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
        )
        self.augmentor = MultiModalAugmentor(
            lidar_augs=["rotation", "scaling", "jittering"],
            image_augs=["color_jitter", "gaussian_blur", "random_crop"],
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def create_multimodal_views(self, lidar_batch, image_batch):
        lidar_view1 = self.augmentor.augment_lidar(lidar_batch)
        lidar_view2 = self.augmentor.augment_lidar(lidar_batch)
        image_view1 = self.augmentor.augment_image(image_batch)
        image_view2 = self.augmentor.augment_image(image_batch)
        pairs = [
            (lidar_view1, image_view1),
            (lidar_view2, image_view2),
            (lidar_view1, lidar_view2),
            (image_view1, image_view2),
        ]
        return pairs

    def info_nce_loss(self, features1, features2, negatives):
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        pos_sim = torch.sum(features1 * features2, dim=-1) / self.temperature
        neg_sim = torch.matmul(features1, negatives.t()) / self.temperature
        loss = -torch.log(
            torch.exp(pos_sim) /
            (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim), dim=-1))
        )
        return loss.mean()
