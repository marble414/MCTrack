"""Example script demonstrating cross-modal tracking on KITTI.

This script loads RGB images and LiDAR point clouds, aligns their features,
updates an implicit object representation, associates frames using a GNN
and fuses cues with a transformer-based tracker.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import KittiMultimodalDataset
from models import FeatureAlignment, ImplicitObject, MultiFrameGNN, CrossModalTracker


def collate_fn(batch):
    return batch[0]


def main(args):
    dataset = KittiMultimodalDataset(
        dataset_root=args.dataset_root,
        detections_json=args.detections_json,
        split=args.split,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    img_transform = transforms.Compose([transforms.ToTensor()])
    align = FeatureAlignment()
    implicit_obj = ImplicitObject()
    gnn = MultiFrameGNN()
    tracker = CrossModalTracker()

    prev_feats = []
    adj = None

    for data in loader:
        image = img_transform(data["image"])
        lidar = torch.from_numpy(data["lidar"])

        # Very simple feature extraction
        lidar_feat = lidar.mean(dim=0)
        img_feat = image.view(3, -1).mean(dim=1)

        lidar_emb, img_emb = align(lidar_feat.unsqueeze(0), img_feat.unsqueeze(0))
        fused = (lidar_emb + img_emb) / 2

        implicit_obj.update_representation(fused)
        cur_feat = fused
        prev_feats.append(cur_feat)

        if adj is None:
            adj = torch.eye(len(prev_feats))
        else:
            adj = torch.ones(len(prev_feats), len(prev_feats)) / len(prev_feats)

        feats = torch.stack(prev_feats)
        _ = gnn(feats, adj)
        out = tracker(feats)

        print(f"Frame {data['scene']}/{data['frame_id']}: output feature norm {out[-1].norm().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to KITTI dataset root")
    parser.add_argument("--detections_json", type=str, required=True, help="Path to converted detection json")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or test)")
    args = parser.parse_args()
    main(args)
