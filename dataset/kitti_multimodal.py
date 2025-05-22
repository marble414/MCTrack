import json
import os
from typing import List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class KittiMultimodalDataset(Dataset):
    """Dataset that loads KITTI RGB images and LiDAR point clouds along with
    detection annotations in BaseVersion format."""

    def __init__(self, dataset_root: str, detections_json: str, split: str = "val"):
        self.dataset_root = dataset_root
        self.split = "training" if split == "val" else "testing"
        with open(detections_json, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.frames: List[Dict] = []
        for scene_id, frames in self.data.items():
            for f_info in frames:
                self.frames.append({"scene": scene_id, **f_info})

    def __len__(self) -> int:
        return len(self.frames)

    def _load_lidar(self, scene: str, frame_id: int) -> np.ndarray:
        lidar_path = os.path.join(
            self.dataset_root,
            self.split,
            "velodyne",
            scene,
            f"{frame_id:06d}.bin",
        )
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        return points

    def _load_image(self, scene: str, frame_id: int) -> Image.Image:
        img_path = os.path.join(
            self.dataset_root,
            self.split,
            "image_02",
            scene,
            f"{frame_id:06d}.png",
        )
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict:
        info = self.frames[idx]
        scene = info["scene"]
        frame_id = info["frame_id"]
        lidar = self._load_lidar(scene, frame_id)
        image = self._load_image(scene, frame_id)
        return {
            "frame_id": frame_id,
            "scene": scene,
            "lidar": lidar,
            "image": image,
            "bboxes": info["bboxes"],
            "transform_matrix": info["transform_matrix"],
        }
