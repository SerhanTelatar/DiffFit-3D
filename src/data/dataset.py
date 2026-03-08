"""
TryOn Dataset — PyTorch Dataset for person-garment pairs.

Loads person images with all preprocessing outputs (pose, segmentation,
DensePose, agnostic mask) alongside garment images.
"""

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.transforms import TryOnTransforms
from src.utils.image_utils import pil_to_tensor


class TryOnDataset(Dataset):
    """
    Dataset for virtual try-on training.

    Expects preprocessed data in the following structure:
        data/raw/images/{image_id}.jpg
        data/processed/poses/{image_id}.npy
        data/processed/segments/{image_id}.png
        data/processed/densepose/{image_id}.npy
        data/processed/agnostic/{image_id}.jpg

    Args:
        pairs_file: CSV file with columns (person_id, garment_id).
        data_root: Root data directory.
        resolution: Target image resolution.
        transforms: Optional augmentation transforms.
    """

    def __init__(self, pairs_file: str, data_root: str = "data",
                 resolution: int = 512, transforms: Optional[TryOnTransforms] = None,
                 use_3d: bool = True):
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.transforms = transforms
        self.use_3d = use_3d

        # Load pairs
        self.pairs = self._load_pairs(pairs_file)

        # 2D Directories
        self.image_dir = self.data_root / "raw" / "images"
        self.pose_dir = self.data_root / "processed" / "poses"
        self.segment_dir = self.data_root / "processed" / "segments"
        self.densepose_dir = self.data_root / "processed" / "densepose"
        self.agnostic_dir = self.data_root / "processed" / "agnostic"

        # 3D Directories
        self.garments_3d_dir = self.data_root / "garments_3d"
        self.smplx_params_dir = self.data_root / "processed" / "smplx_params"
        self.renders_3d_dir = self.data_root / "processed" / "renders_3d"
        self.normal_maps_dir = self.data_root / "processed" / "normal_maps"
        self.depth_maps_dir = self.data_root / "processed" / "depth_maps"

    def _load_pairs(self, pairs_file: str) -> list[tuple[str, str]]:
        pairs = []
        path = Path(pairs_file)
        if not path.exists():
            return pairs
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    pairs.append((row[0].strip(), row[1].strip()))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        person_id, garment_id = self.pairs[idx]

        # Load person image
        person_img = self._load_image(self.image_dir / f"{person_id}.jpg")

        # Load garment image (2D flat photo as fallback)
        garment_img = self._load_image(self.image_dir / f"{garment_id}.jpg")

        # Load 2D preprocessed data
        pose_map = self._load_pose(person_id)
        segment_map = self._load_segment(person_id)
        densepose_map = self._load_densepose(person_id)
        agnostic_img = self._load_agnostic(person_id)

        # Apply transforms
        if self.transforms:
            person_img, garment_img, pose_map, agnostic_img = self.transforms(
                person_img, garment_img, pose_map, agnostic_img
            )

        # Convert to tensors
        sample = {
            "person_image": pil_to_tensor(person_img, self.resolution),
            "garment_image": pil_to_tensor(garment_img, self.resolution),
            "agnostic_image": pil_to_tensor(agnostic_img, self.resolution),
            "pose_map": self._pose_to_tensor(pose_map),
            "segment_map": self._segment_to_tensor(segment_map),
            "densepose_map": self._densepose_to_tensor(densepose_map),
            "person_id": person_id,
            "garment_id": garment_id,
        }

        # 3D verileri ekle
        if self.use_3d:
            render_name = f"{person_id}_{garment_id}"

            # Render edilmiş 3D giysi görüntüsü
            render_3d = self._load_image(self.renders_3d_dir / f"{render_name}.png")
            sample["garment_render_3d"] = pil_to_tensor(render_3d, self.resolution)

            # Normal haritası
            normal_map = self._load_image(self.normal_maps_dir / f"{render_name}.png")
            sample["normal_map"] = pil_to_tensor(normal_map, self.resolution)

            # Derinlik haritası
            depth_map = self._load_depth(self.depth_maps_dir / f"{render_name}.png")
            sample["depth_map"] = depth_map

            # SMPL-X beden parametreleri
            smplx_params = self._load_smplx_params(person_id)
            sample["smplx_betas"] = smplx_params["betas"]
            sample["smplx_body_pose"] = smplx_params["body_pose"]

        return sample

    def _load_image(self, path: Path) -> Image.Image:
        if path.exists():
            return Image.open(path).convert("RGB")
        return Image.new("RGB", (self.resolution, self.resolution), (128, 128, 128))

    def _load_pose(self, person_id: str) -> np.ndarray:
        path = self.pose_dir / f"{person_id}.npy"
        if path.exists():
            return np.load(path)
        return np.zeros((18, 3), dtype=np.float32)

    def _load_segment(self, person_id: str) -> np.ndarray:
        path = self.segment_dir / f"{person_id}.png"
        if path.exists():
            return np.array(Image.open(path))
        return np.zeros((self.resolution, self.resolution), dtype=np.uint8)

    def _load_densepose(self, person_id: str) -> np.ndarray:
        path = self.densepose_dir / f"{person_id}.npy"
        if path.exists():
            return np.load(path)
        return np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)

    def _load_agnostic(self, person_id: str) -> Image.Image:
        path = self.agnostic_dir / f"{person_id}.jpg"
        if path.exists():
            return Image.open(path).convert("RGB")
        return Image.new("RGB", (self.resolution, self.resolution), (128, 128, 128))

    def _load_depth(self, path: Path) -> torch.Tensor:
        """Derinlik haritasını yükle (grayscale → tensor)."""
        if path.exists():
            depth = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        else:
            depth = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        if tensor.shape[-1] != self.resolution:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(self.resolution, self.resolution),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return tensor

    def _load_smplx_params(self, person_id: str) -> dict[str, torch.Tensor]:
        """SMPL-X beden parametrelerini yükle."""
        path = self.smplx_params_dir / f"{person_id}.npz"
        if path.exists():
            data = np.load(path)
            return {
                "betas": torch.from_numpy(data["betas"]).float(),
                "body_pose": torch.from_numpy(data["body_pose"]).float(),
            }
        return {
            "betas": torch.zeros(10, dtype=torch.float32),
            "body_pose": torch.zeros(63, dtype=torch.float32),
        }

    def _pose_to_tensor(self, pose: np.ndarray) -> torch.Tensor:
        """Convert pose keypoints to rendered pose map tensor."""
        from src.modules.pose_estimator import PoseEstimator
        estimator = PoseEstimator()
        canvas = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        pose_img = estimator.render_pose(canvas, pose)
        tensor = torch.from_numpy(pose_img).float().permute(2, 0, 1) / 255.0
        return tensor

    def _segment_to_tensor(self, segment: np.ndarray) -> torch.Tensor:
        seg = torch.from_numpy(segment).long()
        if seg.shape[0] != self.resolution or seg.shape[1] != self.resolution:
            seg = torch.nn.functional.interpolate(
                seg.float().unsqueeze(0).unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="nearest",
            ).squeeze().long()
        return seg

    def _densepose_to_tensor(self, dp: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(dp).float()
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[-1] != self.resolution:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return tensor
