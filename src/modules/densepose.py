"""
DensePose UV Map Extraction.

Extracts dense body-surface correspondence (IUV maps) from person images
using DensePose for fine-grained body geometry conditioning.
"""

import numpy as np
import cv2
import torch
from typing import Optional


class DensePoseExtractor:
    """
    Extracts DensePose IUV maps for body-surface correspondence.

    IUV maps encode:
    - I: Body part index (24 parts)
    - U, V: Surface coordinates within each part

    Args:
        model_name: DensePose model identifier.
        device: Computation device.
        confidence_threshold: Detection confidence threshold.
    """

    # Body part names for DensePose (24 parts)
    BODY_PARTS = [
        "background", "torso_back", "torso_front", "right_hand", "left_hand",
        "left_foot", "right_foot", "upper_leg_right_back", "upper_leg_left_back",
        "upper_leg_right_front", "upper_leg_left_front", "lower_leg_right_back",
        "lower_leg_left_back", "lower_leg_right_front", "lower_leg_left_front",
        "upper_arm_left_back", "upper_arm_right_back", "upper_arm_left_front",
        "upper_arm_right_front", "lower_arm_left_back", "lower_arm_right_back",
        "lower_arm_left_front", "lower_arm_right_front", "head_back", "head_front",
    ]

    # Color palette for visualization (24 parts + background)
    PART_COLORS = np.array([
        [0, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
        [255, 0, 170], [255, 0, 85], [128, 0, 0], [0, 128, 0],
        [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
        [128, 128, 128],
    ], dtype=np.uint8)

    def __init__(self, model_name: str = "densepose_rcnn_R_50_FPN_s1x",
                 device: str = "cuda", confidence_threshold: float = 0.5):
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.predictor = None

    def load_model(self):
        """Load DensePose model using detectron2."""
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from densepose import add_densepose_config
            from densepose.vis.extractor import DensePoseResultExtractor

            cfg = get_cfg()
            add_densepose_config(cfg)
            cfg.MODEL.DEVICE = self.device
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            self.predictor = DefaultPredictor(cfg)
            self.extractor = DensePoseResultExtractor()
        except ImportError:
            print("Warning: detectron2/densepose not available. Using placeholder.")
            self.predictor = None

    def extract(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Extract DensePose IUV map from a person image.

        Args:
            image: (H, W, 3) BGR image.

        Returns:
            Dict with:
                - 'iuv': (H, W, 3) IUV map (I=part index, U,V=surface coords).
                - 'part_mask': (H, W) body part segmentation.
                - 'visualization': (H, W, 3) colored visualization.
        """
        if self.predictor is None:
            self.load_model()

        h, w = image.shape[:2]

        if self.predictor is not None:
            return self._extract_densepose(image)
        else:
            return self._generate_placeholder(h, w)

    def _extract_densepose(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Extract using detectron2 DensePose."""
        h, w = image.shape[:2]
        outputs = self.predictor(image)
        instances = outputs["instances"]

        iuv = np.zeros((h, w, 3), dtype=np.float32)

        if len(instances) > 0:
            results = self.extractor(instances)
            if results is not None:
                for result in results:
                    bbox = result.bbox.cpu().numpy().astype(int)
                    iuv_data = result.labels.cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    rh, rw = y2 - y1, x2 - x1
                    if rh > 0 and rw > 0:
                        resized = cv2.resize(iuv_data.transpose(1, 2, 0), (rw, rh))
                        iuv[y1:y2, x1:x2] = resized

        part_mask = iuv[:, :, 0].astype(np.uint8)
        vis = self.visualize_iuv(iuv)

        return {"iuv": iuv, "part_mask": part_mask, "visualization": vis}

    def _generate_placeholder(self, h: int, w: int) -> dict[str, np.ndarray]:
        """Generate placeholder IUV map when model is unavailable."""
        iuv = np.zeros((h, w, 3), dtype=np.float32)
        part_mask = np.zeros((h, w), dtype=np.uint8)
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        return {"iuv": iuv, "part_mask": part_mask, "visualization": vis}

    def visualize_iuv(self, iuv: np.ndarray) -> np.ndarray:
        """Create colored visualization of IUV map."""
        h, w = iuv.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        part_indices = iuv[:, :, 0].astype(int)
        for i in range(1, 25):
            mask = part_indices == i
            if mask.any():
                color = self.PART_COLORS[i]
                u_vals = (iuv[:, :, 1] * 255).astype(np.uint8)
                v_vals = (iuv[:, :, 2] * 255).astype(np.uint8)
                vis[mask] = (color * 0.5 + np.stack([u_vals, v_vals, np.zeros_like(u_vals)], axis=-1)[mask] * 0.5).astype(np.uint8)
        return vis

    def iuv_to_tensor(self, iuv: np.ndarray) -> torch.Tensor:
        """Convert IUV numpy array to normalized tensor for model input."""
        tensor = torch.from_numpy(iuv).float().permute(2, 0, 1)
        tensor[0] = tensor[0] / 24.0  # Normalize part index
        return tensor  # U, V already in [0, 1]

    def get_torso_mask(self, part_mask: np.ndarray) -> np.ndarray:
        """Extract torso region mask (for upper body try-on)."""
        mask = np.zeros_like(part_mask)
        mask[(part_mask == 1) | (part_mask == 2)] = 255
        return mask
