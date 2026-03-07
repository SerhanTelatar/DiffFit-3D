"""
Human Segmentation Module.

Performs human parsing/segmentation to identify body parts,
used for generating agnostic masks and conditioning signals.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


# ATR segmentation label map (18 classes)
ATR_LABELS = {
    0: "background", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper_clothes",
    5: "skirt", 6: "pants", 7: "dress", 8: "belt", 9: "left_shoe",
    10: "right_shoe", 11: "face", 12: "left_leg", 13: "right_leg",
    14: "left_arm", 15: "right_arm", 16: "bag", 17: "scarf",
}

# Body part groups for masking
UPPER_CLOTHING_LABELS = {4, 7}  # upper_clothes, dress
LOWER_CLOTHING_LABELS = {5, 6}  # skirt, pants
ALL_CLOTHING_LABELS = {4, 5, 6, 7, 8, 17}  # All garment regions
PRESERVE_LABELS = {2, 3, 11, 14, 15}  # hair, sunglasses, face, arms


class HumanSegmentation:
    """
    Human parsing segmentation for body part identification.

    Args:
        model_type: 'atr', 'lip', or 'sam'.
        device: Computation device.
        num_classes: Number of segmentation classes.
    """

    def __init__(self, model_type: str = "atr", device: str = "cuda", num_classes: int = 18):
        self.model_type = model_type
        self.device = device
        self.num_classes = num_classes
        self.model = None

    def load_model(self):
        """Load segmentation model."""
        if self.model_type == "atr":
            self._load_atr()
        elif self.model_type == "sam":
            self._load_sam()

    def _load_atr(self):
        """Load ATR human parsing model."""
        try:
            # Use a simple segmentation network as fallback
            self.model = SimpleSegNet(num_classes=self.num_classes)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load ATR model: {e}")
            self.model = None

    def _load_sam(self):
        """Load SAM (Segment Anything Model)."""
        try:
            from segment_anything import SamPredictor, sam_model_registry
            sam = sam_model_registry["vit_h"](checkpoint="checkpoints/pretrained/sam_vit_h.pth")
            sam.to(self.device)
            self.model = SamPredictor(sam)
        except ImportError:
            print("Warning: SAM not available.")
            self.model = None

    def segment(self, image: np.ndarray) -> dict:
        """
        Segment a person image into body parts.

        Args:
            image: (H, W, 3) BGR image.

        Returns:
            Dict with 'segmentation' (H, W) label map,
            'visualization' (H, W, 3) colored segmentation.
        """
        if self.model is None:
            self.load_model()

        h, w = image.shape[:2]

        if self.model is not None and isinstance(self.model, SimpleSegNet):
            seg = self._segment_with_net(image)
        else:
            seg = np.zeros((h, w), dtype=np.uint8)

        vis = self.colorize_segmentation(seg)
        return {"segmentation": seg, "visualization": vis}

    def _segment_with_net(self, image: np.ndarray) -> np.ndarray:
        """Run segmentation with neural network."""
        h, w = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(dim=1)[0].cpu().numpy()

        pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return pred

    def get_clothing_mask(self, segmentation: np.ndarray,
                          mask_type: str = "upper") -> np.ndarray:
        """
        Extract clothing region mask from segmentation.

        Args:
            segmentation: (H, W) label map.
            mask_type: 'upper', 'lower', or 'full'.

        Returns:
            (H, W) binary mask of clothing region.
        """
        if mask_type == "upper":
            labels = UPPER_CLOTHING_LABELS
        elif mask_type == "lower":
            labels = LOWER_CLOTHING_LABELS
        elif mask_type == "full":
            labels = ALL_CLOTHING_LABELS
        else:
            labels = UPPER_CLOTHING_LABELS

        mask = np.zeros_like(segmentation, dtype=np.uint8)
        for label in labels:
            mask[segmentation == label] = 255
        return mask

    @staticmethod
    def colorize_segmentation(seg: np.ndarray) -> np.ndarray:
        """Convert label map to RGB visualization."""
        # Color palette for 18 ATR classes
        palette = np.array([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0],
        ], dtype=np.uint8)
        h, w = seg.shape
        vis = palette[seg.flatten()].reshape(h, w, 3)
        return vis


class SimpleSegNet(nn.Module):
    """Simple FCN segmentation network (placeholder for pretrained models)."""

    def __init__(self, num_classes: int = 18):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
