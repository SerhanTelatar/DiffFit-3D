"""
Agnostic Mask Generator.

Generates clothing-agnostic person representations by masking the garment
region while preserving identity cues (face, hair, hands, lower body).
"""

import numpy as np
import cv2
from typing import Optional


class AgnosticMaskGenerator:
    """
    Creates clothing-agnostic person images that remove the current garment
    while preserving body shape, pose, and identity features.

    Args:
        mask_type: 'upper', 'lower', or 'full' body masking.
        dilate_kernel: Dilation kernel size for mask smoothing.
        preserve_face: Whether to preserve the face region.
        preserve_hands: Whether to preserve the hand regions.
    """

    UPPER_LABELS = {4, 7}       # upper_clothes, dress
    LOWER_LABELS = {5, 6}       # skirt, pants
    FULL_LABELS = {4, 5, 6, 7, 8, 17}
    FACE_LABELS = {11}
    HAIR_LABELS = {2}
    HAND_LABELS = {14, 15}      # left_arm, right_arm (includes hands)

    def __init__(self, mask_type: str = "upper", dilate_kernel: int = 15,
                 preserve_face: bool = True, preserve_hands: bool = True):
        self.mask_type = mask_type
        self.dilate_kernel = dilate_kernel
        self.preserve_face = preserve_face
        self.preserve_hands = preserve_hands

    def generate(self, image: np.ndarray, segmentation: np.ndarray,
                 keypoints: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        """
        Generate a clothing-agnostic representation.

        Args:
            image: (H, W, 3) person image (BGR).
            segmentation: (H, W) segmentation label map.
            keypoints: (N, 3) optional pose keypoints for arm masking.

        Returns:
            Dict with:
                - 'agnostic_image': Person image with garment masked.
                - 'mask': Binary mask of masked region.
                - 'preserved_mask': Binary mask of preserved regions.
        """
        h, w = image.shape[:2]

        # Get clothing mask based on type
        if self.mask_type == "upper":
            target_labels = self.UPPER_LABELS
        elif self.mask_type == "lower":
            target_labels = self.LOWER_LABELS
        else:
            target_labels = self.FULL_LABELS

        # Create clothing mask
        clothing_mask = np.zeros((h, w), dtype=np.uint8)
        for label in target_labels:
            clothing_mask[segmentation == label] = 255

        # Extend mask to cover arms (for upper body try-on)
        if self.mask_type in ("upper", "full") and keypoints is not None:
            arm_mask = self._create_arm_mask(keypoints, h, w)
            clothing_mask = cv2.bitwise_or(clothing_mask, arm_mask)

        # Dilate mask for smooth edges
        if self.dilate_kernel > 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8)
            clothing_mask = cv2.dilate(clothing_mask, kernel, iterations=1)

        # Create preserved regions mask
        preserved = np.zeros((h, w), dtype=np.uint8)
        if self.preserve_face:
            for label in self.FACE_LABELS | self.HAIR_LABELS:
                preserved[segmentation == label] = 255
        if self.preserve_hands:
            # Only preserve hand tips, not full arms
            if keypoints is not None:
                hand_mask = self._create_hand_mask(keypoints, h, w)
                preserved = cv2.bitwise_or(preserved, hand_mask)

        # Remove preserved regions from clothing mask
        clothing_mask[preserved > 0] = 0

        # Create agnostic image
        agnostic = image.copy()
        # Fill masked region with gray (noise-neutral)
        fill_color = np.array([128, 128, 128], dtype=np.uint8)
        agnostic[clothing_mask > 0] = fill_color

        return {
            "agnostic_image": agnostic,
            "mask": clothing_mask,
            "preserved_mask": preserved,
        }

    def _create_arm_mask(self, keypoints: np.ndarray, h: int, w: int) -> np.ndarray:
        """Create mask covering arm regions based on skeleton."""
        mask = np.zeros((h, w), dtype=np.uint8)
        arm_connections = [
            (2, 3), (3, 4),  # Right arm
            (5, 6), (6, 7),  # Left arm
        ]
        thickness = max(w // 15, 20)

        for i, j in arm_connections:
            if (keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3):
                pt1 = tuple(keypoints[i, :2].astype(int))
                pt2 = tuple(keypoints[j, :2].astype(int))
                cv2.line(mask, pt1, pt2, 255, thickness)

        return mask

    def _create_hand_mask(self, keypoints: np.ndarray, h: int, w: int) -> np.ndarray:
        """Create mask for hand regions (small circles at wrist keypoints)."""
        mask = np.zeros((h, w), dtype=np.uint8)
        wrist_indices = [4, 7]  # right_wrist, left_wrist
        radius = max(w // 25, 10)

        for idx in wrist_indices:
            if keypoints[idx, 2] > 0.3:
                center = tuple(keypoints[idx, :2].astype(int))
                cv2.circle(mask, center, radius, 255, -1)

        return mask

    def generate_noise_agnostic(self, image: np.ndarray, segmentation: np.ndarray,
                                 keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate agnostic image filled with random noise instead of gray."""
        result = self.generate(image, segmentation, keypoints)
        agnostic = result["agnostic_image"].copy()
        mask = result["mask"]
        noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        agnostic[mask > 0] = noise[mask > 0]
        return agnostic
