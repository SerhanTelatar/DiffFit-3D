"""
Data Augmentation Pipeline for Try-On Training.
"""

import random
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF


class TryOnTransforms:
    """
    Augmentation pipeline for try-on pairs.
    Ensures person and garment images receive consistent geometric transforms.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.h_flip = config.get("horizontal_flip", 0.5)
        self.color_jitter = config.get("color_jitter", {})
        self.affine = config.get("random_affine", {})

    def __call__(self, person: Image.Image, garment: Image.Image,
                 pose: np.ndarray, agnostic: Image.Image
                 ) -> Tuple[Image.Image, Image.Image, np.ndarray, Image.Image]:
        # Horizontal flip (applied consistently)
        if random.random() < self.h_flip:
            person = TF.hflip(person)
            agnostic = TF.hflip(agnostic)
            garment = TF.hflip(garment)
            if pose is not None and len(pose) > 0:
                w = person.width
                pose = pose.copy()
                pose[:, 0] = w - pose[:, 0]

        # Color jitter (person + agnostic only, NOT garment to preserve texture)
        if self.color_jitter.get("enabled", False):
            person = self._color_jitter(person)
            agnostic = self._color_jitter(agnostic)

        return person, garment, pose, agnostic

    def _color_jitter(self, img: Image.Image) -> Image.Image:
        cj = self.color_jitter
        if cj.get("brightness", 0) > 0:
            factor = 1.0 + random.uniform(-cj["brightness"], cj["brightness"])
            img = ImageEnhance.Brightness(img).enhance(factor)
        if cj.get("contrast", 0) > 0:
            factor = 1.0 + random.uniform(-cj["contrast"], cj["contrast"])
            img = ImageEnhance.Contrast(img).enhance(factor)
        if cj.get("saturation", 0) > 0:
            factor = 1.0 + random.uniform(-cj["saturation"], cj["saturation"])
            img = ImageEnhance.Color(img).enhance(factor)
        return img
