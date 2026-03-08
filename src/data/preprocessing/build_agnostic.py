"""Batch clothing-agnostic representation builder."""

import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

from src.modules.agnostic_mask import AgnosticMaskGenerator


def build_agnostic(image_dir: str, pose_dir: str, segment_dir: str,
                   output_dir: str, mask_type: str = "upper"):
    img_dir, out_dir = Path(image_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = AgnosticMaskGenerator(mask_type=mask_type)

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in extensions]

    print(f"Building agnostic representations for {len(images)} images...")
    for img_path in tqdm(images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        stem = img_path.stem
        pose_path = Path(pose_dir) / f"{stem}.npy"
        seg_path = Path(segment_dir) / f"{stem}.png"

        keypoints = np.load(pose_path) if pose_path.exists() else None
        segmentation = np.array(Image.open(seg_path)) if seg_path.exists() else np.zeros(image.shape[:2], dtype=np.uint8)

        result = generator.generate(image, segmentation, keypoints)
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), result["agnostic_image"])
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), result["mask"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--pose_dir", default="data/processed/poses")
    parser.add_argument("--segment_dir", default="data/processed/segments")
    parser.add_argument("--output_dir", default="data/processed/agnostic")
    parser.add_argument("--mask_type", default="upper")
    args = parser.parse_args()
    build_agnostic(args.image_dir, args.pose_dir, args.segment_dir, args.output_dir, args.mask_type)
