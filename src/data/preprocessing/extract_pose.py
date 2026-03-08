"""Batch pose extraction from raw images."""

import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from src.modules.pose_estimator import PoseEstimator


def extract_poses(image_dir: str, output_dir: str, model_type: str = "dwpose",
                  device: str = "cuda", save_visualization: bool = True):
    """Batch extract poses from all images in a directory."""
    img_dir = Path(image_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "rendered"
    if save_visualization:
        vis_dir.mkdir(exist_ok=True)

    estimator = PoseEstimator(model_type=model_type, device=device)
    estimator.load_model()

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in extensions]

    print(f"Extracting poses from {len(images)} images...")
    for img_path in tqdm(images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = estimator.estimate(image)
        stem = img_path.stem

        # Save keypoints
        np.save(out_dir / f"{stem}.npy", result["keypoints"])

        # Save visualization
        if save_visualization:
            cv2.imwrite(str(vis_dir / f"{stem}.png"), result["pose_image"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--output_dir", default="data/processed/poses")
    parser.add_argument("--model", default="dwpose")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_poses(args.image_dir, args.output_dir, args.model, args.device)
