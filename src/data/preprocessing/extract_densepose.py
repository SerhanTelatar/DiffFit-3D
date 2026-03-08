"""Batch DensePose IUV map extraction."""

import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from src.modules.densepose import DensePoseExtractor


def extract_densepose(image_dir: str, output_dir: str, device: str = "cuda",
                      save_visualization: bool = True):
    img_dir, out_dir = Path(image_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = DensePoseExtractor(device=device)
    extractor.load_model()

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in extensions]

    print(f"Extracting DensePose from {len(images)} images...")
    for img_path in tqdm(images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        result = extractor.extract(image)
        stem = img_path.stem
        np.save(out_dir / f"{stem}.npy", result["iuv"])
        if save_visualization:
            vis_dir = out_dir / "visualized"
            vis_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(vis_dir / f"{stem}.png"), result["visualization"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--output_dir", default="data/processed/densepose")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_densepose(args.image_dir, args.output_dir, args.device)
