"""Batch human segmentation extraction."""

import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from src.modules.segmentation import HumanSegmentation


def extract_segments(image_dir: str, output_dir: str, model_type: str = "atr",
                     device: str = "cuda", save_visualization: bool = True):
    img_dir, out_dir = Path(image_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = HumanSegmentation(model_type=model_type, device=device)
    seg.load_model()

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in extensions]

    print(f"Segmenting {len(images)} images...")
    for img_path in tqdm(images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        result = seg.segment(image)
        stem = img_path.stem
        Image.fromarray(result["segmentation"]).save(out_dir / f"{stem}.png")
        if save_visualization:
            vis_dir = out_dir / "visualized"
            vis_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(vis_dir / f"{stem}.png"), result["visualization"][:, :, ::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--output_dir", default="data/processed/segments")
    parser.add_argument("--model", default="atr")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_segments(args.image_dir, args.output_dir, args.model, args.device)
