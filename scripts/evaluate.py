"""Evaluation: python scripts/evaluate.py --real_dir data/test --gen_dir results/"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.metrics import compute_ssim, compute_psnr, compute_fid


def main():
    parser = argparse.ArgumentParser(description="Evaluate try-on results")
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    real_dir, gen_dir = Path(args.real_dir), Path(args.gen_dir)
    extensions = {".jpg", ".jpeg", ".png"}
    real_images = sorted([f for f in real_dir.iterdir() if f.suffix.lower() in extensions])
    gen_images = sorted([f for f in gen_dir.iterdir() if f.suffix.lower() in extensions])

    # Pair-wise metrics
    ssim_scores, psnr_scores = [], []
    paired = min(len(real_images), len(gen_images))

    for i in tqdm(range(paired), desc="Computing metrics"):
        real = np.array(Image.open(real_images[i]).convert("RGB"))
        gen = np.array(Image.open(gen_images[i]).convert("RGB").resize(
            (real.shape[1], real.shape[0])))
        ssim_scores.append(compute_ssim(real, gen))
        psnr_scores.append(compute_psnr(real, gen))

    print(f"\n{'='*40}")
    print(f"Evaluation Results ({paired} pairs)")
    print(f"{'='*40}")
    print(f"SSIM:  {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"PSNR:  {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f}")

    # FID (dataset-level)
    fid = compute_fid(str(real_dir), str(gen_dir), args.device)
    if fid >= 0:
        print(f"FID:   {fid:.2f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
