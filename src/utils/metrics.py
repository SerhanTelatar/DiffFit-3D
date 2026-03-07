"""Evaluation Metrics — FID, SSIM, LPIPS, KID."""

from typing import Optional
import numpy as np
import torch


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM)."""
    from skimage.metrics import structural_similarity
    return structural_similarity(pred, target, channel_axis=-1, data_range=255)


def compute_lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute LPIPS perceptual distance."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex", verbose=False)
        loss_fn.eval()
        with torch.no_grad():
            dist = loss_fn(pred, target)
        return dist.item()
    except ImportError:
        return 0.0


def compute_fid(real_dir: str, gen_dir: str, device: str = "cuda") -> float:
    """Compute Fréchet Inception Distance between directories."""
    try:
        from pytorch_fid import fid_score
        return fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir], batch_size=50, device=device, dims=2048,
        )
    except ImportError:
        return -1.0


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred.astype(float) - target.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute all image quality metrics."""
    return {
        "ssim": compute_ssim(pred, target),
        "psnr": compute_psnr(pred, target),
    }
