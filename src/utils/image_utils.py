"""Image I/O and conversion utilities."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def load_image(path: str, size: Optional[int] = None) -> Image.Image:
    """Load image from path and optionally resize."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def save_image(image: Union[Image.Image, torch.Tensor, np.ndarray], path: str):
    """Save image to path (supports PIL, tensor, numpy)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)


def pil_to_tensor(image: Image.Image, size: Optional[int] = None) -> torch.Tensor:
    """Convert PIL image to normalized [-1, 1] tensor."""
    if size:
        image = image.resize((size, size), Image.LANCZOS)
    tensor = TF.to_tensor(image)  # [0, 1]
    tensor = tensor * 2.0 - 1.0   # [-1, 1]
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL image. Handles both [-1,1] and [0,1] ranges."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1) / 2  # [-1,1] -> [0,1]
    return TF.to_pil_image(tensor)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy uint8 BGR array."""
    pil = tensor_to_pil(tensor)
    return np.array(pil)[:, :, ::-1]


def numpy_to_tensor(image: np.ndarray, size: Optional[int] = None) -> torch.Tensor:
    """Convert BGR numpy to normalized tensor."""
    pil = Image.fromarray(image[:, :, ::-1])
    return pil_to_tensor(pil, size)


def make_grid(images: list[torch.Tensor], nrow: int = 4, padding: int = 2) -> Image.Image:
    """Create a grid of images for visualization."""
    from torchvision.utils import make_grid as tv_make_grid
    tensors = []
    for img in images:
        if img.dim() == 3:
            tensors.append((img.clamp(-1, 1) + 1) / 2)
    if tensors:
        grid = tv_make_grid(tensors, nrow=nrow, padding=padding)
        return TF.to_pil_image(grid)
    return Image.new("RGB", (64, 64))
