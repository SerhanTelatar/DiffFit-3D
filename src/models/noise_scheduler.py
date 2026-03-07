"""
Noise Schedulers for Diffusion Process.

Implements DDPM and DDIM noise schedulers with configurable beta schedules.
"""

import math
from typing import Optional
import torch
import numpy as np


def _linear_beta_schedule(n: int, start: float, end: float) -> torch.Tensor:
    return torch.linspace(start, end, n, dtype=torch.float64)


def _scaled_linear_beta_schedule(n: int, start: float, end: float) -> torch.Tensor:
    return torch.linspace(start**0.5, end**0.5, n, dtype=torch.float64) ** 2


def _cosine_beta_schedule(n: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, n, n + 1, dtype=torch.float64)
    ac = torch.cos(((x / n) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clamp(betas, 0, 0.999)


def _get_betas(schedule: str, n: int, start: float, end: float) -> torch.Tensor:
    if schedule == "linear":
        return _linear_beta_schedule(n, start, end)
    elif schedule == "scaled_linear":
        return _scaled_linear_beta_schedule(n, start, end)
    elif schedule == "cosine":
        return _cosine_beta_schedule(n)
    raise ValueError(f"Unknown schedule: {schedule}")


class DDPMScheduler:
    """DDPM noise scheduler."""

    def __init__(self, num_train_timesteps=1000, beta_start=0.00085,
                 beta_end=0.012, beta_schedule="scaled_linear",
                 prediction_type="epsilon", clip_sample=False):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        self.betas = _get_betas(beta_schedule, num_train_timesteps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        device = original.device
        sa = self.sqrt_alphas_cumprod.to(device)[timesteps].float()
        som = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps].float()
        while sa.ndim < original.ndim:
            sa = sa.unsqueeze(-1)
            som = som.unsqueeze(-1)
        return sa * original + som * noise

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        device = sample.device
        t = timestep
        sa = self.sqrt_alphas_cumprod.to(device)[t].float()
        som = self.sqrt_one_minus_alphas_cumprod.to(device)[t].float()

        if self.prediction_type == "epsilon":
            pred_x0 = (sample - som * model_output) / sa
        elif self.prediction_type == "v_prediction":
            pred_x0 = sa * sample - som * model_output
        else:
            pred_x0 = model_output

        if self.clip_sample:
            pred_x0 = pred_x0.clamp(-1, 1)

        c1 = self.posterior_mean_coef1.to(device)[t].float()
        c2 = self.posterior_mean_coef2.to(device)[t].float()
        while c1.ndim < sample.ndim:
            c1, c2 = c1.unsqueeze(-1), c2.unsqueeze(-1)

        mean = c1 * pred_x0 + c2 * sample
        if t > 0:
            var = self.posterior_variance.to(device)[t].float()
            while var.ndim < sample.ndim:
                var = var.unsqueeze(-1)
            noise = torch.randn(sample.shape, generator=generator, device=device, dtype=sample.dtype)
            return mean + torch.sqrt(var) * noise
        return mean


class DDIMScheduler:
    """DDIM noise scheduler for faster inference."""

    def __init__(self, num_train_timesteps=1000, num_inference_steps=50,
                 beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                 prediction_type="epsilon", eta=0.0):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type
        self.eta = eta
        betas = _get_betas(beta_schedule, num_train_timesteps, beta_start, beta_end)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.timesteps = self._compute_timesteps()

    def _compute_timesteps(self) -> torch.Tensor:
        ratio = self.num_train_timesteps // self.num_inference_steps
        ts = (np.arange(0, self.num_inference_steps) * ratio).round()[::-1].copy().astype(np.int64)
        return torch.from_numpy(ts)

    def set_timesteps(self, n: int):
        self.num_inference_steps = n
        self.timesteps = self._compute_timesteps()

    def add_noise(self, original, noise, timesteps):
        device = original.device
        ap = self.alphas_cumprod.to(device)[timesteps].float()
        while ap.ndim < original.ndim:
            ap = ap.unsqueeze(-1)
        return torch.sqrt(ap) * original + torch.sqrt(1 - ap) * noise

    def step(self, model_output, timestep, sample, prev_timestep=None, generator=None):
        device = sample.device
        apt = self.alphas_cumprod.to(device)[timestep].float()
        if prev_timestep is None:
            idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
            prev_timestep = self.timesteps[idx + 1].item() if idx + 1 < len(self.timesteps) else 0
        app = self.alphas_cumprod.to(device)[prev_timestep].float() if prev_timestep > 0 else torch.tensor(1.0, device=device)

        if self.prediction_type == "epsilon":
            pred_x0 = (sample - torch.sqrt(1 - apt) * model_output) / torch.sqrt(apt)
        elif self.prediction_type == "v_prediction":
            pred_x0 = torch.sqrt(apt) * sample - torch.sqrt(1 - apt) * model_output
        else:
            pred_x0 = model_output

        sigma = self.eta * torch.sqrt((1 - app) / (1 - apt) * (1 - apt / app))
        pred_dir = torch.sqrt(1 - app - sigma**2) * model_output
        x_prev = torch.sqrt(app) * pred_x0 + pred_dir
        if sigma > 0:
            noise = torch.randn(sample.shape, generator=generator, device=device, dtype=sample.dtype)
            x_prev = x_prev + sigma * noise
        return x_prev


def create_noise_scheduler(scheduler_type="ddpm", **kwargs):
    """Factory function to create a noise scheduler."""
    if scheduler_type == "ddpm":
        return DDPMScheduler(**kwargs)
    elif scheduler_type == "ddim":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unknown scheduler: {scheduler_type}")
