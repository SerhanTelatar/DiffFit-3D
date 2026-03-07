"""Tests for the TryOnPipeline forward pass and component integration."""

import pytest
import torch
from src.models.person_unet import PersonUNet, TimestepEmbedding, ResBlock
from src.models.garment_unet import GarmentUNet
from src.models.vae import VAEWrapper
from src.models.noise_scheduler import create_noise_scheduler, DDPMScheduler, DDIMScheduler
from src.models.tryon_pipeline import TryOnPipeline


class TestTimestepEmbedding:
    def test_output_shape(self):
        emb = TimestepEmbedding(dim=320)
        t = torch.tensor([0, 100, 500, 999])
        out = emb(t)
        assert out.shape == (4, 320 * 4)

    def test_different_timesteps_produce_different_embeddings(self):
        emb = TimestepEmbedding(dim=64)
        t1 = emb(torch.tensor([0]))
        t2 = emb(torch.tensor([500]))
        assert not torch.allclose(t1, t2)


class TestResBlock:
    def test_same_channels(self):
        block = ResBlock(64, 64, time_emb_dim=256)
        x = torch.randn(2, 64, 16, 16)
        t = torch.randn(2, 256)
        out = block(x, t)
        assert out.shape == (2, 64, 16, 16)

    def test_different_channels(self):
        block = ResBlock(64, 128, time_emb_dim=256)
        x = torch.randn(2, 64, 16, 16)
        t = torch.randn(2, 256)
        out = block(x, t)
        assert out.shape == (2, 128, 16, 16)


class TestNoiseScheduler:
    def test_ddpm_creation(self):
        scheduler = create_noise_scheduler("ddpm", num_train_timesteps=100)
        assert isinstance(scheduler, DDPMScheduler)
        assert len(scheduler.betas) == 100

    def test_ddim_creation(self):
        scheduler = create_noise_scheduler("ddim", num_train_timesteps=1000, num_inference_steps=50)
        assert isinstance(scheduler, DDIMScheduler)
        assert len(scheduler.timesteps) == 50

    def test_add_noise(self):
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        x = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(x)
        t = torch.tensor([100, 500])
        noisy = scheduler.add_noise(x, noise, t)
        assert noisy.shape == x.shape
        assert not torch.allclose(noisy, x)


class TestVAE:
    def test_encode_decode(self):
        vae = VAEWrapper(latent_channels=4, block_out_channels=(32, 64))
        x = torch.randn(1, 3, 64, 64)
        enc = vae.encode(x)
        assert enc["latent"].shape[1] == 4
        assert enc["mean"].shape[1] == 4

    def test_decode_shape(self):
        vae = VAEWrapper(latent_channels=4, block_out_channels=(32, 64))
        z = torch.randn(1, 4, 8, 8)
        out = vae.decode(z)
        assert out.shape[1] == 3  # RGB output


class TestPipelineCreation:
    def test_from_config(self):
        config = {
            "person_in_channels": 9,
            "garment_in_channels": 4,
            "model_channels": 32,
            "out_channels": 4,
            "context_dim": 64,
            "latent_channels": 4,
            "scheduler_type": "ddpm",
            "num_train_timesteps": 100,
        }
        pipeline = TryOnPipeline.from_config(config)
        assert isinstance(pipeline.person_unet, PersonUNet)
        assert isinstance(pipeline.garment_unet, GarmentUNet)
        assert isinstance(pipeline.vae, VAEWrapper)
