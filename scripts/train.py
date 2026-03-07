"""Training entry point: python scripts/train.py --config configs/train.yaml"""

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.models.tryon_pipeline import TryOnPipeline
from src.data.dataset import TryOnDataset
from src.data.transforms import TryOnTransforms
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="DiffFit-3D Training")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    # Set seed
    seed = args.seed or config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Config: {args.config}")
    print(f"Seed: {seed}")

    # Build model
    pipeline = TryOnPipeline.from_config(config.get("model", {}))
    num_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Build dataset
    data_cfg = config.get("data", {})
    aug_cfg = OmegaConf.load(data_cfg.get("config", "configs/data/dataset.yaml"))
    aug_cfg = OmegaConf.to_container(aug_cfg, resolve=True)

    transforms = TryOnTransforms(aug_cfg.get("augmentation", {}))
    train_dataset = TryOnDataset(
        pairs_file=aug_cfg.get("train_pairs", "data/raw/train_pairs.csv"),
        data_root=aug_cfg.get("data_root", "data"),
        resolution=aug_cfg.get("image", {}).get("resolution", 512),
        transforms=transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", True) if data_cfg.get("num_workers", 4) > 0 else False,
        drop_last=True,
    )

    # Validation dataset
    val_loader = None
    val_pairs = aug_cfg.get("val_pairs", "data/raw/val_pairs.csv")
    if Path(val_pairs).exists():
        val_dataset = TryOnDataset(
            pairs_file=val_pairs, data_root=aug_cfg.get("data_root", "data"),
            resolution=aug_cfg.get("image", {}).get("resolution", 512),
        )
        val_loader = DataLoader(val_dataset, batch_size=data_cfg.get("batch_size", 8), shuffle=False, num_workers=2)

    # Build trainer
    trainer = Trainer(pipeline, train_loader, val_loader, config)

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
