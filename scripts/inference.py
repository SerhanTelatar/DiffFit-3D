"""Inference entry point: python scripts/inference.py --person img.jpg --garment garment.jpg"""

import argparse
from omegaconf import OmegaConf
import torch

from src.models.tryon_pipeline import TryOnPipeline
from src.inference.image_tryon import ImageTryOn
from src.inference.video_tryon import VideoTryOn


def main():
    parser = argparse.ArgumentParser(description="DiffFit-3D Inference")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--person", required=True, help="Person image/video path")
    parser.add_argument("--garment", required=True, help="Garment image path")
    parser.add_argument("--output", default="results/output.png")
    parser.add_argument("--mode", default="image", choices=["image", "video"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    # Override from CLI
    if args.steps:
        config.setdefault("sampling", {})["num_inference_steps"] = args.steps
    if args.guidance:
        config.setdefault("sampling", {})["guidance_scale"] = args.guidance
    if args.seed is not None:
        config.setdefault("sampling", {})["seed"] = args.seed

    # Build pipeline
    device = config.get("device", "cuda")
    pipeline = TryOnPipeline.from_config(config.get("model", {}))

    # Load weights
    model_cfg = config.get("model", {})
    checkpoint = model_cfg.get("person_unet_path")
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        pipeline.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")

    if args.mode == "image":
        tryon = ImageTryOn(pipeline, config)
        result = tryon.run(args.person, args.garment, args.output)
        print(f"Result saved to: {args.output}")
    else:
        tryon = ImageTryOn(pipeline, config)
        video_tryon = VideoTryOn(tryon, config.get("video", {}))
        video_tryon.run(args.person, args.garment, args.output)
        print(f"Video saved to: {args.output}")


if __name__ == "__main__":
    main()
