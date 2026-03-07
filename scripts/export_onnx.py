"""Export model to ONNX: python scripts/export_onnx.py --config configs/inference.yaml"""

import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf

from src.models.tryon_pipeline import TryOnPipeline


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--output", default="checkpoints/exported")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build pipeline
    pipeline = TryOnPipeline.from_config(config.get("model", {}))
    pipeline.eval()

    # Export VAE Encoder
    print("Exporting VAE Encoder...")
    dummy_img = torch.randn(1, 3, 512, 512)
    torch.onnx.export(
        pipeline.vae.encoder, dummy_img,
        str(output_dir / "vae_encoder.onnx"),
        opset_version=args.opset,
        input_names=["image"], output_names=["latent"],
        dynamic_axes={"image": {0: "batch"}, "latent": {0: "batch"}},
    )

    # Export VAE Decoder
    print("Exporting VAE Decoder...")
    dummy_latent = torch.randn(1, 4, 64, 64)
    torch.onnx.export(
        pipeline.vae.decoder, dummy_latent,
        str(output_dir / "vae_decoder.onnx"),
        opset_version=args.opset,
        input_names=["latent"], output_names=["image"],
        dynamic_axes={"latent": {0: "batch"}, "image": {0: "batch"}},
    )

    print(f"Models exported to: {output_dir}")
    for f in output_dir.glob("*.onnx"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
