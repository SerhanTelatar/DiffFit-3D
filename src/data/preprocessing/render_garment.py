"""Batch 3D garment rendering: mesh → 2D render for conditioning."""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm

from src.modules.mesh_renderer import MeshRenderer
from src.modules.garment_draper import GarmentDraper, load_garment_mesh
from src.modules.smplx_estimator import SMPLXEstimator


def render_garments(garments_dir: str, smplx_params_dir: str, output_dir: str,
                    normal_maps_dir: str = None, depth_maps_dir: str = None,
                    resolution: int = 512, device: str = "cuda",
                    num_views: int = 1):
    """
    3D giysi mesh'lerini SMPL-X beden modeline giydirip 2D'ye render et.

    Her (kişi, giysi) çifti için:
    1. Kişinin SMPL-X parametrelerini yükle
    2. Giysi mesh'ini yükle
    3. Giysi mesh'ini beden modeline drape et
    4. Drape edilmiş giysiyi 2D'ye render et
    """
    garments = Path(garments_dir)
    params_dir = Path(smplx_params_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if normal_maps_dir:
        Path(normal_maps_dir).mkdir(parents=True, exist_ok=True)
    if depth_maps_dir:
        Path(depth_maps_dir).mkdir(parents=True, exist_ok=True)

    # Renderer ve Draper kur
    renderer = MeshRenderer(image_size=resolution, device=device)
    renderer.setup()

    draper = GarmentDraper().to(device)
    draper.eval()

    smplx_est = SMPLXEstimator(device=device)

    # Giysi mesh'lerini bul
    mesh_extensions = {".obj", ".glb", ".ply", ".off"}
    garment_files = []
    for ext in mesh_extensions:
        garment_files.extend(garments.rglob(f"*{ext}"))

    if not garment_files:
        print(f"Uyarı: {garments_dir} altında mesh dosyası bulunamadı!")
        return

    # Kişi parametrelerini bul
    person_params = sorted(params_dir.glob("*.npz"))

    print(f"Render: {len(garment_files)} giysi × {len(person_params)} kişi")

    for garment_path in tqdm(garment_files, desc="Giysi render"):
        garment_id = garment_path.stem
        garment_data = load_garment_mesh(str(garment_path))

        garment_verts = torch.tensor(
            garment_data["vertices"], dtype=torch.float32
        ).unsqueeze(0).to(device)

        garment_faces = torch.tensor(
            garment_data["faces"], dtype=torch.long
        ).to(device)

        for param_path in person_params:
            person_id = param_path.stem
            output_name = f"{person_id}_{garment_id}"

            # SMPL-X parametrelerini yükle
            params = SMPLXEstimator.load_params(str(param_path))
            body_mesh = smplx_est.get_body_mesh(params)

            body_verts = torch.tensor(
                body_mesh["vertices"], dtype=torch.float32
            ).unsqueeze(0).to(device)

            body_faces = torch.tensor(
                body_mesh["faces"], dtype=torch.long
            ).to(device)

            # Giysiyi bedene drape et
            with torch.no_grad():
                drape_result = draper(
                    garment_verts, garment_faces,
                    body_verts, body_faces,
                )

            draped_verts = drape_result["draped_verts"]

            # Doku hazırla
            texture_img = garment_data.get("texture")
            if texture_img is not None and len(texture_img.shape) == 3:
                # Basit vertex coloring (UV mapping tam versiyonda)
                tex_colors = torch.ones_like(draped_verts) * 0.7
            else:
                tex_colors = torch.ones_like(draped_verts) * 0.7

            # Ana render
            render = renderer.render(draped_verts, garment_faces, tex_colors)
            render_rgb = render[:, :3]  # RGBA → RGB

            # Tensor → numpy → kaydet
            render_np = (render_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{output_name}.png"), render_np[:, :, ::-1])

            # Normal haritası
            if normal_maps_dir:
                normal_render = renderer.render_normal_map(draped_verts, garment_faces)
                normal_np = (normal_render[0, :3].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(Path(normal_maps_dir) / f"{output_name}.png"), normal_np[:, :, ::-1])

            # Derinlik haritası
            if depth_maps_dir:
                depth_render = renderer.render_depth_map(draped_verts, garment_faces)
                depth_np = (depth_render[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(Path(depth_maps_dir) / f"{output_name}.png"), depth_np)

    print(f"3D render tamamlandı: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D giysi render")
    parser.add_argument("--garments_dir", default="data/garments_3d")
    parser.add_argument("--smplx_params_dir", default="data/processed/smplx_params")
    parser.add_argument("--output_dir", default="data/processed/renders_3d")
    parser.add_argument("--normal_maps_dir", default="data/processed/normal_maps")
    parser.add_argument("--depth_maps_dir", default="data/processed/depth_maps")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    render_garments(args.garments_dir, args.smplx_params_dir, args.output_dir,
                    args.normal_maps_dir, args.depth_maps_dir, args.resolution, args.device)
