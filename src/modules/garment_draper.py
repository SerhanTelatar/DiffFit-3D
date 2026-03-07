"""
Garment Draper — 3D giysi mesh'ini SMPL-X beden modeline giydirme.

3D giysi asset'ını (mesh) SMPL-X beden mesh'ine deform ederek
beden şekline uyduran modül.
"""

from typing import Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GarmentDraper(nn.Module):
    """
    3D giysi mesh'ini beden modeline drape eder.

    İki aşamalı süreç:
    1. Coarse draping: Giysi mesh'ini beden mesh'ine Linear Blend Skinning (LBS) ile deform et
    2. Fine draping: Neural network ile fizik-tabanlı ince ayar (kırışıklar, sarkmalar)

    Args:
        num_body_verts: SMPL-X beden mesh köşe sayısı (10475).
        garment_feature_dim: Giysi özellik boyutu.
        hidden_dim: Gizli katman boyutu.
        num_refine_layers: İnce ayar katman sayısı.
    """

    def __init__(self, num_body_verts: int = 10475, garment_feature_dim: int = 256,
                 hidden_dim: int = 512, num_refine_layers: int = 4):
        super().__init__()
        self.num_body_verts = num_body_verts

        # Coarse draping: Giysi→Beden eşleştirme ağı
        self.correspondence_net = nn.Sequential(
            nn.Linear(6, hidden_dim),  # Giysi vertex (3) + nearest body vertex (3)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Offset vektörü
        )

        # Fine draping: Fizik-tabanlı ince ayar
        self.refine_layers = nn.ModuleList()
        for i in range(num_refine_layers):
            self.refine_layers.append(
                GarmentRefineBlock(
                    in_dim=3 + garment_feature_dim if i == 0 else hidden_dim,
                    out_dim=hidden_dim if i < num_refine_layers - 1 else 3,
                )
            )

        # Kumaş malzeme özellikleri etkisi
        self.material_mlp = nn.Sequential(
            nn.Linear(8, 64),  # 8 malzeme parametresi
            nn.ReLU(),
            nn.Linear(64, garment_feature_dim),
        )

    def forward(self, garment_verts: torch.Tensor, garment_faces: torch.Tensor,
                body_verts: torch.Tensor, body_faces: torch.Tensor,
                skinning_weights: Optional[torch.Tensor] = None,
                material_params: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Giysi mesh'ini beden mesh'ine drape et.

        Args:
            garment_verts: (B, Vg, 3) giysi köşe noktaları (T-pose'da).
            garment_faces: (Fg, 3) giysi yüz indeksleri.
            body_verts: (B, Vb, 3) SMPL-X beden köşeleri (hedef poz).
            body_faces: (Fb, 3) beden yüz indeksleri.
            skinning_weights: (Vg, J) LBS ağırlıkları (opsiyonel).
            material_params: (B, 8) kumaş malzeme parametreleri.

        Returns:
            Dict:
                'draped_verts': (B, Vg, 3) deform edilmiş giysi köşeleri
                'offsets': (B, Vg, 3) ince ayar offset'leri
                'normals': (B, Vg, 3) yüzey normalleri
        """
        b, vg, _ = garment_verts.shape

        # 1. Coarse Draping: En yakın beden noktasına göre deformasyon
        coarse_verts = self._coarse_drape(garment_verts, body_verts)

        # 2. Malzeme özelliklerini hesapla
        if material_params is not None:
            mat_feat = self.material_mlp(material_params)  # (B, feat_dim)
            mat_feat = mat_feat.unsqueeze(1).expand(-1, vg, -1)  # (B, Vg, feat_dim)
        else:
            mat_feat = torch.zeros(b, vg, 256, device=garment_verts.device)

        # 3. Fine Draping: Detay ince ayarı
        x = torch.cat([coarse_verts, mat_feat], dim=-1)
        for layer in self.refine_layers:
            x = layer(x)
        offsets = x  # (B, Vg, 3) — ince deformasyon offset'leri

        # Çarpışma önleme: Beden mesh'inin dışında kalmasını sağla
        draped_verts = coarse_verts + offsets * 0.05  # Küçük ölçekli offset
        draped_verts = self._collision_handling(draped_verts, body_verts)

        # Normalleri hesapla
        normals = self._compute_normals(draped_verts, garment_faces)

        return {
            "draped_verts": draped_verts,
            "offsets": offsets,
            "normals": normals,
        }

    def _coarse_drape(self, garment_verts: torch.Tensor,
                      body_verts: torch.Tensor) -> torch.Tensor:
        """Giysi köşelerini en yakın beden noktalarına göre yerleştir."""
        b, vg, _ = garment_verts.shape

        # Her giysi köşesinin en yakın beden noktasını bul
        # (B, Vg, 1, 3) - (B, 1, Vb, 3) → (B, Vg, Vb)
        dists = torch.cdist(garment_verts, body_verts)
        nearest_idx = dists.argmin(dim=-1)  # (B, Vg)

        # En yakın beden noktalarını topla
        nearest_body = torch.gather(
            body_verts, 1,
            nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        )

        # Correspondence network ile offset hesapla
        combined = torch.cat([garment_verts, nearest_body], dim=-1)  # (B, Vg, 6)
        offsets = self.correspondence_net(combined)

        return nearest_body + offsets

    def _collision_handling(self, garment_verts: torch.Tensor,
                            body_verts: torch.Tensor,
                            margin: float = 0.005) -> torch.Tensor:
        """Giysi-beden çarpışmasını önle (basit itme yöntemi)."""
        dists = torch.cdist(garment_verts, body_verts)
        min_dists, nearest_idx = dists.min(dim=-1)

        nearest_body = torch.gather(
            body_verts, 1,
            nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        )

        # Yön vektörü (giysi → dışarı)
        direction = garment_verts - nearest_body
        direction_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Çok yakın köşeleri dışarı it
        too_close = min_dists < margin
        push = direction_norm * margin
        correction = push * too_close.unsqueeze(-1).float()

        return garment_verts + correction

    def _compute_normals(self, vertices: torch.Tensor,
                         faces: torch.Tensor) -> torch.Tensor:
        """Köşe normallerini hesapla."""
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1)

        v0 = torch.gather(vertices, 1, faces[:, :, 0:1].expand(-1, -1, 3))
        v1 = torch.gather(vertices, 1, faces[:, :, 1:2].expand(-1, -1, 3))
        v2 = torch.gather(vertices, 1, faces[:, :, 2:3].expand(-1, -1, 3))

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)

        # Köşe normallerini yüz normallerinden ortalayarak hesapla
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.scatter_add_(1, faces[:, :, i:i+1].expand(-1, -1, 3), face_normals)
        vertex_normals = vertex_normals / (vertex_normals.norm(dim=-1, keepdim=True) + 1e-8)

        return vertex_normals


class GarmentRefineBlock(nn.Module):
    """Giysi deformasyon ince ayar bloğu."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


def load_garment_mesh(path: str) -> dict[str, np.ndarray]:
    """
    3D giysi mesh dosyasını yükle (.obj, .glb, .ply).

    Args:
        path: Mesh dosya yolu.

    Returns:
        Dict: 'vertices' (V, 3), 'faces' (F, 3), 'uv' (V, 2), 'texture' (H, W, 3)
    """
    ext = Path(path).suffix.lower()

    try:
        import trimesh

        mesh = trimesh.load(path, force="mesh")

        result = {
            "vertices": np.array(mesh.vertices, dtype=np.float32),
            "faces": np.array(mesh.faces, dtype=np.int64),
        }

        # UV koordinatları
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            result["uv"] = np.array(mesh.visual.uv, dtype=np.float32)
        else:
            result["uv"] = np.zeros((len(mesh.vertices), 2), dtype=np.float32)

        # Doku
        if hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "image"):
            result["texture"] = np.array(mesh.visual.material.image)[:, :, :3]
        else:
            result["texture"] = np.ones((512, 512, 3), dtype=np.uint8) * 200

        return result

    except ImportError:
        print("Uyarı: trimesh bulunamadı. pip install trimesh ile yükleyin.")
        return {
            "vertices": np.zeros((100, 3), dtype=np.float32),
            "faces": np.zeros((50, 3), dtype=np.int64),
            "uv": np.zeros((100, 2), dtype=np.float32),
            "texture": np.ones((512, 512, 3), dtype=np.uint8) * 200,
        }
