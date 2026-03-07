"""
Diferansiyellenebilir Mesh Renderer — PyTorch3D tabanlı 3D→2D render.

SMPL-X beden mesh'i üzerine giydirilen giysi mesh'ini
diferansiyellenebilir şekilde 2D görüntüye dönüştürür.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class MeshRenderer:
    """
    PyTorch3D tabanlı diferansiyellenebilir mesh renderer.

    3D mesh + doku → 2D render görüntüsü dönüşümü yapar.
    Eğitim sırasında geri yayılım destekler.

    Args:
        image_size: Çıktı render boyutu.
        device: Hesaplama cihazı.
        shader_type: 'phong', 'flat' veya 'silhouette'.
        faces_per_pixel: Piksel başına render edilen yüz sayısı.
        bg_color: Arka plan rengi (R, G, B) [0-1].
    """

    def __init__(self, image_size: int = 512, device: str = "cuda",
                 shader_type: str = "phong", faces_per_pixel: int = 1,
                 bg_color: tuple[float, ...] = (1.0, 1.0, 1.0)):
        self.image_size = image_size
        self.device = device
        self.shader_type = shader_type
        self.faces_per_pixel = faces_per_pixel
        self.bg_color = bg_color
        self.renderer = None

    def setup(self):
        """PyTorch3D renderer bileşenlerini kur."""
        try:
            from pytorch3d.renderer import (
                MeshRenderer as PT3DMeshRenderer,
                MeshRasterizer,
                RasterizationSettings,
                SoftPhongShader,
                SoftSilhouetteShader,
                FoVPerspectiveCameras,
                PointLights,
                TexturesVertex,
                TexturesUV,
                look_at_view_transform,
            )

            # Kamera ayarları
            R, T = look_at_view_transform(dist=2.7, elev=0, azim=0)
            self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

            # Aydınlatma
            self.lights = PointLights(
                device=self.device,
                location=[[0.0, 1.0, 2.0]],
                ambient_color=((0.5, 0.5, 0.5),),
                diffuse_color=((0.7, 0.7, 0.7),),
                specular_color=((0.3, 0.3, 0.3),),
            )

            # Rasterizasyon ayarları
            raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=self.faces_per_pixel,
            )

            # Shader seçimi
            if self.shader_type == "silhouette":
                shader = SoftSilhouetteShader()
            else:
                shader = SoftPhongShader(
                    device=self.device,
                    cameras=self.cameras,
                    lights=self.lights,
                )

            self.renderer = PT3DMeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=raster_settings,
                ),
                shader=shader,
            )
            print(f"PyTorch3D renderer kuruldu: {self.shader_type}")

        except ImportError:
            print("Uyarı: PyTorch3D bulunamadı. Placeholder renderer kullanılacak.")
            self.renderer = None

    def render(self, vertices: torch.Tensor, faces: torch.Tensor,
               textures: Optional[torch.Tensor] = None,
               camera_params: Optional[dict] = None) -> torch.Tensor:
        """
        3D mesh'i 2D görüntüye render et.

        Args:
            vertices: (B, V, 3) köşe noktaları.
            faces: (F, 3) üçgen yüz indeksleri.
            textures: (B, V, 3) köşe renkleri veya (B, H, W, 3) UV doku.
            camera_params: Opsiyonel kamera parametreleri override.

        Returns:
            (B, 4, H, W) RGBA render sonucu.
        """
        if self.renderer is None:
            self.setup()

        if self.renderer is not None:
            return self._render_pytorch3d(vertices, faces, textures, camera_params)
        else:
            return self._render_placeholder(vertices.shape[0])

    def _render_pytorch3d(self, vertices, faces, textures, camera_params):
        """PyTorch3D ile render."""
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex

        b = vertices.shape[0]

        # Doku hazırla
        if textures is None:
            textures = torch.ones_like(vertices) * 0.7  # Gri varsayılan
        tex = TexturesVertex(verts_features=textures)

        # Mesh oluştur
        faces_batch = faces.unsqueeze(0).expand(b, -1, -1) if faces.dim() == 2 else faces
        meshes = Meshes(verts=vertices, faces=faces_batch, textures=tex)

        # Kamera güncelle
        if camera_params is not None:
            from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
            R, T = look_at_view_transform(
                dist=camera_params.get("dist", 2.7),
                elev=camera_params.get("elev", 0),
                azim=camera_params.get("azim", 0),
            )
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            self.renderer.rasterizer.cameras = cameras
            if hasattr(self.renderer.shader, "cameras"):
                self.renderer.shader.cameras = cameras

        # Render
        images = self.renderer(meshes)  # (B, H, W, 4) RGBA
        images = images.permute(0, 3, 1, 2)  # (B, 4, H, W)

        return images

    def _render_placeholder(self, batch_size: int) -> torch.Tensor:
        """PyTorch3D olmadan placeholder render."""
        return torch.ones(batch_size, 4, self.image_size, self.image_size,
                          device=self.device) * 0.5

    def render_normal_map(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Normal haritası render et (geometri koşullandırması için)."""
        b = vertices.shape[0]
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(b, -1, -1)

        # Yüz normallerini hesapla
        v0 = torch.gather(vertices, 1, faces[:, :, 0:1].expand(-1, -1, 3))
        v1 = torch.gather(vertices, 1, faces[:, :, 1:2].expand(-1, -1, 3))
        v2 = torch.gather(vertices, 1, faces[:, :, 2:3].expand(-1, -1, 3))

        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

        # Köşe normallerini basitle (orta değer)
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.scatter_add_(
                1,
                faces[:, :, i:i+1].expand(-1, -1, 3),
                normals,
            )
        vertex_normals = vertex_normals / (vertex_normals.norm(dim=-1, keepdim=True) + 1e-8)

        # Normal haritasını doku olarak render et
        normal_colors = (vertex_normals + 1) / 2  # [-1,1] → [0,1]
        return self.render(vertices, faces[0] if faces.dim() == 3 else faces, normal_colors)

    def render_depth_map(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Derinlik haritası render et."""
        z_values = vertices[:, :, 2:3]  # Z koordinatı
        z_min = z_values.min(dim=1, keepdim=True)[0]
        z_max = z_values.max(dim=1, keepdim=True)[0]
        depth_normalized = (z_values - z_min) / (z_max - z_min + 1e-8)
        depth_colors = depth_normalized.expand(-1, -1, 3)
        return self.render(vertices, faces, depth_colors)

    def render_multiview(self, vertices: torch.Tensor, faces: torch.Tensor,
                         textures: Optional[torch.Tensor] = None,
                         num_views: int = 4) -> list[torch.Tensor]:
        """Birden fazla açıdan render et."""
        azimuths = torch.linspace(0, 360, num_views + 1)[:-1]
        renders = []
        for azim in azimuths:
            cam_params = {"dist": 2.7, "elev": 0, "azim": float(azim)}
            img = self.render(vertices, faces, textures, cam_params)
            renders.append(img)
        return renders
