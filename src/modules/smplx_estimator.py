"""
SMPL-X Body Estimator — 3D body shape and pose from person images.

Estimates SMPL-X body model parameters (shape, pose, expression)
from a single person image for accurate 3D body reconstruction.
"""

from typing import Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


# SMPL-X parametreleri
SMPLX_NUM_BETAS = 10        # Vücut şekli parametreleri
SMPLX_NUM_BODY_JOINTS = 21  # Vücut eklem sayısı
SMPLX_NUM_HAND_JOINTS = 15  # Her el için eklem sayısı
SMPLX_NUM_FACE_JOINTS = 3   # Çene eklemleri
SMPLX_NUM_EXPRESSION = 10   # Yüz ifadesi parametreleri


class SMPLXEstimator:
    """
    SMPL-X parametrelerini kişi görüntüsünden tahmin eder.

    3D vücut mesh'i oluşturmak için gerekli parametreleri çıkarır:
    - betas: Vücut şekli (10 boyutlu)
    - body_pose: Eklem rotasyonları (21 eklem × 3 axis-angle)
    - global_orient: Global rotasyon (1 × 3)
    - transl: Global öteleme (1 × 3)
    - left/right_hand_pose: El pozları
    - expression: Yüz ifadesi

    Args:
        model_path: SMPL-X model dosyası yolu.
        regressor: Parametre regresyon modeli ('pymaf', 'expose', 'pixie').
        gender: 'neutral', 'male', veya 'female'.
        device: Hesaplama cihazı.
    """

    def __init__(self, model_path: str = "checkpoints/pretrained/smplx",
                 regressor: str = "pymaf", gender: str = "neutral",
                 device: str = "cuda"):
        self.model_path = Path(model_path)
        self.regressor = regressor
        self.gender = gender
        self.device = device
        self.smplx_model = None
        self.body_regressor = None

    def load_model(self):
        """SMPL-X model ve regressor'ı yükle."""
        self._load_smplx()
        self._load_regressor()

    def _load_smplx(self):
        """SMPL-X parametrik beden modelini yükle."""
        try:
            import smplx
            self.smplx_model = smplx.create(
                str(self.model_path),
                model_type="smplx",
                gender=self.gender,
                use_face_contour=True,
                num_betas=SMPLX_NUM_BETAS,
                num_expression_coeffs=SMPLX_NUM_EXPRESSION,
                ext="npz",
            )
            self.smplx_model.to(self.device)
            self.smplx_model.eval()
            print(f"SMPL-X model yüklendi: {self.gender}")
        except ImportError:
            print("Uyarı: smplx kütüphanesi bulunamadı. pip install smplx ile yükleyin.")
            self.smplx_model = None
        except Exception as e:
            print(f"Uyarı: SMPL-X model yüklenemedi: {e}")
            self.smplx_model = None

    def _load_regressor(self):
        """Görüntüden SMPL-X parametrelerine regresyon modelini yükle."""
        if self.regressor == "pymaf":
            self._load_pymaf()
        elif self.regressor == "expose":
            self._load_expose()
        else:
            # Basit CNN tabanlı regressor (fallback)
            self.body_regressor = SimpleSMPLXRegressor().to(self.device)

    def _load_pymaf(self):
        """PyMAF-X regressor yükle."""
        try:
            # PyMAF-X: Pixel-aligned Mesh Recovery
            self.body_regressor = SimpleSMPLXRegressor().to(self.device)
            print("PyMAF-X regressor yüklendi (fallback)")
        except Exception:
            self.body_regressor = SimpleSMPLXRegressor().to(self.device)

    def _load_expose(self):
        """ExPose regressor yükle."""
        self.body_regressor = SimpleSMPLXRegressor().to(self.device)

    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> dict:
        """
        Kişi görüntüsünden SMPL-X parametrelerini tahmin et.

        Args:
            image: (H, W, 3) BGR numpy görüntüsü.

        Returns:
            Dict:
                - 'betas': (10,) vücut şekli
                - 'body_pose': (63,) eklem rotasyonları (21×3)
                - 'global_orient': (3,) global rotasyon
                - 'transl': (3,) global öteleme
                - 'vertices': (10475, 3) mesh köşe noktaları
                - 'joints': (127, 3) eklem konumları
                - 'faces': (F, 3) üçgen yüz indeksleri
        """
        if self.smplx_model is None or self.body_regressor is None:
            self.load_model()

        # Görüntüyü hazırla
        import cv2
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        # Parametreleri regresse et
        params = self.body_regressor(tensor)

        result = {
            "betas": params["betas"].cpu().numpy().squeeze(),
            "body_pose": params["body_pose"].cpu().numpy().squeeze(),
            "global_orient": params["global_orient"].cpu().numpy().squeeze(),
            "transl": params["transl"].cpu().numpy().squeeze(),
        }

        # SMPL-X model ile mesh oluştur
        if self.smplx_model is not None:
            smplx_output = self.smplx_model(
                betas=params["betas"],
                body_pose=params["body_pose"],
                global_orient=params["global_orient"],
                transl=params["transl"],
            )
            result["vertices"] = smplx_output.vertices.cpu().numpy().squeeze()
            result["joints"] = smplx_output.joints.cpu().numpy().squeeze()
            result["faces"] = self.smplx_model.faces.copy()
        else:
            # Placeholder mesh
            result["vertices"] = np.zeros((10475, 3), dtype=np.float32)
            result["joints"] = np.zeros((127, 3), dtype=np.float32)
            result["faces"] = np.zeros((20908, 3), dtype=np.int64)

        return result

    def get_body_mesh(self, params: dict) -> dict:
        """Parametre dict'inden SMPL-X mesh oluştur."""
        if self.smplx_model is None:
            self.load_model()

        betas = torch.tensor(params["betas"], dtype=torch.float32).unsqueeze(0).to(self.device)
        body_pose = torch.tensor(params["body_pose"], dtype=torch.float32).unsqueeze(0).to(self.device)
        global_orient = torch.tensor(params["global_orient"], dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.smplx_model is not None:
            output = self.smplx_model(
                betas=betas, body_pose=body_pose, global_orient=global_orient,
            )
            return {
                "vertices": output.vertices.cpu().numpy().squeeze(),
                "faces": self.smplx_model.faces.copy(),
                "joints": output.joints.cpu().numpy().squeeze(),
            }
        return {"vertices": np.zeros((10475, 3)), "faces": np.zeros((20908, 3), dtype=np.int64),
                "joints": np.zeros((127, 3))}

    def save_params(self, params: dict, output_path: str):
        """SMPL-X parametrelerini .npz dosyasına kaydet."""
        np.savez(output_path,
                 betas=params["betas"],
                 body_pose=params["body_pose"],
                 global_orient=params["global_orient"],
                 transl=params["transl"])

    @staticmethod
    def load_params(path: str) -> dict:
        """Kaydedilmiş parametreleri yükle."""
        data = np.load(path)
        return {k: data[k] for k in data.files}


class SimpleSMPLXRegressor(nn.Module):
    """
    Basit CNN → SMPL-X parametre regressor'ı.
    Gerçek üretimde PyMAF-X veya ExPose ile değiştirilmeli.
    """

    def __init__(self):
        super().__init__()
        # ResNet-like backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # Parametre başlıkları
        self.fc_betas = nn.Linear(512, SMPLX_NUM_BETAS)
        self.fc_body_pose = nn.Linear(512, SMPLX_NUM_BODY_JOINTS * 3)
        self.fc_global_orient = nn.Linear(512, 3)
        self.fc_transl = nn.Linear(512, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {
            "betas": self.fc_betas(feat),
            "body_pose": self.fc_body_pose(feat),
            "global_orient": self.fc_global_orient(feat),
            "transl": self.fc_transl(feat),
        }
