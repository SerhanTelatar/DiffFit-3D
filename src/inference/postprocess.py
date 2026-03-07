"""
Post-Processing for Try-On Results.

Face restoration (GFPGAN/CodeFormer), edge smoothing, color correction.
"""

from typing import Optional
import numpy as np
from PIL import Image, ImageFilter
import cv2


class PostProcessor:
    """
    Post-processing pipeline for try-on results.

    Args:
        config: Post-processing configuration dict.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.face_restore = self.config.get("face_restore", True)
        self.edge_smooth = self.config.get("edge_smooth", True)
        self.color_correction = self.config.get("color_correction", True)
        self.edge_kernel = self.config.get("edge_smooth_kernel", 5)
        self.face_model = None

    def process(self, result: Image.Image, original_person: np.ndarray) -> Image.Image:
        """
        Apply post-processing to try-on result.

        Args:
            result: Generated try-on image (PIL).
            original_person: Original person image (BGR numpy) for reference.

        Returns:
            Post-processed PIL Image.
        """
        if self.face_restore:
            result = self._restore_face(result, original_person)
        if self.edge_smooth:
            result = self._smooth_edges(result)
        if self.color_correction:
            result = self._correct_colors(result, original_person)
        return result

    def _restore_face(self, result: Image.Image, original: np.ndarray) -> Image.Image:
        """Restore face quality using the original face region."""
        try:
            # Detect face region in original
            result_np = np.array(result)
            orig_rgb = original[:, :, ::-1]  # BGR to RGB

            # Simple face detection using haar cascade
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Expand face region slightly
                pad = int(0.2 * max(w, h))
                y1 = max(0, y - pad)
                y2 = min(result_np.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(result_np.shape[1], x + w + pad)

                # Resize original face to match result dimensions
                orig_resized = cv2.resize(orig_rgb, (result_np.shape[1], result_np.shape[0]))

                # Create blending mask (soft edges)
                mask = np.zeros((result_np.shape[0], result_np.shape[1]), dtype=np.float32)
                mask[y1:y2, x1:x2] = 1.0
                mask = cv2.GaussianBlur(mask, (21, 21), 10)
                mask = mask[:, :, np.newaxis]

                # Blend original face into result
                result_np = (result_np * (1 - mask) + orig_resized * mask).astype(np.uint8)

            return Image.fromarray(result_np)
        except Exception:
            return result

    def _smooth_edges(self, result: Image.Image) -> Image.Image:
        """Apply subtle edge smoothing to reduce artifacts."""
        result_np = np.array(result)
        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(result_np, 5, 50, 50)
        # Blend with original to preserve details
        blended = cv2.addWeighted(result_np, 0.7, smoothed, 0.3, 0)
        return Image.fromarray(blended)

    def _correct_colors(self, result: Image.Image, original: np.ndarray) -> Image.Image:
        """Match color histogram of result to original person image."""
        result_np = np.array(result)
        orig_rgb = cv2.resize(original[:, :, ::-1], (result_np.shape[1], result_np.shape[0]))

        # Simple histogram matching per channel
        for i in range(3):
            result_np[:, :, i] = self._match_histogram(result_np[:, :, i], orig_rgb[:, :, i])

        return Image.fromarray(result_np)

    @staticmethod
    def _match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram of source to reference."""
        src_vals, src_unique, src_counts = np.unique(
            source.ravel(), return_inverse=True, return_counts=True
        )
        ref_vals, _, ref_counts = np.unique(reference.ravel(), return_inverse=True, return_counts=True)

        src_cdf = np.cumsum(src_counts).astype(np.float64) / source.size
        ref_cdf = np.cumsum(ref_counts).astype(np.float64) / reference.size

        interp_values = np.interp(src_cdf, ref_cdf, ref_vals)
        return interp_values[src_unique].reshape(source.shape).astype(np.uint8)
