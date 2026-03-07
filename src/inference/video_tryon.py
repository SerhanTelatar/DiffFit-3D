"""
Video Virtual Try-On with Temporal Consistency.

Processes video frames with temporal attention for consistent garment rendering.
"""

from typing import Optional
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.inference.image_tryon import ImageTryOn
from src.utils.video_utils import extract_frames, encode_video


class VideoTryOn:
    """
    Video virtual try-on with temporal consistency.

    Args:
        image_tryon: Single-frame try-on pipeline.
        config: Video configuration dict.
    """

    def __init__(self, image_tryon: ImageTryOn, config: Optional[dict] = None):
        self.image_tryon = image_tryon
        self.config = config or {}
        self.num_frames = self.config.get("num_frames", 24)
        self.fps = self.config.get("fps", 8)
        self.temporal_window = self.config.get("temporal_window", 4)

    @torch.no_grad()
    def run(self, person_video: str, garment_image: str | np.ndarray | Image.Image,
            output_path: str, max_frames: Optional[int] = None) -> str:
        """
        Run video try-on.

        Args:
            person_video: Path to person video.
            garment_image: Garment image.
            output_path: Output video path.
            max_frames: Maximum number of frames to process.

        Returns:
            Output video path.
        """
        # Extract frames
        frames = extract_frames(person_video, max_frames=max_frames)
        print(f"Processing {len(frames)} frames...")

        result_frames = []
        prev_result = None

        for i, frame in enumerate(tqdm(frames, desc="Video try-on")):
            # Run single-frame try-on
            result = self.image_tryon.run(frame, garment_image)
            result_np = np.array(result)

            # Apply temporal smoothing with previous frame
            if prev_result is not None:
                alpha = 0.85  # Blend factor for temporal consistency
                result_np = (alpha * result_np + (1 - alpha) * prev_result).astype(np.uint8)

            result_frames.append(result_np)
            prev_result = result_np.astype(np.float32)

        # Encode to video
        encode_video(result_frames, output_path, fps=self.fps)
        print(f"Video saved to: {output_path}")
        return output_path

    @torch.no_grad()
    def run_with_temporal_attention(self, person_video: str,
                                    garment_image: str | np.ndarray | Image.Image,
                                    output_path: str) -> str:
        """
        Run video try-on with temporal attention for better consistency.
        Processes frames in sliding windows.
        """
        frames = extract_frames(person_video)
        result_frames = []

        # Process in overlapping windows
        window_size = self.temporal_window
        stride = window_size // 2

        for start in tqdm(range(0, len(frames), stride), desc="Temporal processing"):
            end = min(start + window_size, len(frames))
            window_frames = frames[start:end]

            window_results = []
            for frame in window_frames:
                result = self.image_tryon.run(frame, garment_image)
                window_results.append(np.array(result))

            if start == 0:
                result_frames.extend(window_results)
            else:
                # Blend overlapping region
                overlap = stride
                for i in range(min(overlap, len(window_results))):
                    idx = len(result_frames) - overlap + i
                    if idx < len(result_frames):
                        alpha = i / overlap
                        result_frames[idx] = (
                            (1 - alpha) * result_frames[idx] + alpha * window_results[i]
                        ).astype(np.uint8)
                result_frames.extend(window_results[overlap:])

        encode_video(result_frames, output_path, fps=self.fps)
        return output_path
