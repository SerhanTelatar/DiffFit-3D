"""Video I/O utilities — frame extraction, encoding/decoding."""

from typing import Optional
from pathlib import Path
import numpy as np


def extract_frames(video_path: str, max_frames: Optional[int] = None,
                   fps: Optional[int] = None) -> list[np.ndarray]:
    """Extract frames from a video file as BGR numpy arrays."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(src_fps / fps) if fps and fps < src_fps else 1

    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            frames.append(frame)
        idx += 1
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames


def encode_video(frames: list[np.ndarray], output_path: str, fps: int = 8,
                 codec: str = "mp4v"):
    """Encode list of BGR numpy frames to a video file."""
    import cv2
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()


def frames_to_gif(frames: list[np.ndarray], output_path: str, fps: int = 8,
                  loop: int = 0):
    """Convert frames to an animated GIF."""
    from PIL import Image
    pil_frames = [Image.fromarray(f[:, :, ::-1]) for f in frames]
    duration = int(1000 / fps)
    pil_frames[0].save(
        output_path, save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=loop, optimize=True,
    )


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["frame_count"] / max(info["fps"], 1)
    cap.release()
    return info
