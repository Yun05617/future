"""Decode inputs before loading large models; keep independent BGR pixels."""
import numpy as np
from PIL import Image, ImageOps

def load_bgr_image(path):
    try:
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            return np.asarray(rgb)[:, :, ::-1].copy()
    except (OSError, ValueError) as exc:
        raise ValueError(f'無法讀取照片：{path} ({exc})') from exc
