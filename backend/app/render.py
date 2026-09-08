"""
EarthPulse — image rendering for the Before / After / Change views (Feature 9).

Everything is emitted as base64 PNG data URIs so the frontend needs no image
server, no tile cache and no CORS configuration — and so the workspace preview
(which has no network access) still renders correctly.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .raster import EpochStack, rgb_preview


def _to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _upscale(img: Image.Image, target: int = 640) -> Image.Image:
    w, h = img.size
    if max(w, h) >= target:
        return img
    scale = target / max(w, h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def render_rgb(epoch: EpochStack, target: int = 640) -> str:
    arr = rgb_preview(epoch)
    return _to_data_uri(_upscale(Image.fromarray(arr, mode="RGB"), target))


def render_change_overlay(
    epoch: EpochStack,
    mask: np.ndarray,
    *,
    direction: str = "increase",
    target: int = 640,
    opacity: float = 0.62,
) -> str:
    """Latest natural-colour image with the change mask burned in."""
    base = rgb_preview(epoch).astype(np.float32)

    if direction == "increase":
        colour = np.array([255.0, 68.0, 88.0])      # red — something appeared
    else:
        colour = np.array([255.0, 176.0, 32.0])     # amber — something was lost

    out = base.copy()
    m = mask.astype(bool)
    if m.any():
        out[m] = base[m] * (1 - opacity) + colour * opacity

        # Crisp outline so small polygons stay visible after downscaling.
        from scipy import ndimage
        edge = m ^ ndimage.binary_erosion(m, np.ones((3, 3)), border_value=0)
        out[edge] = np.array([255.0, 255.0, 255.0])

    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")
    return _to_data_uri(_upscale(img, target))


def render_mask(mask: np.ndarray, *, direction: str = "increase", target: int = 640) -> str:
    """Transparent PNG of just the change mask, for map overlay."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    colour = (255, 68, 88) if direction == "increase" else (255, 176, 32)
    m = mask.astype(bool)
    rgba[m] = (*colour, 210)
    img = Image.fromarray(rgba, mode="RGBA")
    if max(w, h) < target:
        scale = target / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.NEAREST)
    return _to_data_uri(img)


_HEAT_STOPS = [
    (0.00, (10, 16, 34)),
    (0.25, (24, 62, 120)),
    (0.45, (16, 132, 148)),
    (0.65, (232, 196, 62)),
    (0.85, (238, 118, 44)),
    (1.00, (240, 52, 72)),
]


def _heat_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for k in range(len(_HEAT_STOPS) - 1):
            t0, c0 = _HEAT_STOPS[k]
            t1, c1 = _HEAT_STOPS[k + 1]
            if t0 <= t <= t1:
                f = (t - t0) / max(t1 - t0, 1e-6)
                lut[i] = [int(c0[j] + f * (c1[j] - c0[j])) for j in range(3)]
                break
        else:
            lut[i] = _HEAT_STOPS[-1][1]
    return lut


def render_heatmap(magnitude: np.ndarray, valid: np.ndarray, *,
                   direction: str = "increase", target: int = 640) -> str:
    """Continuous change-magnitude heatmap (Feature 14 map layer)."""
    m = magnitude.copy()
    if direction == "decrease":
        m = -m
    sample = m[valid] if valid.any() else m.ravel()
    if sample.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(np.percentile(sample, 2)), float(np.percentile(sample, 99))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((m - lo) / (hi - lo), 0, 1)
    idx = (norm * 255).astype(np.uint8)
    rgb = _heat_lut()[idx]
    rgb[~valid] = 22
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return _to_data_uri(_upscale(img, target))


_SCL_COLOURS = {
    0: (12, 12, 14), 1: (255, 0, 0), 2: (47, 47, 47), 3: (100, 50, 0),
    4: (46, 160, 67), 5: (196, 168, 96), 6: (30, 110, 220), 7: (128, 128, 128),
    8: (200, 200, 200), 9: (245, 245, 245), 10: (170, 200, 235), 11: (180, 230, 255),
}


def render_scl(epoch: EpochStack, target: int = 640) -> str:
    """Scene-classification map — the visual proof behind the cloud filter."""
    h, w = epoch.scl.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for code, colour in _SCL_COLOURS.items():
        rgb[epoch.scl == code] = colour
    return _to_data_uri(_upscale(Image.fromarray(rgb, mode="RGB"), target))


def render_thumb(epoch: EpochStack, size: int = 132) -> str:
    """Small timeline thumbnail."""
    arr = rgb_preview(epoch)
    img = Image.fromarray(arr, mode="RGB").resize((size, size), Image.LANCZOS)
    return _to_data_uri(img)
