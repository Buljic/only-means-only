"""Image processing utilities: Lab conversion, resize, tensor conversions."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.color import rgb2lab, lab2rgb


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image to (size, size) using LANCZOS. Returns HxWxC uint8."""
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB [0,255] to float32 Lab. L in [0,100], a/b in [-128,127]."""
    return rgb2lab(img).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert float32 Lab to uint8 RGB [0,255]."""
    rgb_float = lab2rgb(lab)  # returns float64 in [0,1]
    return (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)


def normalize_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 [0,255] -> CHW float32 [0,1]."""
    t = torch.from_numpy(img).float() / 255.0
    if t.ndim == 3:
        t = t.permute(2, 0, 1)  # HWC -> CHW
    elif t.ndim == 2:
        t = t.unsqueeze(0)  # HW -> 1HW
    return t


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """CHW float32 [0,1] -> HWC uint8 [0,255]."""
    if t.ndim == 3:
        img = t.permute(1, 2, 0).detach().cpu().numpy()
    else:
        img = t.detach().cpu().numpy()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def save_comparison_grid(
    images: list[np.ndarray],
    titles: list[str],
    save_path: str | Path,
) -> None:
    """Save a row of images as a single comparison grid PNG."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
