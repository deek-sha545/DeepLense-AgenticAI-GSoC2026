"""
Real-galaxy source images for DeepLense Model IV.

DeepLense Model IV lenses Galaxy10 DECaLS galaxies through lenstronomy's INTERPOL
light profile. This module loads those images (or any user image), applies the same
preprocessing as DeepLenseSim (median sky subtraction from a corner patch), and
falls back to procedurally generated spiral galaxies when no image file is given.
"""

import os
from functools import lru_cache
from typing import List, Optional

import numpy as np

# Galaxy10 DECaLS selection used by DeepLenseSim Model IV
GALAXY10_CLASS = 6
GALAXY10_MAX_REDSHIFT = 0.02


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Collapse color channels, subtract the sky median from a corner patch, clip and normalize."""
    img = np.asarray(image, dtype=float)
    if img.ndim == 3:
        img = img.mean(axis=-1) if img.shape[-1] in (3, 4) else img[0]
    corner = max(4, min(img.shape) // 5)
    img = np.clip(img - np.median(img[:corner, :corner]), 0.0, None)
    total = img.sum()
    if total <= 0:
        raise ValueError("Galaxy image has no flux above the sky level.")
    return img / total


@lru_cache(maxsize=4)
def load_galaxy_images(path: str) -> tuple:
    """
    Load candidate source galaxies from a file.

    Supported: Galaxy10_DECals.h5 (class 6, z < 0.02, as in DeepLense Model IV),
    .npy arrays of shape (H, W), (N, H, W) or (H, W, 3), FITS images, and PNG/JPEG.
    """
    path = os.path.expanduser(path)
    ext = os.path.splitext(path)[1].lower()
    images: List[np.ndarray]
    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            labels = np.asarray(f["ans"])
            redshift = np.asarray(f["redshift"])
            idx = np.where((labels == GALAXY10_CLASS) & (redshift < GALAXY10_MAX_REDSHIFT))[0]
            if len(idx) == 0:
                idx = np.where(labels == GALAXY10_CLASS)[0]
            images = [f["images"][i] for i in idx[:500]]
    elif ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
            images = list(arr)
        else:
            images = [arr]
    elif ext in (".fits", ".fit", ".fz"):
        from astropy.io import fits
        images = [fits.getdata(path)]
    else:
        import matplotlib.image as mpimg
        images = [mpimg.imread(path)]
    if not images:
        raise ValueError(f"No usable galaxy images found in {path}")
    return tuple(_preprocess(img) for img in images)


def procedural_spiral_galaxy(rng: np.random.RandomState, size: int = 96) -> np.ndarray:
    """
    A seeded synthetic spiral: Sersic-like bulge, exponential disk modulated by
    logarithmic arms, and a handful of star-forming clumps, viewed at a random inclination.
    """
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    cos_i = rng.uniform(0.45, 1.0)
    pa = rng.uniform(0, np.pi)
    xr = x * np.cos(pa) + y * np.sin(pa)
    yr = (-x * np.sin(pa) + y * np.cos(pa)) / cos_i
    r = np.hypot(xr, yr) + 1e-6
    theta = np.arctan2(yr, xr)

    r_disk = rng.uniform(0.18, 0.28)
    n_arms = rng.choice([2, 2, 3])
    pitch = np.deg2rad(rng.uniform(12, 25))
    phase = rng.uniform(0, 2 * np.pi)
    arms = 1.0 + rng.uniform(0.8, 1.6) * np.cos(n_arms * (theta - np.log(r) / np.tan(pitch)) + phase) ** 8
    disk = np.exp(-r / r_disk) * arms
    bulge = rng.uniform(0.4, 1.2) * np.exp(-7.67 * ((r / rng.uniform(0.05, 0.1)) ** 0.25 - 1)) * 1e-3

    clumps = np.zeros_like(r)
    for _ in range(rng.randint(4, 10)):
        cr, ct = rng.uniform(0.1, 0.5), rng.uniform(0, 2 * np.pi)
        cx, cy = cr * np.cos(ct), cr * np.sin(ct)
        clumps += rng.uniform(0.3, 1.0) * np.exp(-((xr - cx) ** 2 + (yr - cy) ** 2) / (2 * rng.uniform(0.015, 0.03) ** 2))

    img = disk + bulge + clumps * np.exp(-r / (2 * r_disk))
    return _preprocess(img)


def draw_source_galaxy(rng: np.random.RandomState, path: Optional[str] = None) -> np.ndarray:
    """One normalized source image: drawn from `path` if given, otherwise procedural."""
    if path:
        images = load_galaxy_images(path)
        return images[rng.randint(len(images))]
    return procedural_spiral_galaxy(rng)
