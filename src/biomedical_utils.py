"""Shared utilities for the biomedical nuclei analysis project.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.measure import label, regionprops_table
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
from skimage.transform import resize


DATASET_REPO = "https://github.com/Nickolay-K/Assingnment-3-dataset.git"
TARGET_SIZE = (256, 256)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for the reproducibility of the experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dataset(
    data_dir: str | Path = "/content/nuclei_dataset",
    repo_dir: str | Path = "/content/repo_clone",
    repo_url: str = DATASET_REPO,
) -> Path:
    """Download the data if it has not been extracted yet."""
    data_dir = Path(data_dir)
    repo_dir = Path(repo_dir)

    if data_dir.exists() and (data_dir / "metadata.csv").exists():
        return data_dir

    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "-q", repo_url, str(repo_dir)],
            check=True,
        )

    zip_path = repo_dir / "nuclei_dataset.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset archive not detected at {zip_path}")

    subprocess.run(
        ["unzip", "-q", str(zip_path), "-d", "/content"],
        check=True,
    )

    if not (data_dir / "metadata.csv").exists():
        raise FileNotFoundError(
            f"Extracting complete, but no data directory {data_dir} found."
        )

    return data_dir


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Load the metadata table with information about each image."""
    return pd.read_csv(Path(data_dir) / "metadata.csv")


def preprocess_image(
    path: str | Path,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Load an RGB image, convert it to grayscale, then rescale the values to [0-1] range."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Apply standard luminance coefficients. Masks are computed individually for each image.
    gray = (
        0.2989 * arr[..., 0]
        + 0.5870 * arr[..., 1]
        + 0.1140 * arr[..., 2]
    )

    if gray.shape != target_size:
        gray = resize(
            gray,
            target_size,
            anti_aliasing=True,
            preserve_range=True,
        )

    return gray.astype(np.float32)


def load_mask(
    path: str | Path,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Load a binary mask and resize it with nearest-neighbour interpolation to maintain labels"""
    mask = np.asarray(Image.open(path).convert("L"))

    if mask.shape != target_size:
        mask = resize(
            mask,
            target_size,
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        )

    return (mask > 127).astype(np.uint8)


def check_split_integrity(
    data_dir: str | Path,
    splits: Iterable[str] = ("train", "val", "test"),
) -> pd.DataFrame:
    """Check that filenames match and that outputs have expected dimensions."""
    rows = []

    for split in splits:
        image_dir = Path(data_dir) / split / "images"
        mask_dir = Path(data_dir) / split / "masks"

        image_names = set(os.listdir(image_dir))
        mask_names = set(os.listdir(mask_dir))

        sample_name = sorted(image_names)[0]
        image = preprocess_image(image_dir / sample_name)
        mask = load_mask(mask_dir / sample_name)

        rows.append(
            {
                "split": split,
                "n_images": len(image_names),
                "n_masks": len(mask_names),
                "filenames_match": image_names == mask_names,
                "image_shape": image.shape,
                "mask_shape": mask.shape,
                "mask_values": tuple(np.unique(mask).tolist()),
            }
        )

    return pd.DataFrame(rows)


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculate binary Dice coefficients for overlap assessment."""
    pred = pred_mask.astype(bool)
    truth = gt_mask.astype(bool)
    intersection = np.logical_and(pred, truth).sum()
    return float(
        2 * intersection / (pred.sum() + truth.sum() + 1e-8)
    )


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculate binary intersection-over-union metrics for 
    overlap assessment."""
    pred = pred_mask.astype(bool)
    truth = gt_mask.astype(bool)
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return float(intersection / (union + 1e-8))


def otsu_segment(
    image_path: str | Path,
    min_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the classic instance segmentation pipeline from the 
    notebook, which utilised Otsu’s thresholding and morphological 
    operations."""
    from skimage.filters import threshold_otsu

    gray = preprocess_image(image_path)
    mask = gray > threshold_otsu(gray)
    mask = binary_opening(mask, disk(1))
    mask = remove_small_objects(mask, min_size=min_size)
    mask = binary_closing(mask, disk(1))
    labelled = label(mask)
    return gray, mask, int(labelled.max())


def compute_region_summary(
    gray: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Summarise connected components in a predicted binary mask."""
    labelled = label(mask)
    n_objects = int(labelled.max())

    if n_objects == 0:
        return {
            "n_objects": 0,
            "mean_area": 0.0,
            "std_area": 0.0,
            "density_fraction": 0.0,
        }

    props = regionprops_table(
        labelled,
        intensity_image=gray,
        properties=["area", "eccentricity", "solidity", "mean_intensity"],
    )
    props_df = pd.DataFrame(props)

    return {
        "n_objects": n_objects,
        "mean_area": float(props_df["area"].mean()),
        "std_area": (
            float(props_df["area"].std()) if n_objects > 1 else 0.0
        ),
        "mean_intensity": float(props_df["mean_intensity"].mean()),
        "density_fraction": float(mask.sum() / mask.size),
    }


def extract_and_validate_json(
    response_text: str,
    required_fields: list[str],
) -> tuple[dict | None, str]:
    """Parse a JSON response and report missing required fields."""
    cleaned = re.sub(r"```json\s*|\s*```", "", response_text).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse failed: {exc}"

    missing = [field for field in required_fields if field not in parsed]
    if missing:
        return parsed, f"Valid JSON but missing fields: {missing}"

    return parsed, "Valid, all required fields present"


def fit_density_classifier(metadata: pd.DataFrame) -> dict[str, float]:
    """Create deterministic count-based density centres from training metadata.

    The original notebook used hand-written count thresholds. Here we derive
    one representative object-count centre per density regime from the
    training split only, then classify a prediction by nearest centre.
    """
    train = metadata[metadata["split"] == "train"].copy()
    centres = train.groupby("density")["n_objects"].median().to_dict()

    expected = {"sparse", "normal", "dense", "clustered"}
    missing = expected - set(centres)
    if missing:
        raise ValueError(f"Training metadata is missing density classes: {missing}")

    return {key: float(value) for key, value in centres.items()}


def classify_density(n_objects: int, centres: dict[str, float]) -> str:
    """Assign the nearest training-derived density regime."""
    return min(centres, key=lambda name: abs(n_objects - centres[name]))


class DoubleConv(torch.nn.Module):
    """Two Conv2D layers with batch normalization and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(torch.nn.Module):
    """A reduced U-Net architecture with three levels used in this project."""
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = torch.nn.MaxPool2d(2)

        self.up3 = torch.nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, 2, stride=2
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = torch.nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 2, stride=2
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = torch.nn.ConvTranspose2d(
            base_channels * 2, base_channels, 2, stride=2
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out_conv = torch.nn.Conv2d(
            base_channels, out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.dec3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return self.out_conv(dec1)
