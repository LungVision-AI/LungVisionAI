"""PyTorch dataset and image transforms."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_image(path_or_array, clahe: bool = True) -> np.ndarray:
    """Read an image as grayscale, apply CLAHE, return HxWx3 uint8 (models expect 3 channels)."""
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.imread(str(path_or_array), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if clahe:
        img = _clahe.apply(img)
    return np.repeat(img[..., None], 3, axis=2)


def build_transforms(image_size: int, train: bool, augment: bool = True) -> A.Compose:
    ops = [A.Resize(image_size, image_size)]
    if train and augment:
        # No horizontal flips: the heart sits on the left, and flipping would
        # teach the model anatomy that doesn't exist.
        ops += [
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(p=0.15),
        ]
    ops += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    return A.Compose(ops)


class LungDataset(Dataset):
    def __init__(self, manifest: str | pd.DataFrame, labels: list[str], split: str | None,
                 image_size: int, train: bool = False, clahe: bool = True, augment: bool = True):
        df = pd.read_csv(manifest) if isinstance(manifest, str) else manifest
        if split is not None:
            df = df[df["split"] == split]
        if len(df) == 0:
            raise ValueError(f"No rows for split '{split}'")
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.y = self.df[labels].to_numpy(dtype=np.float32)
        self.clahe = clahe
        self.tf = build_transforms(image_size, train, augment)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img = load_image(self.df.at[i, "path"], self.clahe)
        x = self.tf(image=img)["image"]
        y = self.y[i]
        mask = (y >= 0).astype(np.float32)
        return x, torch.from_numpy(np.clip(y, 0, 1)), torch.from_numpy(mask)

    def pos_weight(self) -> torch.Tensor:
        """neg/pos ratio per label over known labels, capped so rare classes don't explode."""
        pos = (self.y == 1).sum(0)
        neg = (self.y == 0).sum(0)
        w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
        return torch.tensor(np.clip(w, 1.0, 50.0), dtype=torch.float32)
