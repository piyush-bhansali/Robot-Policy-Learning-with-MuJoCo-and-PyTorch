"""Image-only Reacher dataset for VAE reconstruction training."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ReacherVAEDataset(Dataset):

    def __init__(self, npz_path: str | Path, split: str):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        # Read only the requested images; preserve the split stored in the file.
        with np.load(npz_path) as data:
            images = data[f"{split}_images"]

        if images.ndim != 4 or images.shape[1:] != (96, 96, 3):
            raise ValueError(f"expected images of shape (N, 96, 96, 3), got {images.shape}")
        if images.dtype != np.uint8:
            raise ValueError(f"expected uint8 images, got {images.dtype}")

        # Keep uint8 in memory; convert to float only when an image is requested.
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.images[i].float() / 255.0
