import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from physai.datasets.reacher import load_reacher_npz

# Fingertip reach is 0.10 + 0.11 m, so dividing by this puts x and y in [-1, 1].
TARGET_SCALE = 0.21


def label_safe_augment() -> v2.Compose:
    return v2.Compose([
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        v2.GaussianNoise(mean=0.0, sigma=0.02),  # clips back to [0, 1] by default
    ])


class ReacherDataset(Dataset):
    """One split ("train", "val" or "test") of a saved Reacher .npz, ready for a DataLoader."""

    def __init__(self, npz_path: str, split: str, augment: bool = False):
        data = load_reacher_npz(npz_path)
        # (N, H, W, 3) uint8 -> (N, 3, H, W) uint8. Kept as uint8 (4x smaller) until __getitem__.
        self.images = torch.from_numpy(data[f"{split}_images"]).permute(0, 3, 1, 2).contiguous()
        self.targets = torch.from_numpy(data[f"{split}_fingertip"]) / TARGET_SCALE
        self.transform = label_safe_augment() if augment else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[i].float() / 255.0
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[i]
