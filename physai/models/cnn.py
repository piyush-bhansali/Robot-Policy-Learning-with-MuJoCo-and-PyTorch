import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU, the basic CNN building block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):

    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, 32, kernel=5, stride=2),   # 96 -> 48
            conv_block(32, 32),                      # 48
            conv_block(32, 64, stride=2),            # 48 -> 24
            conv_block(64, 64),                      # 24
            conv_block(64, out_ch),                  # 24
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNGap(nn.Module):
    """Model A: SmallCNN + global average pooling + MLP head -> (x, y)."""

    def __init__(self, out_ch: int = 64, hidden: int = 128):
        super().__init__()
        self.backbone = SmallCNN(out_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 24, 24) -> (B, C, 1, 1): one number per channel
        self.head = nn.Sequential(
            nn.Flatten(),                    # (B, C, 1, 1) -> (B, C)
            nn.Linear(out_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),            # (x, y), normalised
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone(x)))


class SpatialSoftmax(nn.Module):
    """Turns each feature map into one 2D keypoint: the expected (x, y) of where it fires.

    (B, C, H, W) feature maps -> (B, C, 2) keypoints, coordinates in [-1, 1]
    (x: -1 = left edge, +1 = right edge; y: -1 = top row, +1 = bottom row).
    From Levine et al. 2016, "End-to-End Training of Deep Visuomotor Policies".
    """

    def __init__(self):
        super().__init__()
        # Learned sharpness. Stored as a log so the scale exp(log_scale) is always positive.
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        logits = features.flatten(2) * self.log_scale.exp()       # (B, C, H*W)
        probs = logits.softmax(dim=-1).view(B, C, H, W)           # each map now sums to 1

        xs = torch.linspace(-1, 1, W, device=features.device)     # column coordinates
        ys = torch.linspace(-1, 1, H, device=features.device)     # row coordinates
        x = (probs.sum(dim=2) * xs).sum(dim=-1)                   # sum over rows -> P(column), then mean
        y = (probs.sum(dim=3) * ys).sum(dim=-1)                   # sum over cols -> P(row), then mean
        return torch.stack([x, y], dim=-1)                        # (B, C, 2)


class CNNSsm(nn.Module):
    """Model B: SmallCNN + spatial softmax + MLP head -> (x, y)."""

    def __init__(self, out_ch: int = 64, hidden: int = 128):
        super().__init__()
        self.backbone = SmallCNN(out_ch)
        self.pool = SpatialSoftmax()          # (B, C, 24, 24) -> (B, C, 2): one keypoint per channel
        self.head = nn.Sequential(
            nn.Flatten(),                     # (B, C, 2) -> (B, 2C)
            nn.Linear(2 * out_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),             # (x, y), normalised
        )

    def keypoints(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, 2) keypoints in [-1, 1], for plotting them on the image."""
        return self.pool(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.keypoints(x))
