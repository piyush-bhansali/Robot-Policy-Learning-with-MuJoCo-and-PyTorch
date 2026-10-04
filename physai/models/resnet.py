import timm
import torch
import torch.nn.functional as F
from torch import nn


class ResNet18Regressor(nn.Module):
    """Models C and D: ImageNet-pretrained ResNet18 backbone + MLP head -> (x, y).

    C (linear probe): freeze_backbone=True for the whole run, only the head learns.
    D (fine-tune):    start frozen, then train.py calls unfreeze_backbone() and gives
                      the backbone a smaller learning rate via param_groups().
                      While fine-tuning, backbone BatchNorm stays on its ImageNet running stats,
                      so training and evaluation normalise the same way.

    Takes images in [0, 1]; ImageNet mean/std normalisation happens inside forward.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True,
                 hidden: int = 128, input_size: int | None = None):
        super().__init__()
        # num_classes=0 removes ImageNet's 1000-class layer: output is the 512 pooled features.
        self.backbone = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden),  # 512 -> 128
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),                           # (x, y), normalised
        )
        self.input_size = input_size  # e.g. 224 to upsample; None keeps 96

        # The mean/std the backbone saw during ImageNet training, shaped to broadcast over (B, 3, H, W).
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

        self.backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Stop the backbone from learning: no gradients, BatchNorm statistics fixed."""
        self.backbone_frozen = True
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Let the backbone learn again (model D, after the head-only warm-up)."""
        self.backbone_frozen = False
        self.backbone.requires_grad_(True)
        self.train(self.training)  # through our train() below, so the BN rule applies

    def train(self, mode: bool = True) -> "ResNet18Regressor":
        # model.train() would normally put every layer in train mode. A frozen backbone
        # must stay in eval mode, or its BatchNorm layers keep changing their statistics.
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        else:
            # Fine-tuning: only the BN layers go to eval mode. They normalise with the stored running
            # stats and never update them, so training and evaluation match. Conv weights, γ and β still learn.
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def param_groups(self, head_lr: float, backbone_lr: float) -> list[dict]:
        """Two optimizer groups so the backbone can learn more slowly than the head."""
        return [
            {"params": self.head.parameters(), "lr": head_lr},
            {"params": self.backbone.parameters(), "lr": backbone_lr},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size is not None:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.head(self.backbone(x))
