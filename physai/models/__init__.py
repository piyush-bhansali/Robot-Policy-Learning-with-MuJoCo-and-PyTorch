from torch import nn

from physai.models.cnn import CNNGap, CNNSsm
from physai.models.resnet import ResNet18Regressor


def build_model(cfg) -> nn.Module:
    """Create model A, B, C or D from its Hydra config (configs/model/*.yaml)."""
    if cfg.name == "cnn_gap":                                   # A
        return CNNGap(out_ch=cfg.out_ch, hidden=cfg.hidden)
    if cfg.name == "cnn_ssm":                                   # B
        return CNNSsm(out_ch=cfg.out_ch, hidden=cfg.hidden)
    if cfg.name in ("resnet18_probe", "resnet18_ft"):           # C, D
        return ResNet18Regressor(
            pretrained=cfg.pretrained,
            freeze_backbone=True,  # D also starts frozen; train.py unfreezes it later
            hidden=cfg.hidden,
            input_size=cfg.input_size,
        )
    raise ValueError(f"unknown model name: {cfg.name}")
