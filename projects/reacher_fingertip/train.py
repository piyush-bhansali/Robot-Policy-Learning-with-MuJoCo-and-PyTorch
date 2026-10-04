from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from physai.datasets.reacher_torch import TARGET_SCALE, ReacherDataset
from physai.models import build_model


def get_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    train_set = ReacherDataset(cfg.data_path, "train", augment=cfg.augment)
    val_set = ReacherDataset(cfg.data_path, "val")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader


def make_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    if hasattr(model, "param_groups"):
        params = model.param_groups(cfg.model.lr, cfg.model.lr * cfg.model.backbone_lr_mult)
    else:
        params = model.parameters()
    return torch.optim.AdamW(params, lr=cfg.model.lr, weight_decay=cfg.weight_decay)


def make_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss(beta=0.05)
    raise ValueError(f"unknown loss: {name}")


@torch.no_grad()
def evaluate(model, loader, loss_fn, accelerator) -> dict[str, float]:
    """Val loss (normalised units) and Euclidean fingertip error in cm."""
    model.eval()
    preds, targets = [], []
    for images, target in loader:
        pred = model(images)
        pred, target = accelerator.gather_for_metrics((pred, target))
        preds.append(pred)
        targets.append(target)
    preds, targets = torch.cat(preds), torch.cat(targets)

    err_cm = ((preds - targets) * TARGET_SCALE).norm(dim=1) * 100  # undo normalisation, m -> cm
    return {
        "loss": loss_fn(preds, targets).item(),
        "err_mean_cm": err_cm.mean().item(),
        "err_median_cm": err_cm.median().item(),
        "err_p95_cm": err_cm.quantile(0.95).item(),
    }


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, log_with="wandb")
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs={"wandb": {"name": cfg.wandb.name}},
    )
    accelerator.print(OmegaConf.to_yaml(cfg))
    out_dir = Path(HydraConfig.get().runtime.output_dir)  # outputs/<date>/<time>/

    train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(cfg.model)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.model.epochs)
    loss_fn = make_loss(cfg.loss)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    unfreeze_epoch = cfg.model.get("unfreeze_epoch")  # None for A, B, C
    best_err = float("inf")
    step = 0
    for epoch in range(cfg.model.epochs):
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            accelerator.unwrap_model(model).unfreeze_backbone()
            accelerator.print(f"epoch {epoch}: backbone unfrozen")

        # ---- train ----
        model.train()
        loss_sum, count = 0.0, 0
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), targets)
            accelerator.backward(loss)
            optimizer.step()

            loss_sum += loss.item() * len(targets)
            count += len(targets)
            if step % cfg.log_every == 0:
                accelerator.log({"train/loss": loss.item(), "epoch": epoch}, step=step)
            step += 1
        scheduler.step()

        # ---- validate ----
        val = evaluate(model, val_loader, loss_fn, accelerator)
        accelerator.log(
            {"train/epoch_loss": loss_sum / count, "lr": optimizer.param_groups[0]["lr"]}
            | {f"val/{k}": v for k, v in val.items()},
            step=step,
        )
        accelerator.print(
            f"epoch {epoch:2d}  train loss {loss_sum / count:.4f}  val loss {val['loss']:.4f}  "
            f"val err {val['err_mean_cm']:.2f} cm (median {val['err_median_cm']:.2f}, p95 {val['err_p95_cm']:.2f})"
        )

        # ---- keep the best checkpoint (by val mean error) and the latest one ----
        if accelerator.is_main_process:
            checkpoint = {"cfg": OmegaConf.to_container(cfg, resolve=True),
                          "state_dict": accelerator.unwrap_model(model).state_dict(),
                          "epoch": epoch, "val_err_mean_cm": val["err_mean_cm"]}
            if val["err_mean_cm"] < best_err:
                best_err = val["err_mean_cm"]
                torch.save(checkpoint, out_dir / "best.pt")
            torch.save(checkpoint, out_dir / "last.pt")  # overwritten every epoch: the final model

    accelerator.print(f"best val err {best_err:.2f} cm, checkpoint: {out_dir / 'best.pt'}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
