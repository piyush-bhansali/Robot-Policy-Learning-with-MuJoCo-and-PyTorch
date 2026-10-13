
import shutil
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from physai.datasets.reacher_vae_torch import ReacherVAEDataset
from physai.models.vae import ReacherVAE, vae_loss


METRIC_NAMES = ("loss", "reconstruction_loss", "kl_loss")


def get_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    data_path = to_absolute_path(cfg.data_path)
    train_set = ReacherVAEDataset(data_path, "train")
    val_set = ReacherVAEDataset(data_path, "val")
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError("training and validation splits must be nonempty")
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, beta: float, accelerator: Accelerator) -> dict[str, float]:
    
    model.eval()
    totals = {name: 0.0 for name in METRIC_NAMES}
    count = 0
    for images in loader:
        reconstruction, mu, logvar = model(images)
        reconstruction, images, mu, logvar = accelerator.gather_for_metrics(
            (reconstruction, images, mu, logvar)
        )
        losses = vae_loss(reconstruction.float(), images.float(), mu.float(), logvar.float(), beta)
        for name in METRIC_NAMES:
            totals[name] += losses[name].item() * len(images)
        count += len(images)
    return {name: total / count for name, total in totals.items()}


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.model.name != "vae":
        raise ValueError(f"unsupported model: {cfg.model.name}")
    if cfg.loss.reconstruction != "mse" or cfg.loss.reduction != "sum_per_image_mean_batch":
        raise ValueError("supported loss is MSE with sum_per_image_mean_batch reduction")
    if not 0 <= cfg.beta < float("inf"):
        raise ValueError("beta must be finite and nonnegative")
    if cfg.epochs < 1 or cfg.log_every < 1:
        raise ValueError("epochs and log_every must be positive")

    set_seed(cfg.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, log_with=cfg.tracker)
    if cfg.tracker:
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    accelerator.print(OmegaConf.to_yaml(cfg))
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    train_loader, val_loader = get_dataloaders(cfg)
    model = ReacherVAE(latent_dim=cfg.model.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_val, step = float("inf"), 0
    for epoch in range(cfg.epochs):
        model.train()
        # Three weighted loss sums and the number of training images.
        totals = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
        for images in train_loader:
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(images)
            losses = vae_loss(reconstruction.float(), images.float(), mu.float(), logvar.float(), cfg.beta)
            accelerator.backward(losses["loss"])
            optimizer.step()

            batch_metrics = torch.stack([losses[name].detach() for name in METRIC_NAMES])
            totals[:3] += batch_metrics.double() * len(images)
            totals[3] += len(images)
            if step % cfg.log_every == 0:
                metrics = accelerator.reduce(batch_metrics, reduction="mean")
                accelerator.log(
                    {f"train/{name}": value.item() for name, value in zip(METRIC_NAMES, metrics)}
                    | {"epoch": epoch},
                    step=step,
                )
            step += 1

        totals = accelerator.reduce(totals, reduction="sum")
        train = {name: (totals[i] / totals[3]).item() for i, name in enumerate(METRIC_NAMES)}

        # Use repeatable validation noise without changing the training RNG state.
        devices = [accelerator.device] if accelerator.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg.seed + 1)
            val = evaluate(model, val_loader, cfg.beta, accelerator)

        accelerator.log(
            {f"train/epoch_{name}": value for name, value in train.items()}
            | {f"val/{name}": value for name, value in val.items()}
            | {"epoch": epoch, "beta": cfg.beta, "lr": optimizer.param_groups[0]["lr"]},
            step=step,
        )
        accelerator.print(
            f"epoch {epoch:2d}  train {train['loss']:.4f}  val {val['loss']:.4f}  "
            f"val reconstruction {val['reconstruction_loss']:.4f}  val KL {val['kl_loss']:.4f}"
        )

        improved = val["loss"] < best_val
        if improved:
            best_val = val["loss"]
        if accelerator.is_main_process:
            checkpoint = {
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "state_dict": accelerator.unwrap_model(model).state_dict(),
                "epoch": epoch,
                "val": val,
            }
            if improved:
                torch.save(checkpoint, out_dir / "best.pt")
            torch.save(checkpoint, out_dir / "last.pt")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir = Path(to_absolute_path(cfg.checkpoint_dir))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stable_path = checkpoint_dir / f"{cfg.run_name}.pt"
        shutil.copy(out_dir / "best.pt", stable_path)
        accelerator.print(f"best val loss {best_val:.4f}, checkpoint: {stable_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
