"""Train a Reacher dynamics model.

    uv run python projects/reacher_dynamics/train.py model=mlp obs=full     # M1
    uv run python projects/reacher_dynamics/train.py model=mlp obs=angles   # M2
    uv run python projects/reacher_dynamics/train.py model=gru obs=angles   # G2

Shape letters: B = batch (256 transitions for the MLP, 32 episodes for the GRU), T = 100 steps per episode,
n_out = target columns (4 for obs=full, 2 for obs=angles).
"""
import shutil
import time
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from physai.datasets.reacher_dynamics import DT, load_dynamics_npz
from physai.datasets.reacher_dynamics_torch import (
    INPUT_NAMES,
    TARGET_NAMES,
    SequenceDataset,
    TransitionDataset,
    compute_norm_stats,
    load_split,
)
from physai.models.dynamics import build_model


def get_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """MLP: shuffled single transitions. GRU: shuffled whole episodes, time order kept inside each."""
    dataset = SequenceDataset if cfg.model.name == "gru" else TransitionDataset
    train_set = dataset(cfg.data_path, "train", cfg.obs.name)
    val_set = dataset(cfg.data_path, "val", cfg.obs.name)
    train_loader = DataLoader(train_set, batch_size=cfg.model.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.model.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader


def predict(model, raw_model, x, y, cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised predictions and normalised targets for one batch of raw x, y.

    model is the Accelerate-prepared module (used for forward); raw_model is the same module unwrapped,
    for the normalisation helpers.
    """
    x_n, y_n = raw_model.normalise_x(x), raw_model.normalise_y(y)
    if cfg.model.name == "gru":
        pred, _ = model(x_n)  # (B, T, n_out), teacher forcing: true inputs at every step
        # The memory is empty at t = 0, so the first `warmup` predictions are not judged.
        return pred[:, cfg.warmup:], y_n[:, cfg.warmup:]
    return model(x_n), y_n  # (B, n_out)


@torch.no_grad()
def evaluate(model, raw_model, loader, cfg: DictConfig, accelerator: Accelerator) -> dict[str, float]:
    """Val loss (normalised MSE, what early stopping watches) and per-target RMSE in raw units."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        pred, target = predict(model, raw_model, x, y, cfg)
        pred, target = accelerator.gather_for_metrics((pred, target))
        preds.append(pred.reshape(-1, pred.shape[-1]))
        targets.append(target.reshape(-1, target.shape[-1]))
    pred, target = torch.cat(preds), torch.cat(targets)

    rmse = ((pred - target) * raw_model.y_std).pow(2).mean(0).sqrt()  # x std undoes the normalisation
    names = TARGET_NAMES[cfg.obs.name]
    return {"loss": F.mse_loss(pred, target).item()} | {f"rmse_{n}": v.item() for n, v in zip(names, rmse)}


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    accelerator = Accelerator(log_with=cfg.tracker)
    if cfg.tracker:
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    accelerator.print(OmegaConf.to_yaml(cfg))
    out_dir = Path(HydraConfig.get().runtime.output_dir)  # outputs/<date>/<time>/

    # Normalisation statistics from the TRAIN split only; the model keeps them in its buffers.
    stats = compute_norm_stats(*load_split(cfg.data_path, "train", cfg.obs.name))
    model = build_model(cfg.model.name, cfg.obs.name, stats, cfg.model.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader, val_loader = get_dataloaders(cfg)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    raw_model = accelerator.unwrap_model(model)

    # Everything a later script needs to rebuild and trust this model (checked in load_checkpoint).
    _, data_meta = load_dynamics_npz(cfg.data_path)
    meta = {
        "kind": cfg.model.name,
        "obs": cfg.obs.name,
        "hidden": cfg.model.hidden,
        "input_names": INPUT_NAMES[cfg.obs.name],
        "target_names": TARGET_NAMES[cfg.obs.name],
        "dt": DT,
        "dataset_version": data_meta["version"],
        "dataset_seed": data_meta["seed"],
        "warmup": cfg.warmup,
        "seed": cfg.seed,
    }

    best_val, bad_epochs, step = float("inf"), 0, 0
    best_path = out_dir / "best.pt"
    for epoch in range(cfg.model.max_epochs):
        start = time.time()

        # ---- train ----
        model.train()
        loss_sum, count = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred, target = predict(model, raw_model, x, y, cfg)
            loss = F.mse_loss(pred, target)
            accelerator.backward(loss)
            if cfg.model.grad_clip is not None:
                accelerator.clip_grad_norm_(model.parameters(), cfg.model.grad_clip)
            optimizer.step()

            loss_sum += loss.item() * len(x)
            count += len(x)
            if step % cfg.log_every == 0:
                accelerator.log({"train/loss": loss.item(), "epoch": epoch}, step=step)
            step += 1

        # ---- validate ----
        val = evaluate(model, raw_model, val_loader, cfg, accelerator)
        accelerator.log({"train/epoch_loss": loss_sum / count} | {f"val/{k}": v for k, v in val.items()}, step=step)
        rmse = "  ".join(f"{k[5:]} {v:.4f}" for k, v in val.items() if k.startswith("rmse_"))
        accelerator.print(
            f"epoch {epoch:3d}  train {loss_sum / count:.4f}  val {val['loss']:.4f}  "
            f"val rmse: {rmse}  ({time.time() - start:.1f}s)"
        )

        # ---- early stopping: keep the best model by val loss ----
        if val["loss"] < best_val:
            best_val, bad_epochs = val["loss"], 0
            if accelerator.is_main_process:
                checkpoint = {
                    "state_dict": raw_model.state_dict(),  # weights + normalisation buffers
                    "meta": meta | {"epoch": epoch, "val_loss": best_val},
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                }
                torch.save(checkpoint, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                accelerator.print(f"early stop: no val improvement for {cfg.patience} epochs")
                break

    if accelerator.is_main_process:
        stable_path = Path(to_absolute_path(cfg.checkpoint_dir)) / f"{cfg.run_name}.pt"
        stable_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_path, stable_path)
        accelerator.print(f"best val loss {best_val:.4f}, checkpoint: {best_path} (copied to {stable_path})")
    accelerator.end_training()


if __name__ == "__main__":
    main()
