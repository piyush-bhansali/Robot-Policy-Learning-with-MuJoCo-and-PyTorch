"""Evaluate VAE checkpoints and save reconstruction and generation figures.

From the repository root, after training:
    uv run python projects/vae/evaluate.py
    uv run python projects/vae/evaluate.py 'checkpoints=[outputs/<run>/best.pt]'

Metrics use sampled latents; reconstruction figures use latent means.
"""
import math
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from physai.datasets.reacher_vae_torch import ReacherVAEDataset
from physai.models.vae import ReacherVAE, vae_loss


def load_model(path: str, device: torch.device) -> tuple[ReacherVAE, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["cfg"]
    if cfg["model"]["name"] != "vae":
        raise ValueError(f"checkpoint is not a VAE: {path}")
    if cfg["loss"] != {"reconstruction": "mse", "reduction": "sum_per_image_mean_batch"}:
        raise ValueError(f"unsupported checkpoint loss settings: {path}")
    model = ReacherVAE(latent_dim=cfg["model"]["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def compute_metrics(model, loader, beta: float, accelerator: Accelerator) -> dict[str, float]:
    totals = {name: 0.0 for name in ("loss", "reconstruction_loss", "kl_loss")}
    count = 0
    for images in loader:
        reconstruction, mu, logvar = model(images)
        reconstruction, images, mu, logvar = accelerator.gather_for_metrics(
            (reconstruction, images, mu, logvar)
        )
        losses = vae_loss(reconstruction, images, mu, logvar, beta)
        for name in totals:
            totals[name] += losses[name].item() * len(images)
        count += len(images)
    metrics = {name: total / count for name, total in totals.items()}
    metrics["pixel_mse"] = metrics["reconstruction_loss"] / (3 * 96 * 96)
    return metrics


def plot_reconstructions(images, reconstructions, indices, title: str, path: Path) -> None:
    fig, axes = plt.subplots(2, len(images), figsize=(2 * len(images), 4), squeeze=False)
    for column, index in enumerate(indices):
        for row, batch in enumerate((images, reconstructions)):
            axes[row, column].imshow(batch[column].permute(1, 2, 0).numpy())
            axes[row, column].axis("off")
        axes[0, column].set_title(f"image {index}", fontsize=9)
    fig.suptitle(f"{title}\nTop: originals; bottom: reconstructions from latent means")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_generated(images, title: str, path: Path) -> None:
    columns = min(4, len(images))
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image in zip(axes.flat, images):
        ax.imshow(image.permute(1, 2, 0).numpy())
    fig.suptitle(f"{title}\nGenerated from standard normal latent vectors")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    if not cfg.checkpoints:
        raise ValueError("provide at least one checkpoint")
    if cfg.num_examples < 1 or cfg.num_generated < 1:
        raise ValueError("num_examples and num_generated must be positive")
    accelerator = Accelerator(mixed_precision="no")
    accelerator.print(OmegaConf.to_yaml(cfg))
    dataset = ReacherVAEDataset(to_absolute_path(cfg.data_path), cfg.split)
    if len(dataset) == 0:
        raise ValueError("evaluation split must be nonempty")
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    loader = accelerator.prepare(loader)
    out_dir = Path(to_absolute_path(cfg.out_dir))
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the same example images for every beta case.
    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:cfg.num_examples].tolist()
    examples = torch.stack([dataset[i] for i in indices])
    table = [
        "| Checkpoint | β | Total loss | Reconstruction loss | Pixel MSE | KL loss |",
        "|---|---|---|---|---|---|",
    ]
    for index, checkpoint_path in enumerate(cfg.checkpoints):
        model, train_cfg = load_model(to_absolute_path(checkpoint_path), accelerator.device)
        beta = train_cfg["beta"]
        # Reset sampling noise before each model for repeatable comparisons.
        set_seed(cfg.seed)
        metrics = compute_metrics(model, loader, beta, accelerator)
        accelerator.print(f"{checkpoint_path}: beta={beta}, {metrics}")
        table.append(
            f"| {checkpoint_path} | {beta} | {metrics['loss']:.4f} | "
            f"{metrics['reconstruction_loss']:.4f} | {metrics['pixel_mse']:.6f} | {metrics['kl_loss']:.4f} |"
        )
        if accelerator.is_main_process:
            with torch.no_grad():
                reconstructions, _, _ = model(examples.to(accelerator.device), sample=False)
                # Identical prior samples across models with the same latent size.
                prior_generator = torch.Generator().manual_seed(cfg.seed)
                z = torch.randn(cfg.num_generated, model.latent_dim, generator=prior_generator)
                generated = model.decode(z.to(accelerator.device))
            tag = f"{index}_beta{beta}"
            title = f"VAE β = {beta} ({cfg.split})"
            plot_reconstructions(
                examples, reconstructions.cpu(), indices, title, out_dir / f"{tag}_reconstructions.png"
            )
            plot_generated(generated.cpu(), f"VAE β = {beta}", out_dir / f"{tag}_generated.png")

    if accelerator.is_main_process:
        notes = (
            f"\nSplit: {cfg.split}, {len(dataset)} images. Seed: {cfg.seed}.\n\n"
            "Metrics use sampled latents; reconstruction figures use latent means.\n\n"
            "Reconstruction loss sums squared RGB pixel errors per image, then averages over images. "
            "Pixel MSE divides that loss by 27,648. KL is summed over latent dimensions, then averaged over images.\n\n"
            "Total loss includes each model's β, so compare reconstruction and KL separately across β cases.\n"
        )
        (out_dir / "results.md").write_text("\n".join(table) + "\n" + notes)
        accelerator.print(f"figures and results saved to {out_dir}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
