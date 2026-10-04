"""Evaluate trained checkpoints on the held-out test set.

Usage (from the repo root):
    uv run python projects/reacher_fingertip/evaluate.py outputs/<run A>/best.pt outputs/<run B>/best.pt ...

Writes results.md (the README table) and plots to figures/eval/, next to this file.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from physai.datasets.reacher import load_reacher_npz, world_to_pixel
from physai.datasets.reacher_torch import TARGET_SCALE, ReacherDataset
from physai.models import build_model

LABELS = {"cnn_gap": "A  CNN + GAP", "cnn_ssm": "B  CNN + spatial softmax",
          "resnet18_probe": "C  ResNet18 probe", "resnet18_ft": "D  ResNet18 fine-tune"}
TRUE_COLOR, PRED_COLOR = "lime", "red"
PROJECT_DIR = Path(__file__).parent


# ---------------------------------------------------------------- loading + metrics

def load_model(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the architecture from the saved config, then load the trained weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_cfg = OmegaConf.create(ckpt["cfg"]["model"])
    if "pretrained" in model_cfg:
        model_cfg.pretrained = False  # weights come from the checkpoint, no need to download
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["cfg"]


@torch.no_grad()
def predict(model, dataset, device) -> np.ndarray:
    """Predicted fingertip (x, y) in metres for every item, shape (N, 2)."""
    preds = [model(images.to(device)).cpu() for images, _ in DataLoader(dataset, batch_size=128)]
    return torch.cat(preds).numpy() * TARGET_SCALE


@torch.no_grad()
def ms_per_image(model, dataset, device, n: int = 100) -> float:
    """Average latency for ONE image at a time (batch size 1), like a robot's control loop."""
    images = [dataset[i][0].unsqueeze(0).to(device) for i in range(n)]
    for img in images[:10]:  # warm-up: first calls include one-off setup costs
        model(img)
    if device.type == "cuda":
        torch.cuda.synchronize()  # GPU runs asynchronously; wait before starting the clock
    start = time.perf_counter()
    for img in images:
        model(img)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n * 1000


# ---------------------------------------------------------------- plots

def plot_scatter(true_m, pred_m, title, path):
    """Predicted vs true, one panel for x and one for y. Perfect = on the diagonal."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, i, name in zip(axes, (0, 1), ("x", "y")):
        ax.plot([-21, 21], [-21, 21], color="0.6", linewidth=1, zorder=0)  # perfect prediction
        ax.scatter(true_m[:, i] * 100, pred_m[:, i] * 100, s=8, alpha=0.6)
        ax.set(xlabel=f"true {name} (cm)", ylabel=f"predicted {name} (cm)", aspect="equal",
               xlim=(-23, 23), ylim=(-23, 23))
        ax.grid(color="0.92")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_frames(images, true_m, pred_m, err_cm, idx, title, path, extra=None):
    """Test frames with true (green ring) and predicted (red cross) fingertip."""
    size = images.shape[1]
    cols = len(idx) if len(idx) <= 5 else 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idx):
        t, p = world_to_pixel(true_m[i], size), world_to_pixel(pred_m[i], size)
        ax.imshow(images[i])
        ax.scatter(*t, s=90, facecolors="none", edgecolors=TRUE_COLOR, linewidths=1.5)
        ax.scatter(*p, s=50, marker="x", color=PRED_COLOR, linewidths=1.5)
        ax.set_title(f"{err_cm[i]:.2f} cm" + (f"\n{extra[i]}" if extra is not None else ""), fontsize=8)
    fig.suptitle(f"{title}\ngreen ring = true, red x = predicted", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_keypoints(model, dataset, images, idx, path):
    """Model B only: the 64 spatial-softmax keypoints drawn on the image."""
    size = images.shape[1]
    batch = torch.stack([dataset[i][0] for i in idx]).to(next(model.parameters()).device)
    kp = model.keypoints(batch).cpu().numpy()  # (n, C, 2) in [-1, 1], x right, y down
    kp_px = (kp + 1) / 2 * (size - 1)          # -> pixel (col, row)
    fig, axes = plt.subplots(1, len(idx), figsize=(2.6 * len(idx), 2.9))
    for ax, img, pts in zip(axes, images[idx], kp_px):
        ax.imshow(img)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, c=np.arange(len(pts)), cmap="cool", edgecolors="black", linewidths=0.3)
        ax.axis("off")
    fig.suptitle("Model B: spatial-softmax keypoints (one dot per feature channel)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="best.pt files from train.py runs")
    parser.add_argument("--data", default="data/reacher_5000_s0.npz")
    parser.add_argument("--out", default=str(PROJECT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    fig_dir = out / "figures" / "eval"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = load_reacher_npz(args.data)
    images, true_m = raw["test_images"], raw["test_fingertip"]
    dot_dist_cm = np.linalg.norm(true_m - raw["test_target"], axis=1) * 100
    elbow = np.abs(raw["test_joints"][:, 1])      # 0 = arm straight, 3 = fully folded
    test_set = ReacherDataset(args.data, "test")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(images), size=8, replace=False)

    rows = []
    for ckpt_path in args.checkpoints:
        model, cfg = load_model(ckpt_path, device)
        name = cfg["model"]["name"]
        label = LABELS.get(name, name)

        pred_m = predict(model, test_set, device)
        err_cm = np.linalg.norm(pred_m - true_m, axis=1) * 100
        params = sum(p.numel() for p in model.parameters())
        ms = ms_per_image(model, test_set, device)
        rows.append((label, err_cm.mean(), np.median(err_cm), np.percentile(err_cm, 95), params, ms))

        # what do the worst 5% have in common?
        worst = err_cm >= np.percentile(err_cm, 95)
        print(f"\n{label}: mean {err_cm.mean():.2f} cm | worst 5% vs rest:"
              f"  fingertip-dot dist {dot_dist_cm[worst].mean():.1f} vs {dot_dist_cm[~worst].mean():.1f} cm,"
              f"  |elbow| {elbow[worst].mean():.2f} vs {elbow[~worst].mean():.2f} rad")

        worst5 = np.argsort(err_cm)[-5:][::-1]
        extra = [f"dot {d:.0f} cm, elbow {e:.1f}" for d, e in zip(dot_dist_cm, elbow)]
        plot_scatter(true_m, pred_m, f"{label}: predicted vs true (test)", fig_dir / f"{name}_scatter.png")
        plot_frames(images, true_m, pred_m, err_cm, sample_idx, f"{label}: random test frames",
                    fig_dir / f"{name}_samples.png")
        plot_frames(images, true_m, pred_m, err_cm, worst5, f"{label}: 5 worst test predictions",
                    fig_dir / f"{name}_worst5.png", extra=extra)
        if name == "cnn_ssm":
            plot_keypoints(model, test_set, images, sample_idx[:5], fig_dir / "cnn_ssm_keypoints.png")

    table = ["| Model | Mean err (cm) | Median | P95 | Params | ms/img |", "|---|---|---|---|---|---|"]
    table += [f"| {l} | {m:.2f} | {md:.2f} | {p95:.2f} | {n / 1e6:.2f} M | {ms:.2f} |" for l, m, md, p95, n, ms in rows]
    table_md = "\n".join(table) + f"\n\nTest set: {len(images)} frames. Timing: batch size 1 on {device.type}.\n"
    (out / "results.md").write_text(table_md)
    print("\n" + table_md)
    print(f"plots in {fig_dir}/")


if __name__ == "__main__":
    main()
