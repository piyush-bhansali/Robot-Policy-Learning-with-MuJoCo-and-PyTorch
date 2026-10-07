"""Evaluate the Reacher dynamics models on held-out episodes.

    uv run python projects/reacher_dynamics/eval_rollouts.py            # print tables, save figures
    uv run python projects/reacher_dynamics/eval_rollouts.py --wandb    # also log everything to W&B

Shape letters:
    n   episodes in a split (100)
    T   steps per episode (100 actions, 101 states)
    W   minimum warm-up (10): G2 needs at least steps 0..W-1 in memory, so every model is judged from step W on
    N   judged predictions, flattened over episodes and steps: n * (T - W) = 9000 for one-step
    H   rollout horizon: steps predicted open loop (50)
    S   rollout start steps per episode, STARTS = 10, 20, 30, 40, 50 (5); each rollout ends by step 100
    R   rollouts in a split: S * n = 500
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from physai.datasets.reacher_dynamics import (
    LINK0, LINK1, SERIES_COLORS, fingertip_xy, load_dynamics_npz, wrap_angle,
)
from physai.models.dynamics import ConstantVelocity, DynamicsModel, load_checkpoint

DATA_PATH = "data/reacher_dynamics_v1.npz"
CHECKPOINTS = {
    "M1": "checkpoints/mlp_full_s0.pt",
    "M2": "checkpoints/mlp_angles_s0.pt",
    "G2": "checkpoints/gru_angles_s0.pt",
}
W = 10
H = 50
STARTS = (10, 20, 30, 40, 50)   # every start >= W (at least W steps of history for G2) and start + H <= T (inside the episode)
REPORT_HORIZONS = (1, 5, 10, 25, 50)
FIG_DIR = Path("projects/reacher_dynamics/figures")
MODEL_COLORS = dict(zip(("B1", "M1", "M2", "G2"), SERIES_COLORS))  # fixed: a model keeps its colour in every plot
TRUTH_COLOR = "#222222"
SPLIT_TITLES = {"test": "test (same action types as training)", "ood": "ood (chirp actions, never seen in training)"}


def load_models() -> dict[str, DynamicsModel]:
    """B1 plus the three trained models, in eval mode on the CPU."""
    models = {"B1": ConstantVelocity()}
    for name, path in CHECKPOINTS.items():
        models[name], _ = load_checkpoint(path)
    return models


def load_episodes(split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, qd: (n, T+1, 2) raw angles / velocities; a: (n, T, 2) actions. float32 tensors."""
    data, _ = load_dynamics_npz(DATA_PATH)
    return tuple(torch.as_tensor(data[f"{split}_{k}"], dtype=torch.float32) for k in ("q", "qd", "a"))


@torch.no_grad()
def one_step_predictions(model: DynamicsModel, q, qd, a) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicts every step t -> t+1 from the TRUE state at t (teacher forcing).

    Walks through each episode in time order so G2 carries its memory h from step to step;
    for B1 / M1 / M2 the state is always None and the loop just predicts each step independently.

    Returns q_pred, qd_pred: (n, T, 2), row t = prediction of step t+1.
    """
    state = None
    q_pred, qd_pred = [], []
    for t in range(a.shape[1]):
        q_next, qd_next, state = model.step(q[:, t], qd[:, t], a[:, t], state)
        q_pred.append(q_next)
        qd_pred.append(qd_next)
    return torch.stack(q_pred, dim=1), torch.stack(qd_pred, dim=1)


def one_step_errors(model: DynamicsModel, q, qd, a) -> dict[str, float]:
    """Computes one-step prediction errors for a model on a split of episodes."""
    q_pred, qd_pred = one_step_predictions(model, q, qd, a)
    q_pred, qd_pred = q_pred[:, W:].numpy(), qd_pred[:, W:].numpy()  # (n, T-W, 2): drop the warm-up steps
    q_true, qd_true = q[:, W + 1:].numpy(), qd[:, W + 1:].numpy()     # truth at t+1 for t = W..T-1

    angle_err = wrap_angle(q_pred - q_true).reshape(-1, 2)                                   # (N, 2) rad
    tip_err = np.linalg.norm(fingertip_xy(q_pred) - fingertip_xy(q_true), axis=-1).ravel() * 100  # (N,) cm

    errors = {
        "angle_rmse_q1": np.sqrt((angle_err[:, 0] ** 2).mean()),
        "angle_rmse_q2": np.sqrt((angle_err[:, 1] ** 2).mean()),
        "tip_mean_cm": tip_err.mean(),
        "tip_p95_cm": np.percentile(tip_err, 95),
    }
    if model.obs == "full":  # angles-only models don't predict velocity
        vel_err = (qd_pred - qd_true).reshape(-1, 2)
        errors["vel_rmse_qd1"] = np.sqrt((vel_err[:, 0] ** 2).mean())
        errors["vel_rmse_qd2"] = np.sqrt((vel_err[:, 1] ** 2).mean())
    return {k: float(v) for k, v in errors.items()}


def print_one_step_table(split: str, results: dict[str, dict[str, float]]) -> None:
    print(f"\nOne-step error, split = {split}  (steps t >= {W})")
    print(f"{'model':6s} {'q1 rad':>8s} {'q2 rad':>8s} {'qd1 rad/s':>10s} {'qd2 rad/s':>10s} {'tip cm':>8s} {'tip p95':>8s}")
    for name, e in results.items():
        vel1 = f"{e['vel_rmse_qd1']:10.3f}" if "vel_rmse_qd1" in e else f"{'-':>10s}"
        vel2 = f"{e['vel_rmse_qd2']:10.3f}" if "vel_rmse_qd2" in e else f"{'-':>10s}"
        print(f"{name:6s} {e['angle_rmse_q1']:8.4f} {e['angle_rmse_q2']:8.4f} {vel1} {vel2} "
              f"{e['tip_mean_cm']:8.3f} {e['tip_p95_cm']:8.3f}")


# ---------- 4.2 open-loop rollouts ----------

@torch.no_grad()
def rollouts(model: DynamicsModel, q, qd, a) -> tuple[torch.Tensor, torch.Tensor]:
    """Open-loop rollouts from every start in STARTS, all episodes at once.

    For each start t0:
      1. warm-up: G2 builds its memory h from ALL true steps 0 .. t0-1 of the episode (others return None).
         h starts at zero at step 0, exactly as in training; a fresh h mid-episode is unlike anything G2 trained on;
      2. the model gets the TRUE state at t0, then its own prediction at every later step;
      3. actions are the true ones the episode applied at t0 .. t0+H-1.

    Returns q_roll, q_true: (R, H+1, 2), index 0 = the true start state, index h = h steps ahead.
    """
    q_roll, q_true = [], []
    for t0 in STARTS:
        past = slice(0, t0)  # whole history: the window grows with t0
        state = model.warmup(q[:, past], qd[:, past], a[:, past])
        q_pred, _ = model.rollout(q[:, t0], qd[:, t0], a[:, t0:t0 + H], state)  # (n, H+1, 2)
        q_roll.append(q_pred)
        q_true.append(q[:, t0:t0 + H + 1])
    return torch.cat(q_roll), torch.cat(q_true)


def rollout_errors(q_roll: torch.Tensor, q_true: torch.Tensor) -> dict[str, np.ndarray]:
    """Error at every horizon h = 1..H over all R rollouts, from the output of rollouts().

    Returns:
        angle: (R, H, 2) wrapped angle error |q_pred - q_true| per joint, rad
        tip:   (R, H) fingertip distance, cm
    """
    q_roll, q_true = q_roll[:, 1:].numpy(), q_true[:, 1:].numpy()  # drop h = 0: the true start, error 0
    angle = np.abs(wrap_angle(q_roll - q_true))
    tip = np.linalg.norm(fingertip_xy(q_roll) - fingertip_xy(q_true), axis=-1) * 100
    return {"angle": angle, "tip": tip}


def random_guess_cm(q_true: torch.Tensor) -> float:
    """Mean distance between the true fingertip and the fingertip of a DIFFERENT, random rollout at the same h.

    The error of a model that knows where the arm can be but not where it is: the "lost" level.
    """
    tip = fingertip_xy(q_true[:, 1:].numpy()) * 100                    # (R, H, 2) cm
    shuffled = tip[np.random.default_rng(0).permutation(len(tip))]
    return float(np.linalg.norm(tip - shuffled, axis=-1).mean())


def print_rollout_table(split: str, errors: dict[str, dict[str, np.ndarray]]) -> None:
    """Fingertip error (mean and 10-90 % range over rollouts) and mean angle error at REPORT_HORIZONS."""
    n_rollouts = next(iter(errors.values()))["tip"].shape[0]
    print(f"\nOpen-loop rollouts, split = {split}  ({n_rollouts} rollouts, 1 step = 0.02 s, h = 50 is 1 s ahead)")
    print("Fingertip error, cm: mean [p10 - p90]")
    print(f"{'model':6s}" + "".join(f"{f'h={h}':>22s}" for h in REPORT_HORIZONS))
    for name, e in errors.items():
        cells = []
        for h in REPORT_HORIZONS:
            tip = e["tip"][:, h - 1]  # column h-1 = h steps ahead
            cells.append(f"{tip.mean():7.2f} [{np.percentile(tip, 10):5.2f} -{np.percentile(tip, 90):6.2f}]")
        print(f"{name:6s}" + "".join(f"{c:>22s}" for c in cells))

    print("Mean angle error q1 / q2, rad")
    print(f"{'model':6s}" + "".join(f"{f'h={h}':>22s}" for h in REPORT_HORIZONS))
    for name, e in errors.items():
        cells = [f"{e['angle'][:, h - 1, 0].mean():.3f} / {e['angle'][:, h - 1, 1].mean():.3f}" for h in REPORT_HORIZONS]
        print(f"{name:6s}" + "".join(f"{c:>22s}" for c in cells))


def check_success(errors: dict[str, dict[str, np.ndarray]]) -> None:
    """The two rollout success criteria from PROGRESS.md, on mean fingertip error."""
    tip = {name: e["tip"].mean(axis=0) for name, e in errors.items()}  # (H,) mean over rollouts
    m1_wins = tip["M1"] < tip["B1"]
    print(f"M1 beats B1 at {m1_wins.sum()} / {H} horizons" + ("" if m1_wins.all() else f" (loses at h = {np.flatnonzero(~m1_wins) + 1})"))
    ratio = tip["M2"][9] / tip["G2"][9]  # index 9 = h = 10
    print(f"M2 / G2 fingertip error at h = 10: {ratio:.1f}x  (criterion: >= 2x)")


# ---------- 4.3 plots and W&B ----------

def plot_error_vs_horizon(split: str, errors: dict, guess_cm: float, path: Path) -> Path:
    """Fingertip error vs horizon for one split: mean line + 10-90 % band over rollouts, log y."""
    import matplotlib

    matplotlib.use("Agg")  # draw into a file, no window
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.25})
    h = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for name, e in errors.items():
        tip = e["tip"]  # (R, H)
        ax.fill_between(h, np.percentile(tip, 10, axis=0), np.percentile(tip, 90, axis=0),
                        color=MODEL_COLORS[name], alpha=0.12, linewidth=0)
        ax.plot(h, tip.mean(axis=0), color=MODEL_COLORS[name], linewidth=2, label=name)
    ax.axhline(guess_cm, color="gray", ls="--", lw=1.2, label=f"random guess ({guess_cm:.0f} cm)")
    ax.set_yscale("log")  # errors span 0.05 to 20 cm
    ax.set(title=f"{SPLIT_TITLES[split]}\nfingertip error vs horizon, open-loop rollouts",
           xlabel="horizon h (steps of 0.02 s; 50 = 1 s ahead)", ylabel="fingertip error [cm] (log)", xlim=(1, H))
    ax.legend(title="model: mean line, band = 10-90 %", fontsize=9, loc="lower right")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_example_rollouts(split: str, rolls: dict, errors: dict, path: Path,
                          percentiles: tuple[int, ...] = (25, 50, 90)) -> Path:
    """Rollouts vs ground truth for 3 example rollouts: a good, a typical and a bad one, ranked by M1's error at h = H.

    Columns: shoulder rotation since the start, elbow angle, fingertip path.
    rolls: model name -> (q_roll, q_true), each (R, H+1, 2); q_true is the same for every model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_true = next(iter(rolls.values()))[1].numpy()   # (R, H+1, 2)
    n = q_true.shape[0] // len(STARTS)               # episodes per start
    order = np.argsort(errors["M1"]["tip"][:, -1])
    picks = [order[round(p / 100 * (len(order) - 1))] for p in percentiles]
    h = np.arange(H + 1)

    fig, axes = plt.subplots(len(picks), 3, figsize=(18, 5 * len(picks)), constrained_layout=True)
    for r, (p, i) in enumerate(zip(percentiles, picks)):
        t0, episode = STARTS[i // n], i % n          # rollouts are stacked start by start (see rollouts())
        truth = q_true[i]
        series = [("ground truth", truth, TRUTH_COLOR, 3.0)]
        series += [(name, q_roll[i].numpy(), MODEL_COLORS[name], 1.6) for name, (q_roll, _) in rolls.items()]

        # Shoulder: relative to the start, since the raw angle keeps counting full turns. Elbow: raw (limited to +-3).
        for j, (title, ylabel) in enumerate([("shoulder rotation since start", "q1 - q1(start) [rad]"),
                                             ("elbow angle", "q2 [rad]")]):
            ax = axes[r, j]
            for label, q, color, lw in series:
                ax.plot(h, q[:, j] - (truth[0, 0] if j == 0 else 0), color=color, lw=lw, label=label)
            # Keep the y range on the truth; a line that leaves the plot is an error beyond that range.
            y = truth[:, j] - (truth[0, 0] if j == 0 else 0)
            pad = max(1.0, 0.3 * (y.max() - y.min()))
            ax.set_ylim(y.min() - pad, y.max() + pad)
            if j == 1:
                for lim in (-3, 3):
                    ax.axhline(lim, color="gray", ls=":", lw=1)
            row = f"{['good', 'typical', 'bad'][r] if len(percentiles) == 3 else ''} for M1 ({p}th percentile)\n" if j == 0 else "\n"
            ax.set(title=f"{row}{title}  (episode {episode}, start t0 = {t0})", xlabel="horizon h (steps)", ylabel=ylabel)

        ax = axes[r, 2]
        reach = (LINK0 + LINK1) * 100
        ax.add_patch(plt.Circle((0, 0), reach, fill=False, ls="--", color="gray"))
        for label, q, color, lw in series:
            xy = fingertip_xy(q) * 100
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, label=label)
            ax.plot(*xy[-1], "o", color=color, ms=8, markeredgecolor="white")  # dot = position at h = H
        ax.plot(*(fingertip_xy(truth[0]) * 100), "s", color=TRUTH_COLOR, ms=9)  # square = shared start
        ax.set(title=f"fingertip path (square = start, dot = h = {H})\nM1 error at h = {H}: {errors['M1']['tip'][i, -1]:.1f} cm",
               xlabel="x [cm]", ylabel="y [cm]", xlim=(-reach - 2, reach + 2), ylim=(-reach - 2, reach + 2))
        ax.set_aspect("equal")
    # 2. one legend for the whole figure: every panel uses the same colours
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=12, frameon=False)
    fig.suptitle(f"{SPLIT_TITLES[split]}: example 50-step rollouts vs ground truth\n"
                 f"rows: a good, typical and bad rollout for M1; lines leaving the angle plots = error beyond the shown range",
                 fontsize=13)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def log_to_wandb(one_step_by_split: dict, errors_by_split: dict, guess_by_split: dict, figures: list[Path]) -> None:
    """One W&B run "eval_rollouts": tables of every number, interactive error-vs-horizon charts, and the figures."""
    import wandb

    run = wandb.init(project="physai-reacher", name="eval_rollouts", job_type="eval",
                     config={"checkpoints": CHECKPOINTS, "horizon": H, "starts": list(STARTS), "min_warmup": W,
                             "g2_warmup": "all true steps from 0 to t0-1"})

    keys = ["angle_rmse_q1", "angle_rmse_q2", "vel_rmse_qd1", "vel_rmse_qd2", "tip_mean_cm", "tip_p95_cm"]
    one_step_rows = [[split, name] + [e.get(k) for k in keys]  # None where a model has no velocity
                     for split, results in one_step_by_split.items() for name, e in results.items()]
    horizon_rows = []
    for split, errors in errors_by_split.items():
        for name, e in errors.items():
            for h in range(1, H + 1):
                tip, angle = e["tip"][:, h - 1], e["angle"][:, h - 1]
                horizon_rows.append([split, name, h, tip.mean(), np.percentile(tip, 10), np.percentile(tip, 90),
                                     angle[:, 0].mean(), angle[:, 1].mean()])
    log = {
        "one_step": wandb.Table(columns=["split", "model"] + keys, data=one_step_rows),
        "rollout_error_vs_horizon": wandb.Table(
            columns=["split", "model", "h", "tip_mean_cm", "tip_p10_cm", "tip_p90_cm", "q1_err_rad", "q2_err_rad"],
            data=[[float(v) if isinstance(v, (np.floating, float)) else v for v in row] for row in horizon_rows]),
    }
    for split, errors in errors_by_split.items():
        log[f"{split}/tip_error_vs_horizon"] = wandb.plot.line_series(
            xs=list(range(1, H + 1)),
            ys=[e["tip"].mean(axis=0).tolist() for e in errors.values()] + [[guess_by_split[split]] * H],
            keys=list(errors) + ["random guess"],
            title=f"{split}: mean fingertip error [cm] vs horizon", xname="horizon h (steps)")
    log |= {p.stem: wandb.Image(str(p)) for p in figures}
    run.log(log)

    for split, errors in errors_by_split.items():  # headline numbers, for the runs table
        for name, e in errors.items():
            for h in REPORT_HORIZONS:
                run.summary[f"{split}/{name}/tip_cm_h{h}"] = float(e["tip"][:, h - 1].mean())
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step and open-loop rollout evaluation of the dynamics models.")
    parser.add_argument("--wandb", action="store_true", help="also log tables, charts and figures to W&B")
    args = parser.parse_args()

    models = load_models()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    one_step_by_split, errors_by_split, guess_by_split, figures = {}, {}, {}, []
    for split in ("test", "ood"):
        q, qd, a = load_episodes(split)
        one_step = {name: one_step_errors(model, q, qd, a) for name, model in models.items()}
        print_one_step_table(split, one_step)

        rolls = {name: rollouts(model, q, qd, a) for name, model in models.items()}  # name -> (q_roll, q_true)
        errors = {name: rollout_errors(*roll) for name, roll in rolls.items()}
        guess_by_split[split] = random_guess_cm(next(iter(rolls.values()))[1])
        print_rollout_table(split, errors)
        print(f"random guess (fingertip of another rollout): {guess_by_split[split]:.2f} cm")
        check_success(errors)

        figures.append(plot_error_vs_horizon(split, errors, guess_by_split[split], FIG_DIR / f"rollout_error_vs_horizon_{split}.png"))
        figures.append(plot_example_rollouts(split, rolls, errors, FIG_DIR / f"rollout_examples_{split}.png"))
        one_step_by_split[split], errors_by_split[split] = one_step, errors

    print("\nfigures: " + ", ".join(str(p) for p in figures))
    if args.wandb:
        log_to_wandb(one_step_by_split, errors_by_split, guess_by_split, figures)


if __name__ == "__main__":
    main()
