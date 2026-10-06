"""Reacher arm dynamics dataset: whole trajectories of (q, qd, a) for learning s' = f(s, a)."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

VERSION = "reacher_dynamics_v1"  
DT = 0.02            # env.dt = frame_skip (2) * timestep (0.01)
JOINT1_LIMIT = 3.0    # elbow range in reacher.xml
LINK0, LINK1 = 0.1, 0.11  # shoulder->elbow, elbow->fingertip in metres
INIT_QD_MAX = np.array([20.0, 10.0])  # rad/s start-velocity range; correlated actions reach ~55 / ~25 (p95)


def make_env():
    """Returns the raw MuJoCo simulator (no time limit or observation wrappers)."""
    env = gym.make("Reacher-v5")
    env.reset(seed=0)  # MuJoCo data is only valid after one reset
    return env.unwrapped


def wrap_angle(x: np.ndarray) -> np.ndarray:
    """Wraps angles into (-pi, pi]."""
    return np.arctan2(np.sin(x), np.cos(x))


def fingertip_xy(q: np.ndarray) -> np.ndarray:
    """Forward kinematics: joint angles (..., 2) ->fingertip (x, y) in metres (..., 2)."""
    q1, q2 = q[..., 0], q[..., 1]
    x = LINK0 * np.cos(q1) + LINK1 * np.cos(q1 + q2)
    y = LINK0 * np.sin(q1) + LINK1 * np.sin(q1 + q2)
    return np.stack([x, y], axis=-1)

# ---------- action sequences: each returns (T, 2) torques in [-1, 1] ----------

def ou_actions(rng: np.random.Generator, T: int, theta: float = 0.15, sigma: float = 0.3) -> np.ndarray:
    """Ornstein-Uhlenbeck noise: a random walk that is pulled back towards 0."""
    a = np.empty((T, 2))
    x = rng.uniform(-1, 1, 2)
    for t in range(T):
        x = np.clip(x + theta * (0 - x) + sigma * rng.normal(size=2), -1, 1)
        a[t] = x
    return a


def sticky_actions(rng: np.random.Generator, T: int, hold: tuple[int, int] = (5, 15)) -> np.ndarray:
    """Pick a random torque and hold it for 5-15 steps, then pick another."""
    a = np.empty((T, 2))
    t = 0
    while t < T:
        k = rng.integers(hold[0], hold[1] + 1)
        a[t : t + k] = rng.uniform(-1, 1, 2)  # slicing past T is fine, numpy stops at the end
        t += k
    return a


def bang_bang_actions(rng: np.random.Generator, T: int, hold: tuple[int, int] = (5, 20)) -> np.ndarray:
    """Full torque (+1 or -1 per joint), switching after a random 5-20 steps. Reaches top speeds."""
    a = np.empty((T, 2))
    t = 0
    while t < T:
        k = rng.integers(hold[0], hold[1] + 1)
        a[t : t + k] = rng.choice([-1.0, 1.0], size=2)
        t += k
    return a


def sine_actions(rng: np.random.Generator, T: int) -> np.ndarray:
    """Smooth sinusoids with random frequency, phase and amplitude per joint."""
    t = np.arange(T)[:, None] * DT        # (T, 1) time in seconds
    freq = rng.uniform(0.2, 2.0, 2)       # Hz, one per joint
    phase = rng.uniform(0, 2 * np.pi, 2)
    amp = rng.uniform(0.3, 1.0, 2)
    return amp * np.sin(2 * np.pi * freq * t + phase)  # broadcasts to (T, 2)


def chirp_actions(rng: np.random.Generator, T: int, f0: float = 0.1, f1: float = 4.0) -> np.ndarray:
    """Sinusoid whose frequency sweeps from f0 to f1 Hz. Not used in training: different-policy test set."""
    t = np.arange(T)[:, None] * DT
    rate = (f1 - f0) / (T * DT)           # Hz per second
    phase = rng.uniform(0, 2 * np.pi, 2)
    amp = rng.uniform(0.5, 1.0, 2)
    return amp * np.sin(2 * np.pi * (f0 * t + 0.5 * rate * t**2) + phase)


def iid_actions(rng: np.random.Generator, T: int) -> np.ndarray:
    """Independent uniform torque every step; only used to show poor coverage."""
    return rng.uniform(-1, 1, (T, 2))


TRAIN_GENERATORS = (ou_actions, sticky_actions, bang_bang_actions, sine_actions)


def mixed_actions(rng: np.random.Generator, T: int, max_segments: int = 3) -> np.ndarray:
    """Training policy: cut the episode into 1-3 segments, each from a random training generator."""
    n_segments = rng.integers(1, max_segments + 1)
    cuts = np.sort(rng.choice(np.arange(10, T - 10), size=n_segments - 1, replace=False))
    lengths = np.diff(np.r_[0, cuts, T])  # e.g. cuts [30, 70] -> lengths [30, 40, 30]
    segments = [TRAIN_GENERATORS[rng.integers(len(TRAIN_GENERATORS))](rng, n) for n in lengths]
    return np.concatenate(segments)


POLICIES = {"mixed": mixed_actions, "chirp": chirp_actions, "iid": iid_actions}


# ---------- collection ----------

def random_start(rng: np.random.Generator, sim) -> None:
    """Teleports the arm to a random pose and velocity. The target is left where it is."""
    mujoco.mj_resetData(sim.model, sim.data)  # clear everything left over from the previous episode
    qpos, qvel = sim.init_qpos.copy(), sim.init_qvel.copy()
    qpos[0] = rng.uniform(-np.pi, np.pi)                # shoulder: full circle
    qpos[1] = rng.uniform(-JOINT1_LIMIT, JOINT1_LIMIT)  # elbow: within its limits
    qvel[:2] = rng.uniform(-INIT_QD_MAX, INIT_QD_MAX)
    sim.set_state(qpos, qvel)


def collect_episodes(n_episodes: int, T: int, policy: str, seed: int) -> dict[str, np.ndarray]:
    """Rolls out n_episodes of T steps, each from a random start with one action sequence.

    Returns:
        q:  (n, T+1, 2) raw joint angles (the shoulder is NOT wrapped)
        qd: (n, T+1, 2) joint velocities in rad/s
        a:  (n, T, 2)   torques; a[:, t] moves state t to state t+1
        episode_id: (n,)
    """
    rng = np.random.default_rng(seed)
    sim = make_env()  # one simulator reused for every episode
    assert np.isclose(sim.dt, DT), sim.dt
    assert np.allclose(sim.model.jnt_range[1], [-JOINT1_LIMIT, JOINT1_LIMIT])

    q = np.empty((n_episodes, T + 1, 2), dtype=np.float32)
    qd = np.empty((n_episodes, T + 1, 2), dtype=np.float32)
    a = np.empty((n_episodes, T, 2), dtype=np.float32)

    for ep in range(n_episodes):
        random_start(rng, sim)
        a[ep] = POLICIES[policy](rng, T)
        q[ep, 0], qd[ep, 0] = sim.data.qpos[:2], sim.data.qvel[:2]
        for t in range(T):
            sim.do_simulation(a[ep, t], sim.frame_skip)  
            q[ep, t + 1], qd[ep, t + 1] = sim.data.qpos[:2], sim.data.qvel[:2]

    sim.close()
    return {"q": q, "qd": qd, "a": a, "episode_id": np.arange(n_episodes)}


# ---------- split + save ----------

def split_by_episode(
    data: dict[str, np.ndarray], seed: int, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, np.ndarray]:
    """Splits whole episodes into train/val/test, so no test trajectory leaks into training."""

    n = len(data["q"])
    perm = np.random.default_rng(seed).permutation(n)  # shuffled episode ids, e.g. [512, 3, 871, ...]
    n_train, n_val = int(fractions[0] * n), int(fractions[1] * n)
    split_idx = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    # Indexing axis 0 picks whole episodes; q, qd, a and episode_id stay paired.
    return {f"{split}_{key}": arr[idx] for split, idx in split_idx.items() for key, arr in data.items()}


def load_dynamics_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Loads the arrays and the metadata dict saved by main()."""
    with np.load(path) as f:
        arrays = {key: f[key] for key in f.files if key != "meta"}
        meta = json.loads(str(f["meta"]))
    return arrays, meta


# ---------- coverage ----------

def coverage_summary(d: dict[str, np.ndarray], bins: int = 30) -> dict[str, float]:
    """Numbers behind the plots: how much of joint space and velocity space the data visits."""
    q = d["q"].reshape(-1, 2)
    qd = d["qd"].reshape(-1, 2)
    q_hist, _, _ = np.histogram2d(wrap_angle(q[:, 0]), q[:, 1], bins=bins, range=[[-np.pi, np.pi], [-3, 3]])
    qd_hist, _, _ = np.histogram2d(qd[:, 0], qd[:, 1], bins=bins, range=[[-100, 100], [-40, 40]])
    return {
        "q_cells_visited": (q_hist > 0).mean(),    # fraction of the (q1, q2) grid with any data
        "qd_cells_visited": (qd_hist > 0).mean(),  # same for (qd1, qd2) within +-100 / +-40 rad/s
        "qd1_p99": np.percentile(np.abs(qd[:, 0]), 99),
        "qd2_p99": np.percentile(np.abs(qd[:, 1]), 99),
    }


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow: one per dataset


def plot_coverage(datasets: dict[str, dict[str, np.ndarray]], out_dir: str | Path) -> list[Path]:
    """Saves coverage_hist.png (1-D histograms) and coverage_space.png (2-D heatmaps).

    Args:
        datasets: name -> dict with q (n, T+1, 2), qd (n, T+1, 2), a (n, T, 2).
        out_dir: folder for the two .png files.

    Returns:
        The paths of the saved figures.
    """
    import matplotlib

    matplotlib.use("Agg")  # draw into a file, no window
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(datasets, SERIES_COLORS))
    joints = ["shoulder (joint 1)", "elbow (joint 2)"]
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.25})

    # ---- Figure 1: one variable at a time. Rows: angle, velocity, action. Columns: shoulder, elbow. ----
    rows = [
        # Only the shoulder is wrapped: the elbow is limited, and wrapping its +-3.4 overshoot would flip sides.
        ("angle", "angle [rad]", lambda d, j: (wrap_angle(d["q"][..., j]) if j == 0 else d["q"][..., j]).ravel(), False),
        ("velocity", "velocity [rad/s]", lambda d, j: d["qd"][..., j].ravel(), True),
        ("torque command", "action a (torque, -1 to +1)", lambda d, j: d["a"][..., j].ravel(), False),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    for r, (what, xlabel, get, log_y) in enumerate(rows):
        for j in range(2):
            ax = axes[r, j]
            # Shared bin edges, so the datasets are compared on the same grid.
            edges = np.histogram_bin_edges(np.concatenate([get(d, j) for d in datasets.values()]), 60)
            for name, d in datasets.items():
                x = get(d, j)
                label = name
                if what == "velocity":
                    label += f"  (99% of samples within +-{np.percentile(np.abs(x), 99):.0f})"
                ax.hist(x, bins=edges, weights=np.full(len(x), 100 / len(x)),  # bar height = % of samples
                        histtype="step", linewidth=2, color=colors[name], label=label)
            if what == "angle" and j == 1:
                for lim in (-JOINT1_LIMIT, JOINT1_LIMIT):
                    ax.axvline(lim, color="gray", ls="--", lw=1)
                ax.text(JOINT1_LIMIT, ax.get_ylim()[1] * 0.95, "elbow limit +-3 ", ha="right", va="top",
                        color="dimgray", fontsize=9)
            if log_y:
                ax.set_yscale("log")  # log scale shows the rare fast tails, where the datasets differ
            unit = xlabel.replace("angle", "angle (wrapped)") if what == "angle" and j == 0 else xlabel
            ax.set(title=f"{joints[j]}: {what}", xlabel=f"{joints[j].split()[0]} {unit}",
                   ylabel="% of samples per bin" + (" (log scale)" if log_y else ""))
            ax.legend(fontsize=9, loc="upper left" if what != "velocity" else "lower center")
    fig.suptitle("Coverage, one variable at a time: how often each value appears in each dataset\n"
                 "Each line is a dataset; a wider line = a wider range of values the model gets to see",
                 fontsize=13)
    hist_path = out_dir / "coverage_hist.png"
    fig.savefig(hist_path, dpi=110)
    plt.close(fig)

    # ---- Figure 2: 2-D heatmaps. Rows: datasets. Columns: joint angles, joint velocities, fingertip. ----
    panels = [
        ("arm configurations", lambda d: (wrap_angle(d["q"][..., 0]), d["q"][..., 1]),
         [[-np.pi, np.pi], [-3.5, 3.5]], "shoulder angle q1 (wrapped) [rad]", "elbow angle q2 [rad]"),
        ("speed combinations", lambda d: (d["qd"][..., 0], d["qd"][..., 1]),
         [[-180, 180], [-50, 50]], "shoulder velocity qd1 [rad/s]", "elbow velocity qd2 [rad/s]"),
        ("fingertip positions", lambda d: tuple(np.moveaxis(fingertip_xy(d["q"]) * 100, -1, 0)),
         [[-23, 23], [-23, 23]], "fingertip x [cm]", "fingertip y [cm]"),
    ]
    # Percent of samples per cell, with one colour scale per column so datasets are comparable.
    grids = {}
    for c, (_, get, rng_xy, _, _) in enumerate(panels):
        for name, d in datasets.items():
            x, y = (v.ravel() for v in get(d))
            h, xe, ye = np.histogram2d(x, y, bins=60, range=rng_xy)
            grids[c, name] = (100 * h / len(x), xe, ye)
    vmax = {c: max(grids[c, n][0].max() for n in datasets) for c in range(len(panels))}

    fig, axes = plt.subplots(len(datasets), 3, figsize=(18, 5.6 * len(datasets)), constrained_layout=True,
                             squeeze=False)
    for r, name in enumerate(datasets):
        for c, (what, _, _, xlabel, ylabel) in enumerate(panels):
            ax = axes[r, c]
            h, xe, ye = grids[c, name]
            mesh = ax.pcolormesh(xe, ye, np.ma.masked_equal(h, 0).T, cmap="Blues",  # empty cells stay white
                                 norm=LogNorm(vmin=1e-3, vmax=vmax[c]))
            ax.set_facecolor("white")
            ax.grid(False)
            ax.set(title=f"{name}: {what}", xlabel=xlabel, ylabel=ylabel)
            if c == 0:
                for lim in (-JOINT1_LIMIT, JOINT1_LIMIT):
                    ax.axhline(lim, color="#eb6834", ls="--", lw=1)
                ax.text(np.pi - 0.1, JOINT1_LIMIT - 0.15, "elbow limit", ha="right", va="top", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "#eb6834", "boxstyle": "round,pad=0.2"})
            if c == 2:
                ax.set_aspect("equal")
                ax.add_patch(plt.Circle((0, 0), (LINK0 + LINK1) * 100, fill=False, ls="--", color="gray"))
                ax.text(0, (LINK0 + LINK1) * 100 - 1.5, "max reach 21 cm", ha="center", va="top", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "gray", "boxstyle": "round,pad=0.2"})
            fig.colorbar(mesh, ax=ax, label="% of samples in cell (log)", shrink=0.85)
    fig.suptitle("Coverage in 2-D: one row per dataset. Darker blue = more samples, white = never visited\n"
                 "Left: which arm poses   Middle: which speed combinations   Right: where the fingertip goes",
                 fontsize=13)
    space_path = out_dir / "coverage_space.png"
    fig.savefig(space_path, dpi=110)
    plt.close(fig)
    return [hist_path, space_path]

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the Reacher dynamics dataset.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ood-episodes", type=int, default=100, help="different-policy test set (chirp)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/reacher_dynamics_v1.npz")
    parser.add_argument("--plots", type=str, default="projects/reacher_dynamics/figures")
    parser.add_argument("--wandb", action="store_true", help="also log the coverage plots to W&B")
    args = parser.parse_args()

    # Different seeds, so the two datasets never share start states or action sequences.
    data = collect_episodes(args.episodes, args.steps, "mixed", args.seed)
    ood = collect_episodes(args.ood_episodes, args.steps, "chirp", args.seed + 1)
    splits = split_by_episode(data, args.seed)

    meta = {
        "version": VERSION,
        "seed": args.seed,
        "ood_seed": args.seed + 1,
        "dt": DT,
        "steps": args.steps,
        "n_episodes": args.episodes,
        "n_ood_episodes": args.ood_episodes,
        "split": [0.8, 0.1, 0.1],
        "train_policy": "mixed: 1-3 segments of OU / sticky(5-15) / bang-bang(5-20) / sine(0.2-2 Hz)",
        "ood_policy": "chirp 0.1 -> 4 Hz",
        "state": "q = qpos[:2] raw (shoulder unwrapped), qd = qvel[:2]",
        "init_qd_max": INIT_QD_MAX.tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **splits,
        **{f"ood_{key}": arr for key, arr in ood.items()},
        meta=np.array(json.dumps(meta)),  # a dict can't go in .npz directly, so store it as a JSON string
    )
    sizes = ", ".join(f"{s} {len(splits[f'{s}_q'])}" for s in ("train", "val", "test"))
    print(f"saved {args.out}: {sizes}, ood {len(ood['q'])} episodes")

    # Coverage check. The i.i.d. set is only for comparison and is not saved.
    train = {key: splits[f"train_{key}"] for key in ("q", "qd", "a")}
    iid = collect_episodes(len(train["q"]), args.steps, "iid", args.seed + 2)  # same size as train
    datasets = {"mixed (train)": train, "iid": iid, "chirp (ood)": ood}
    for name, d in datasets.items():
        c = coverage_summary(d)
        print(f"{name:>14}: q cells {c['q_cells_visited']:.0%}  qd cells {c['qd_cells_visited']:.0%}  "
              f"|qd| p99 [{c['qd1_p99']:.0f}, {c['qd2_p99']:.0f}] rad/s")
    paths = plot_coverage(datasets, args.plots)
    print("saved " + ", ".join(str(p) for p in paths))

    if args.wandb:
        import wandb

        run = wandb.init(project="physai-reacher", name="dynamics_coverage", job_type="data", config=meta)
        run.log({p.stem: wandb.Image(str(p)) for p in paths})
        run.finish()


if __name__ == "__main__":
    main()
