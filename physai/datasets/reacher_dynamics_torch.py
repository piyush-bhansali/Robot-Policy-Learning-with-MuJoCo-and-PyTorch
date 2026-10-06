"""Two views of the same data:
    A, obs="full":   x = [sin q, cos q, qd, a]  ->  y = [dq, dqd]   (Markov: x holds the whole state)
    B, obs="angles": x = [sin q, cos q, a]      ->  y = [dq]        (not Markov: the velocity is hidden)

Shape letters used in this file:
    n: number of episodes
    T: steps per episode (T actions, T+1 states)
    F: input features per step  = len(INPUT_NAMES[obs]):  8 for "full", 6 for "angles"
    D: target values per step   = len(TARGET_NAMES[obs]): 4 for "full", 2 for "angles"
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from physai.datasets.reacher_dynamics import load_dynamics_npz

INPUT_NAMES = {
    "full": ["sin_q1", "sin_q2", "cos_q1", "cos_q2", "qd1", "qd2", "a1", "a2"],
    "angles": ["sin_q1", "sin_q2", "cos_q1", "cos_q2", "a1", "a2"],
}
TARGET_NAMES = {
    "full": ["dq1", "dq2", "dqd1", "dqd2"],
    "angles": ["dq1", "dq2"],
}


def make_features(q: np.ndarray, qd: np.ndarray, a: np.ndarray, obs: str) -> tuple[np.ndarray, np.ndarray]:
    """Builds per-step inputs and targets, keeping the episode axis.

    Args:
        q:  (n, T+1, 2) raw joint angles.
        qd: (n, T+1, 2) joint velocities.
        a:  (n, T, 2) actions.
        obs: "full" (dataset A) or "angles" (dataset B).

    Returns:
        x: (n, T, F) inputs at steps 0..T-1, columns as in INPUT_NAMES[obs] (F = 8 full / 6 angles).
        y: (n, T, D) change from step t to t+1, columns as in TARGET_NAMES[obs] (D = 4 full / 2 angles).
    """
    if obs not in INPUT_NAMES:
        raise ValueError(f"obs must be 'full' or 'angles', got {obs!r}")

    # Slice along the time axis of each episode, so step t and t+1 always come from the same episode.
    q_now, qd_now = q[:, :-1], qd[:, :-1]
    dq = q[:, 1:] - q[:, :-1]     # raw angles: exact even when one step moves more than pi
    dqd = qd[:, 1:] - qd[:, :-1]

    parts = [np.sin(q_now), np.cos(q_now)]  # (sin, cos) is the same for q and q + 2*pi
    if obs == "full":
        parts.append(qd_now)
    parts.append(a)
    x = np.concatenate(parts, axis=-1)
    y = np.concatenate([dq, dqd], axis=-1) if obs == "full" else dq
    return x.astype(np.float32), y.astype(np.float32)


def load_split(npz_path: str, split: str, obs: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads one split ("train", "val", "test" or "ood") and returns make_features(...) for it."""
    data, _ = load_dynamics_npz(npz_path)
    return make_features(data[f"{split}_q"], data[f"{split}_qd"], data[f"{split}_a"], obs)


def compute_norm_stats(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> dict[str, np.ndarray]:
    """Per-column mean and std of the inputs and targets. Call this on the TRAIN split only.

    Args:
        x: (n, T, F) inputs, y: (n, T, D) targets.

    Returns:
        {"x_mean": (F,), "x_std": (F,), "y_mean": (D,), "y_std": (D,)}, float32.
        The models copy these into register_buffers, so a checkpoint carries its own normalisation.
    """
    x = x.reshape(-1, x.shape[-1])  # every step of every episode counts once
    y = y.reshape(-1, y.shape[-1])
    return {
        "x_mean": x.mean(0), "x_std": x.std(0) + eps,  # eps: never divide by 0
        "y_mean": y.mean(0), "y_std": y.std(0) + eps,
    }


class TransitionDataset(Dataset):
    """Independent (x_t, y_t) pairs for the MLPs. Returns raw units; the model normalises."""

    def __init__(self, npz_path: str, split: str, obs: str):
        x, y = load_split(npz_path, split, obs)
        # Pairs were already formed inside each episode, so flattening now is safe.
        self.x = torch.from_numpy(x.reshape(-1, x.shape[-1]))
        self.y = torch.from_numpy(y.reshape(-1, y.shape[-1]))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]


class SequenceDataset(Dataset):
    """Whole episodes (x of shape (T, F), y of shape (T, D)) in time order, for the GRUs."""

    def __init__(self, npz_path: str, split: str, obs: str):
        x, y = load_split(npz_path, split, obs)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]
