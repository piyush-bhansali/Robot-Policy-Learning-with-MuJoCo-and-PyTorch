"""Forward dynamics models for the Reacher arm: (q, qd, a) -> (q', qd').

All models share one interface, in raw units (rad, rad/s, torque):
    step(q, qd, a, state)            -> q', qd', state
    warmup(q_hist, qd_hist, a_hist)  -> state
    rollout(q0, qd0, actions, state) -> q, qd of shape (B, H+1, 2)

    B1  ConstantVelocity             baseline, no training
    M1  MLPDynamics(obs="full")      main model
    M2  MLPDynamics(obs="angles")    velocity hidden, no memory
    G2  GRUDynamics(obs="angles")    velocity hidden, memory

Shape letters:
    B      batch: states or episodes processed together (e.g. 256 steps for the MLP, 32 episodes for the GRU)
    T      time steps in a training sequence (100)
    H      rollout horizon, steps predicted open loop (50)
    W      warm-up steps of true history before a rollout (10)
    n_in   input columns: 8 for obs="full", 6 for obs="angles"
    n_out  target columns: 4 for obs="full" (dq, dqd), 2 for obs="angles" (dq)
"""
import torch
from torch import nn

from physai.datasets.reacher_dynamics import DT
from physai.datasets.reacher_dynamics_torch import INPUT_NAMES, TARGET_NAMES


def make_features_torch(q: torch.Tensor, qd: torch.Tensor, a: torch.Tensor, obs: str) -> torch.Tensor:
    """Must match reacher_dynamics_torch.make_features column for column (INPUT_NAMES)."""
    parts = [torch.sin(q), torch.cos(q)]
    if obs == "full":
        parts.append(qd)
    parts.append(a)
    return torch.cat(parts, dim=-1)


class DynamicsModel(nn.Module):
    obs = "full"

    def step(self, q, qd, a, state=None):
        """(B, 2) each -> q', qd' (B, 2) and the updated state (None for models without memory)."""
        raise NotImplementedError

    def warmup(self, q_hist, qd_hist, a_hist):
        """Builds memory from W true past steps, each (B, W, 2). Only the GRU uses it."""
        return None

    def rollout(self, q0, qd0, actions, state=None):
        """Open loop: start from the true state, then feed each prediction back in. actions: (B, H, 2)."""
        q, qd = q0, qd0
        qs, qds = [q], [qd]
        for t in range(actions.shape[1]):
            q, qd, state = self.step(q, qd, actions[:, t], state)
            qs.append(q)
            qds.append(qd)
        return torch.stack(qs, dim=1), torch.stack(qds, dim=1)


class LearnedDynamics(DynamicsModel):
    """Base for fitted models. Train-split statistics live in buffers, so checkpoints carry them."""

    def __init__(self, obs: str, stats: dict | None = None):
        super().__init__()
        if obs not in INPUT_NAMES:
            raise ValueError(f"obs must be 'full' or 'angles', got {obs!r}")
        self.obs = obs
        self.n_in, self.n_out = len(INPUT_NAMES[obs]), len(TARGET_NAMES[obs])
        sizes = {"x_mean": self.n_in, "x_std": self.n_in, "y_mean": self.n_out, "y_std": self.n_out}
        for name, size in sizes.items():
            if stats is not None:
                value = torch.as_tensor(stats[name]).float()
            else:  # placeholders; load_state_dict() overwrites them
                value = torch.ones(size) if name.endswith("std") else torch.zeros(size)
            self.register_buffer(name, value)

    def normalise_x(self, x):
        return (x - self.x_mean) / self.x_std

    def normalise_y(self, y):
        return (y - self.y_mean) / self.y_std

    def features(self, q, qd, a):
        return self.normalise_x(make_features_torch(q, qd, a, self.obs))

    def apply_delta(self, q, qd, y_norm):
        """Normalised predicted change -> next state in raw units."""
        y = y_norm * self.y_std + self.y_mean
        dq = y[..., :2]
        if self.obs == "full":
            return q + dq, qd + y[..., 2:]
        # Angles-only models don't predict velocity; this finite-difference estimate is for reporting only.
        return q + dq, dq / DT

    def step(self, q, qd, a, state=None):
        """Default for memoryless models: forward() maps normalised features to the normalised change."""
        q_next, qd_next = self.apply_delta(q, qd, self(self.features(q, qd, a)))
        return q_next, qd_next, None


# ---------- baseline ----------

class ConstantVelocity(DynamicsModel):
    """B1: q' = q + qd * dt, qd' = qd. Ignores torque and the joint limit."""

    def step(self, q, qd, a, state=None):
        return q + qd * DT, qd, None


# ---------- learned models ----------

class MLPDynamics(LearnedDynamics):
    """M1 (obs="full") and M2 (obs="angles"): predicts each step from the current input only."""

    def __init__(self, obs: str, stats: dict | None = None, hidden: int = 256):
        super().__init__(obs, stats)
        self.net = nn.Sequential(
            nn.Linear(self.n_in, hidden),   # hidden layer 1: n_in (8 or 6) -> 256
            nn.SiLU(),
            nn.Linear(hidden, hidden),      # hidden layer 2: 256 -> 256
            nn.SiLU(),
            nn.Linear(hidden, hidden),      # hidden layer 3: 256 -> 256
            nn.SiLU(),
            nn.Linear(hidden, self.n_out),  # output: 256 -> n_out (4 or 2), no activation: a change can be any sign
        )

    def forward(self, x_norm):
        """(..., n_in) -> (..., n_out), both normalised."""
        return self.net(x_norm)


class GRUDynamics(LearnedDynamics):
    """G2 (obs="angles"): the hidden state h stands in for the unseen velocity."""

    def __init__(self, obs: str, stats: dict | None = None, hidden: int = 256):
        super().__init__(obs, stats)
        self.gru = nn.GRU(self.n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, self.n_out)

    def forward(self, x_norm, h=None):
        """(B, T, n_in) -> (B, T, n_out) and the final h (1, B, hidden).

        Training calls this on whole episodes of true inputs (teacher forcing).
        """
        out, h = self.gru(x_norm, h)
        return self.head(out), h

    def warmup(self, q_hist, qd_hist, a_hist):
        _, h = self(self.features(q_hist, qd_hist, a_hist))
        return h

    def step(self, q, qd, a, state=None):
        y_norm, state = self(self.features(q, qd, a).unsqueeze(1), state)  # one time step: (B, 1, n_in)
        q_next, qd_next = self.apply_delta(q, qd, y_norm.squeeze(1))
        return q_next, qd_next, state


# ---------- build / load ----------

def build_model(kind: str, obs: str, stats: dict | None = None, hidden: int = 256) -> LearnedDynamics:
    """kind "mlp" -> M1 (obs="full") or M2 (obs="angles"); kind "gru" -> G2 (obs="angles")."""
    if kind == "mlp":
        return MLPDynamics(obs, stats, hidden)
    if kind == "gru":
        return GRUDynamics(obs, stats, hidden)
    raise ValueError(f"unknown model kind: {kind!r}")


def load_checkpoint(path: str, device: str = "cpu") -> tuple[LearnedDynamics, dict]:
    """Rebuilds a trained model from a checkpoint saved by train.py and checks its metadata."""
    checkpoint = torch.load(path, map_location=device)
    meta = checkpoint["meta"]
    model = build_model(meta["kind"], meta["obs"], hidden=meta["hidden"])  # placeholder buffers...
    model.load_state_dict(checkpoint["state_dict"])                         # ...replaced by the saved ones
    # A model is only valid with the input layout and time step it was trained on.
    assert meta["input_names"] == INPUT_NAMES[meta["obs"]], "input column layout changed since training"
    assert meta["target_names"] == TARGET_NAMES[meta["obs"]], "target layout changed since training"
    assert meta["dt"] == DT, f"checkpoint dt {meta['dt']} != current dt {DT}"
    return model.to(device).eval(), meta
