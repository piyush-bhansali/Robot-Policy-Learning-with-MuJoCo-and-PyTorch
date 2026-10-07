# Reacher arm dynamics — progress tracker

Goal: learn **f(sₜ, aₜ) → sₜ₊₁** for the Reacher arm, test it with 50-step open-loop rollouts
against baselines, and show when memory (a GRU) is needed by hiding the joint velocities.

Status key: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Part 1 — Data collection (`physai/datasets/reacher_dynamics.py`)
- [x] 1.1 Simulator basics: raw env, `wrap_angle` (atan2), `fingertip_xy` (forward kinematics) — FK matches MuJoCo to 1e-8 m
- [x] 1.2 Action generators: OU, sticky (5–15), bang-bang (±1, 5–20), sine, chirp, i.i.d.; `mixed` = 1–3 segments of OU/sticky/bang/sine — lag-1 autocorr 0.82–0.98 (i.i.d. 0.01)
- [x] 1.3 Collect episodes: `mj_resetData` + random start via `set_state`, whole trajectories (n, T+1, 2) — 50 eps in 0.19 s, reproducible
- [x] 1.4 Split by episode 80/10/10 + different-policy (chirp 0.1→4 Hz) test set; saved `data/reacher_dynamics_v1.npz` (2.1 MB) with JSON metadata
- [x] 1.5 Coverage plots: `figures/coverage_hist.png`, `figures/coverage_space.png` (train vs i.i.d. vs chirp). W&B upload: run with `--wandb` (not done yet)

## Part 2 — Two views of the data (`physai/datasets/reacher_dynamics_torch.py`)
- [x] 2.1 Dataset A (full state): x (n,100,8) = [sin q, cos q, q̇, a] → y (n,100,4) = [Δq, Δq̇] — `make_features(..., 'full')`
- [x] 2.2 Dataset B (angles only): x (n,100,6) = [sin q, cos q, a] → y (n,100,2) = [Δq] — `make_features(..., 'angles')`
- [x] 2.3 z-score stats on train split only (`compute_norm_stats`) + `TransitionDataset` (MLP) / `SequenceDataset` (GRU)

## Part 3 — Models (`physai/models/dynamics.py`)
Scope (user decision): **M1** (MLP, full state), **M2** (MLP, angles only — the control), **G2** (GRU, angles only).
Baseline: **B1** constant velocity only (user decision). Dropped: B0, B2, G1, M3.
- [x] 3.1 Shared interface: `step`, `warmup`, `rollout`; `LearnedDynamics` with normalisation in `register_buffer`
- [x] 3.1b Baseline B1 `ConstantVelocity`: q' = q + q̇·dt, q̇' = q̇
- [x] 3.2 M1: `MLPDynamics('full')` — 3 × 256, SiLU, predicts normalised [Δq, Δq̇] — trained: best epoch 62, val loss 0.0141
- [x] 3.2b M2: `MLPDynamics('angles')` — same MLP with `obs=angles`, predicts normalised Δq — trained: best epoch 21, val loss 0.579
- [x] 3.3 G2: `GRUDynamics('angles')` — GRU 256 + linear head; `forward` on whole episodes (teacher forcing), `warmup` builds h, `step` = one GRU step — trained: best epoch 186, val loss 0.00045
- [x] 3.4 Training: `projects/reacher_dynamics/train.py` + configs (`model=mlp|gru`, `obs=full|angles`), Adam 1e-3, early stopping (patience 20), grad clip 1.0 (GRU), W&B, checkpoints → `checkpoints/<run_name>.pt`

## Part 4 — Evaluation (`projects/reacher_dynamics/eval_rollouts.py`)
- [x] 4.1 One-step test + chirp (OOD) error for every model, steps t ≥ W = 10 (G2 warm-up), via `model.step` with the true state
- [x] 4.2 50-step open-loop rollouts (wrapped angle error, fingertip error in cm): starts t0 = 10..50, 500 rollouts per split; G2 warm-up = all true steps 0..t0-1
- [x] 4.3 Plots: `figures/rollout_error_vs_horizon_{test,ood}.png`, `figures/rollout_examples_{test,ood}.png`; W&B run `eval_rollouts` (tables, line charts, images)
- [ ] 4.4 (optional) multi-step loss (k = 5) fine-tune of M1

## Part 5 — Deliverables
- [ ] 5.1 Checkpoints with normalisation buffers + metadata; reload test in a fresh script
- [ ] 5.2 Results table + 4–6 lines of findings in README
- [ ] 5.3 Push code

---

## Success criteria
- [x] Coverage spans the workspace and velocity range
- [x] M1 beats B1 at every horizon 1–50 (50/50 on test and chirp)
- [x] G2 clearly beats M2 on angles only (≥ 2× lower error at h = 10) — 15.8× test, 21.1× chirp
- [x] M1 vs G2: how close memory gets to seeing the velocity — G2 beats M1 at every horizon (2–5× lower fingertip error)
- [x] Error-vs-horizon curves and example rollouts logged and explained
- [ ] Checkpoint reloads and passes the metadata assert

---

## Notes & facts (filled in as we go)
- Reacher-v5: `dt = 0.02 s` (frame_skip 2 × timestep 0.01); torques clipped to [−1, 1], gear 200.
- Shoulder (joint0) has **no limit**: its angle is not wrapped by MuJoCo. Elbow (joint1) range is [−3, 3] rad.
- Fingertip FK: x = 0.1·cos q₁ + 0.11·cos(q₁+q₂), y = same with sin (matches `get_body_com("fingertip")`).
- Speeds: correlated actions reach |q̇₁| ≈ 55 rad/s (p95), up to ~120; i.i.d. actions only ≈ 21 rad/s (p95).
- Measured on 50 mixed episodes: max |q̇| ≈ [142, 46] rad/s → shoulder moves up to ~2.85 rad/step (close to π). Elbow overshoots its soft limit to ±3.44.
- **Δq target = raw `q[t+1] − q[t]`** (stored q is unwrapped), never a wrapped difference: wrapping would flip steps > π. Wrap only for errors/plots.
- **Confirmed (2.1):** max |Δq₁| = 3.42 rad > π in train — 74 of 80k steps (0.09%, 7 episodes). A wrapped Δq would have flipped these.
  Side effect for view B: from angles alone (sin/cos), a 3.42 rad step looks like −2.86 rad (aliasing), so even the GRU can't resolve these steps. 6% of steps exceed π/2.
- corr(Δq₁, q̇₁·dt) = 0.999: Δq is almost velocity × dt; the model learns the small correction from the torque.
- Train stats (full): x std [0.71 0.56 0.71 0.72 | q̇ 39.7 13.4 | a 0.68 0.68], cos q₂ mean −0.42 (elbow-pinned bias);
  y std [Δq 0.80 0.27 | Δq̇ 2.52 4.89]. Δq std ≈ q̇ std × dt. Δq̇₂ > Δq̇₁: elbow velocity jumps at limit hits.
  Val normalised y std 0.85–0.95 (fewer fast episodes in 100 val eps) — expected, stats are train-only.
- **Model checks (3.1–3.3):** torch features == numpy (max diff 6e-8); params M1 134,916 / M2 133,890 / G2 203,266;
  all rollouts (100, 51, 2) from the true state; GRU step-by-step == whole sequence (3e-8); state_dict reload restores buffers.
- **B1 one-step (test, 10k steps):** angle RMSE 0.026 / 0.052 rad, velocity RMSE 2.56 / 4.83 rad/s, fingertip mean 0.34 cm (p95 1.0, max 5.9).
  Elbow error 3× larger at the limit (0.077 vs 0.024 rad) — B1 can't see the stop coming.
- **Training smoke tests (3 epochs):** M1 val loss 0.034→0.022, RMSE Δq 0.015/0.024 rad, Δq̇ 0.05/1.38 rad/s (~1.7 s/epoch).
  G2 val loss 0.27→0.085, RMSE Δq 0.17/0.09 rad (~0.5 s/epoch). Both checkpoints reload via `load_checkpoint`.
- Build pairs by slicing the time axis per episode (`q[:, :-1]`, `q[:, 1:]`), then flatten — never flatten first (would pair ep k's last state with ep k+1's first).
- Sine is in training (user choice), so the different-policy test set uses a **chirp** instead of sinusoids.
- **Coverage (1.5):** q₁ uniform for all sets; fingertip fills the 21 cm disc (hole r ≈ 1.8 cm: elbow can't fold past ±3).
  |q̇₁| p99 / max — train 120 / 171, i.i.d. 29 / 48, chirp 56 / 74 rad/s → correlated actions give ~4× the speed range of i.i.d.
  Chirp states lie inside the training range, so the OOD set tests *new action patterns*, not unseen states.
- **Bias to watch:** 41% of train states have the elbow pinned at its limit (|q₂| > 2.9; i.i.d. 17%, chirp 25%) — the light elbow
  slams into the limit under any sustained torque, and 27% of train actions are at ±1 (bang-bang). Shows as the bright q̇₂ = 0 line.
  Accepted for now; revisit if errors are worst mid-range of q₂.
  *Why:* the elbow has no spring and weak damping (1.0 vs gear 200), so any held torque accelerates it until it hits the stop
  (torque 1.0 → limit in 9 steps, 0.05 → 44 steps; episode = 100). Pinned share by generator: bang 55%, sticky 40%, OU 37%, sine 31%, i.i.d. 16%.
  *Fix if needed:* scale the elbow torque down in some segments, or add more fast-switching segments.
- **Training results (3.2–3.4, val split, seed 0).** Loss = MSE on normalised targets (1.0 ≈ always predicting the train mean);
  RMSE in raw units: Δq in rad per step, Δq̇ in rad/s per step.

  | Model | Best epoch | Train / val loss | dq1 | dq2 | dqd1 | dqd2 | Checkpoint |
  |---|---|---|---|---|---|---|---|
  | B1 (test, for reference) | – | – | 0.026 | 0.052 | 2.56 | 4.83 | – |
  | M1 mlp full | 62 | 0.0146 / 0.0141 | 0.0176 | 0.0152 | 0.039 | 1.12 | `checkpoints/mlp_full_s0.pt` |
  | M2 mlp angles | 21 | 0.694 / 0.579 | 0.632 | 0.193 | – | – | `checkpoints/mlp_angles_s0.pt` |
  | G2 gru angles | 186 | 0.0004 / 0.0004 | 0.0084 | 0.0074 | – | – | `checkpoints/gru_angles_s0.pt` |

  - M1 ≈ train/val equal (no overfitting); beats B1 on every target (Δq̇₁ 65×, Δq̇₂ 4×).
  - M2 fails as expected: without q̇ one frame can't tell a still arm from a spinning one (dq1 error 0.63 rad ≈ 36°/step).
  - G2's memory recovers the hidden velocity: dq1 75× better than M2 and even below M1. Caveats: G2 skips the first 10 (warmup)
    steps of each episode, uses teacher forcing, and spends all capacity on 2 targets — the 50-step rollouts (Part 4) are the fair test.
  - M2 val loss < train loss because val has fewer fast episodes (mean-predictor scores 0.72 / 0.91 on val, not 1.0), not a bug.
- **M1 Δq̇₂ error = elbow joint-limit impacts.** Val RMSE by situation: elbow free (61% of steps) 0.41 rad/s, at limit |q₂| > 2.9
  (39%) 1.72, big impacts |Δq̇₂| > 10 (3%) 3.35. Median |error| 0.094, max 18.2; worst 1% of steps = 61% of squared error.
  Δq̇₂ is ~93% of M1's val loss (normalised MSE per target: 0.0005 / 0.0033 / 0.0002 / 0.0525). The MLP smooths the
  near-discontinuous "hit the stop within this step or not". Revisit only if rollouts drift after impacts
  (options: Huber loss, distance-to-limit feature, more near-limit data).
- **Reproducibility:** M1 trained twice with identical config (`outputs/2026-10-06/23-18-36`, `outputs/2026-10-07/12-02-34`):
  same best epoch and max weight difference 0.0 — MLP training is deterministic for a fixed seed.
  G2's first run (`outputs/2026-10-06/23-32-18`) was stopped by hand at epoch 31 (val 0.0030); the full run is `2026-10-07/11-55-24`.
- **Where runs live:** every run keeps its own `outputs/<date>/<time>/best.pt`; `checkpoints/<run_name>.pt` is overwritten by the
  latest run with the same name — use `seed=N` for separate copies. `train.log` is empty (`accelerator.print` goes to stdout);
  the full per-epoch log is in `wandb/run-*/files/output.log`.
- **One-step error (4.1)**, 100 episodes × 90 steps, wrapped angle RMSE (rad), velocity RMSE (rad/s), fingertip (cm):

  | Model | Split | q1 | q2 | qd1 | qd2 | tip mean | tip p95 |
  |---|---|---|---|---|---|---|---|
  | B1 | test | 0.0258 | 0.0528 | 2.57 | 4.91 | 0.331 | 0.97 |
  | M1 | test | 0.0167 | 0.0160 | 0.042 | 1.21 | 0.172 | 0.58 |
  | M2 | test | 0.730 | 0.200 | – | – | 4.57 | 17.7 |
  | G2 | test | 0.0084 | 0.0079 | – | – | 0.083 | 0.25 |
  | B1 | chirp | 0.0208 | 0.0406 | 2.07 | 3.93 | 0.349 | 0.87 |
  | M1 | chirp | 0.0180 | 0.0143 | 0.037 | 1.07 | 0.211 | 0.62 |
  | M2 | chirp | 0.417 | 0.240 | – | – | 4.01 | 12.1 |
  | G2 | chirp | 0.0077 | 0.0062 | – | – | 0.083 | 0.24 |

  - Checks passed: B1 matches its earlier numbers; M1 / M2 / G2 test ≈ their val numbers, so `step()` agrees with training.
  - No drop on chirp for M1 / G2: chirp states lie inside the training range, so new action patterns alone don't hurt.
  - G2 has ~half M1's one-step angle and fingertip error. Likely cause: M1's loss is dominated by Δq̇₂ (elbow impacts),
    so Δq gets less of its capacity; G2 only predicts Δq. Open question until the rollouts: M1's velocity feeds the next step, G2's h must carry it.
- **G2 warm-up (4.2):** a fresh h started mid-episode (10-step window t0-10..t0-1) gave 6× worse first steps
  (one-step at t0 = 50: 0.78 cm vs 0.08 cm with full history). In training h was only ever zero at step 0, where the arm is slow
  (start |q̇| ≤ 20 rad/s); mid-episode it is often 50+. Fix (option A): warm up on all true steps 0..t0-1 — also how it runs online.
  Option B (not done): train on random windows so a short warm-up works anywhere.
- **Rollouts (4.2):** M1 and M2 trained on every step of every episode (incl. fast mid-episode states), so mid-episode starts are fair.
  M1's weakness is the elbow: it runs through the ±3 limit after an impact (q2 error 0.99 rad at h = 50 vs G2 0.11) and then the
  wrong velocity is carried forever. G2 never predicts velocity, so the stop costs it little. Chirp mainly hurts at long horizons
  (M1 9.8 → 15.0 cm, G2 4.5 → 7.3 cm at h = 50). B1 and M2 hit the random-guess level by h ≈ 10.
- File lives in `physai/datasets/` (matches the existing repo layout) instead of the spec's `physai/data/`.

## Results
Fingertip error in cm, mean over 500 rollouts (100 episodes × starts t0 = 10, 20, 30, 40, 50). 1-step = teacher forcing, steps t ≥ 10.
Random guess (fingertip of another rollout) = 14 cm test, 16 cm chirp.

| Model | Obs | Split | 1-step | @1 | @5 | @10 | @25 | @50 |
|---|---|---|---|---|---|---|---|---|
| B1 | full | test | 0.331 | 0.37 | 6.39 | 13.32 | 14.48 | 15.20 |
| M1 | full | test | 0.172 | 0.17 | 1.20 | 3.44 | 7.52 | 9.78 |
| M2 | angles | test | 4.567 | 4.10 | 11.36 | 12.06 | 12.23 | 11.91 |
| G2 | angles | test | **0.083** | **0.10** | **0.39** | **0.76** | **2.23** | **4.54** |
| B1 | full | chirp | 0.349 | 0.32 | 6.67 | 15.01 | 17.08 | 18.25 |
| M1 | full | chirp | 0.211 | 0.18 | 1.28 | 3.94 | 9.54 | 15.00 |
| M2 | angles | chirp | 4.011 | 3.71 | 13.28 | 14.64 | 15.13 | 15.99 |
| G2 | angles | chirp | **0.085** | **0.09** | **0.34** | **0.70** | **2.48** | **7.26** |

Mean wrapped angle error q1 / q2 (rad) at h = 10 and 50: test M1 0.103 / 0.288 → 0.488 / 0.990, G2 0.079 / 0.029 → 0.526 / 0.111;
chirp M1 0.094 / 0.351 → 0.429 / 1.503, G2 0.075 / 0.030 → 0.492 / 0.239.

## Findings
_(4–6 lines, written at the end)_
