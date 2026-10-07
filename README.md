# PhysAI

Learning-based robotics experiments in PyTorch on a simulated robot arm: the MuJoCo **Reacher**, a 2-joint arm
from Gymnasium. Each project asks one question about how a neural network can perceive or model a physical system,
and answers it with a controlled comparison of several models.

![Open-loop rollout: M1 and G2 vs ground truth](projects/reacher_dynamics/media/rollout_m1_g2_test_ep21_t40.gif)

*Two learned dynamics models predicting the arm 1 s ahead from the starting state and the torques alone, feeding on
their own predictions. Orange-red = M1 (MLP, sees velocity), yellow = G2 (GRU, angles + memory), transparent white =
ground truth. MP4: [`rollout_m1_g2_test_ep21_t40.mp4`](projects/reacher_dynamics/media/rollout_m1_g2_test_ep21_t40.mp4).
Details in the [dynamics project](projects/reacher_dynamics/README.md).*

---

## Projects

| Project | Question | Models compared | Headline result |
|---|---|---|---|
| [**Reacher fingertip**](projects/reacher_fingertip/README.md) | Where is the fingertip in this camera image? | A small CNN + global pooling, B small CNN + spatial softmax, C ResNet18 frozen, D ResNet18 fine-tuned | B wins with 0.11 cm mean error (validation), a fifth of a pixel, at 1 % of ResNet18's size |
| [**Reacher dynamics**](projects/reacher_dynamics/README.md) | Where will the arm be 1 s from now, given the torques? | B1 constant velocity, M1 MLP (sees velocity), M2 MLP (angles only), G2 GRU (angles only, memory) | G2 is best at every horizon: 0.8 cm at 0.2 s and 4.5 cm at 1 s, against 3.4 and 9.8 cm for M1 |

Each project README explains how its dataset was made, the models, training, the results and the commands to
reproduce them. The shared code is described in [`physai/README.md`](physai/README.md).

### Reacher fingertip, in short

5,000 rendered 96 × 96 top-down images with exact fingertip labels from the simulator. The comparison shows that
*how* a network pools its features matters more than its size: a spatial-softmax keypoint layer keeps track of where
things are in the image and beats a frozen and a fine-tuned ImageNet ResNet18 by a wide margin.
→ [Full README](projects/reacher_fingertip/README.md)

### Reacher dynamics, in short

1,000 simulated episodes driven by a mix of torque patterns (smooth noise, held torques, bang-bang, sines), plus a test
set with an unseen pattern (a chirp). Models predict the change in state per step and are judged on 50-step open-loop
rollouts. Findings: one-step error hides how fast errors compound; a GRU with memory recovers the hidden velocity and
beats the model that sees velocity directly; and the elbow's joint limit is where the MLP breaks.
→ [Full README](projects/reacher_dynamics/README.md)

| Fingertip error (cm), test | 0.02 s | 0.2 s | 1 s |
|---|---|---|---|
| B1 constant velocity | 0.37 | 13.3 | 15.2 |
| M1 MLP, sees velocity | 0.17 | 3.44 | 9.78 |
| M2 MLP, angles only | 4.10 | 12.1 | 11.9 |
| **G2 GRU, angles + memory** | **0.10** | **0.76** | **4.54** |

---

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # creates .venv with all dependencies and installs physai (editable)
uv run wandb login   # once: training and evaluation log to Weights & Biases (project physai-reacher)
```

Run every command from the repo root with `uv run python ...`; the project READMEs list them. Datasets are generated
locally into `data/` (a few seconds to a minute each) and are not stored in git.

**Docker** (optional):

```bash
docker build -t physai .
docker run --rm --gpus all physai    # prints the PyTorch version and whether CUDA is available
```

**Rendering.** The dynamics videos render MuJoCo offscreen with EGL (`MUJOCO_GL=egl`, set by the script). Without a
GPU, use `MUJOCO_GL=osmesa`.

## Repository layout

```
PhysAI/
├── physai/                      shared package: datasets and models          → physai/README.md
│   ├── datasets/                Reacher simulators, data generation, PyTorch datasets
│   └── models/                  CNNs, ResNet18 regressor, dynamics models
├── projects/
│   ├── reacher_fingertip/       image → fingertip position                    → README.md
│   │   ├── train.py, evaluate.py, configs/, figures/
│   └── reacher_dynamics/        state + torque → next state, rollouts         → README.md
│       ├── train.py, eval_rollouts.py, render_rollout.py, configs/, figures/, media/
├── pyproject.toml, uv.lock      dependencies (uv)
└── Dockerfile
```

Created when you run things (all in `.gitignore`):

| Folder | Contents |
|---|---|
| `data/` | generated datasets (`.npz`) |
| `outputs/`, `multirun/` | one folder per training run (Hydra): config, `best.pt` |
| `checkpoints/` | dynamics checkpoints used by the evaluation scripts |
| `wandb/` | local Weights & Biases files |

## Tools

PyTorch · MuJoCo / Gymnasium · Hydra (configs and sweeps) · Hugging Face Accelerate · Weights & Biases · timm ·
matplotlib · uv
