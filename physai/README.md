# `physai`: the shared package

Reusable code for the projects in [`projects/`](../projects/): simulators and datasets, and model definitions.
`uv sync` installs it in editable mode, so `import physai` works from any script in the repo.

## `physai/datasets/`

| Module | Used by | What it does |
|---|---|---|
| [`reacher.py`](datasets/reacher.py) | [fingertip](../projects/reacher_fingertip/README.md) | top-down camera, random arm states, image rendering, random 80/10/10 split. Run as a script to build the dataset. |
| [`reacher_torch.py`](datasets/reacher_torch.py) | fingertip | PyTorch `Dataset`: images as tensors, labels scaled to [−1, 1], label-safe augmentation |
| [`reacher_dynamics.py`](datasets/reacher_dynamics.py) | [dynamics](../projects/reacher_dynamics/README.md) | the 4 training action generators + chirp, episode collection from random starts, split by episode, forward kinematics, coverage plots. Run as a script to build the dataset. |
| [`reacher_dynamics_torch.py`](datasets/reacher_dynamics_torch.py) | dynamics | the two data views (`full` / `angles`), train-only normalisation statistics, step and episode datasets |

## `physai/models/`

| Module | Used by | Models |
|---|---|---|
| [`cnn.py`](models/cnn.py) | fingertip | small CNN backbone; A (global average pooling), B (spatial softmax) |
| [`resnet.py`](models/resnet.py) | fingertip | C and D: ImageNet ResNet18 regressor, frozen or fine-tuned |
| [`__init__.py`](models/__init__.py) | fingertip | `build_model(cfg)`: picks A–D from a Hydra config |
| [`dynamics.py`](models/dynamics.py) | dynamics | B1 constant velocity, M1/M2 MLP, G2 GRU; shared `step` / `warmup` / `rollout` interface; `load_checkpoint` |
