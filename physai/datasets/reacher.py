import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

HALF_EXTENT = 0.25
FOVY_DEG = 45.0
TOP_DOWN_CAMERA = {
    "trackbodyid": -1,                                            # fixed camera
    "distance": HALF_EXTENT / np.tan(np.deg2rad(FOVY_DEG / 2)),   # height so the view is +/- 0.25 m
    "elevation": -90.0,                                           # look straight down
    "azimuth": 90.0,                                              # +x right, +y up in the image
    "lookat": np.array([0.0, 0.0, 0.0]),                          # centre on the arm base
}

JOINT1_LIMIT = 3.0   # elbow range in reacher.xml is [-3.0, 3.0] rad
TARGET_RADIUS = 0.2  # Reacher's own reset keeps the target within 0.2 m of the base

def make_env(size: int = 96) -> gym.Env:
    """Creates a Reacher environment with a top-down camera.

    Args:
        size: The width and height of the rendered image in pixels.

    Returns:
        A Reacher environment.
    """
    env = gym.make(
        "Reacher-v5",
        render_mode="rgb_array",
        width = size,
        height = size,
        default_camera_config = TOP_DOWN_CAMERA,
    )
    env.reset(seed=0)
    return env


def world_to_pixel(xy: np.ndarray, size: int = 96) -> np.ndarray:
    """Converts world (x, y) in metres to image (col, row) in pixels.

    Only valid for TOP_DOWN_CAMERA.

    Args:
        xy: One point of shape (2,) or a batch of shape (N, 2), in metres.
        size: The width and height of the image in pixels.

    Returns:
        Pixel coordinates (col, row) with the same leading shape as xy.
    """
    xy = np.asarray(xy)
    centre = (size - 1) / 2              # 47.5 for 96 px: pixels are numbered 0..95
    px_per_m = (size / 2) / HALF_EXTENT  # 48 px per 0.25 m = 192 px per metre
    col = centre + xy[..., 0] * px_per_m
    row = centre - xy[..., 1] * px_per_m # minus: image rows grow down, world y grows up
    return np.stack([col, row], axis=-1)

def sample_qpos(rng: np.random.Generator) -> np.ndarray:
    """Samples a random simulator state: arm pose and red target position.

    Args:
        rng: NumPy random generator. Using only this keeps the dataset reproducible.

    Returns:
        qpos of shape (4,): [shoulder angle, elbow angle, target x, target y].
    """
    shoulder = rng.uniform(-np.pi, np.pi)           # joint0 has no limit: full circle
    elbow = rng.uniform(-JOINT1_LIMIT, JOINT1_LIMIT)  # joint1 is limited to +/- 3 rad

    # Rejection sampling: draw from the square, keep only points inside the circle.
    while True:
        target = rng.uniform(-TARGET_RADIUS, TARGET_RADIUS, size=2)
        if np.linalg.norm(target) < TARGET_RADIUS:
            break

    return np.array([shoulder, elbow, target[0], target[1]])


def render_sample(env: gym.Env, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Teleports the simulator to qpos and takes one picture.

    Args:
        env: Environment from make_env.
        qpos: State of shape (4,) from sample_qpos.

    Returns:
        image: (size, size, 3) uint8 RGB frame.
        fingertip_xy: (2,) float32 fingertip position in metres (the label).
        target_xy: (2,) float32 red target position in metres.
    """
    sim = env.unwrapped                   
    qvel = np.zeros(sim.model.nv)         
    sim.set_state(qpos, qvel)             

    image = env.render()
    fingertip_xy = sim.get_body_com("fingertip")[:2].astype(np.float32)
    target_xy = sim.get_body_com("target")[:2].astype(np.float32)
    return image, fingertip_xy, target_xy


def make_reacher_dataset(n: int, size: int = 96, seed: int = 0) -> dict[str, np.ndarray]:
    """Renders n random frames with labels. Day 5 imports this.

    Args:
        n: Number of frames.
        size: Image width and height in pixels.
        seed: Random seed; the same seed always gives the same dataset.

    Returns:
        Dict of arrays:
            images:    (n, size, size, 3) uint8
            fingertip: (n, 2) float32, the label in metres
            target:    (n, 2) float32, red dot position, for failure analysis
            joints:    (n, 2) float32, shoulder/elbow angles, for the stretch goal
    """
    rng = np.random.default_rng(seed)
    env = make_env(size)  # one env reused for every frame: creating it is slow

    # Allocate the full arrays once, then fill row by row.
    images = np.empty((n, size, size, 3), dtype=np.uint8)
    fingertip = np.empty((n, 2), dtype=np.float32)
    target = np.empty((n, 2), dtype=np.float32)
    joints = np.empty((n, 2), dtype=np.float32)

    for i in range(n):
        qpos = sample_qpos(rng)
        images[i], fingertip[i], target[i] = render_sample(env, qpos)
        joints[i] = qpos[:2]

    env.close()
    return {"images": images, "fingertip": fingertip, "target": target, "joints": joints}


def save_sample_grid(images: np.ndarray, fingertip: np.ndarray, path: str | Path) -> None:
    """Saves a 4x4 grid of images with the fingertip labelled."""
    import matplotlib

    matplotlib.use("Agg")  # draw into a file, no window
    import matplotlib.pyplot as plt

    size = images.shape[1]
    pixels = world_to_pixel(fingertip, size)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, img, (col, row), (x, y) in zip(axes.flat, images, pixels, fingertip):
        ax.imshow(img)
        ax.scatter([col], [row], s=80, facecolors="none", edgecolors="lime", linewidths=1.5)
        ax.set_title(f"({x * 100:.1f}, {y * 100:.1f}) cm", fontsize=8)
        ax.axis("off")
    fig.suptitle("Reacher samples: green circle = labelled fingertip")
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def split_dataset(
    data: dict[str, np.ndarray], seed: int, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, np.ndarray]:
    
    n = len(data["images"])
    perm = np.random.default_rng(seed).permutation(n)  # e.g. [3812, 17, 4410, ...]
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)

    split_idx = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    # The same indices are used for every array, so image i and label i stay paired.
    return {f"{split}_{key}": arr[idx] for split, idx in split_idx.items() for key, arr in data.items()}


def load_reacher_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Loads a saved dataset back into a plain dict of arrays."""
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Reacher fingertip dataset.")
    parser.add_argument("--n", type=int, default=5000, help="number of frames")
    parser.add_argument("--size", type=int, default=96, help="image width/height in pixels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=str, default=None, help="also save a 4x4 sample grid to this .png")
    parser.add_argument("--out", type=str, default=None, help="save the split dataset to this .npz")
    args = parser.parse_args()

    data = make_reacher_dataset(args.n, args.size, args.seed)
    print(f"rendered {args.n} frames of {args.size}x{args.size}")

    if args.grid:
        save_sample_grid(data["images"][:16], data["fingertip"][:16], args.grid)
        print(f"saved grid to {args.grid}")

    if args.out:
        splits = split_dataset(data, args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **splits, seed=np.array(args.seed))
        sizes = {split: len(splits[f"{split}_images"]) for split in ("train", "val", "test")}
        print(f"saved {args.out}  {sizes}")


if __name__ == "__main__":
    main()
