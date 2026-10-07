"""Render open-loop rollouts in MuJoCo as an MP4: one solid arm per model over the true arm (transparent ghost).

    uv run python projects/reacher_dynamics/render_rollout.py                          # M1 + G2, default example
    uv run python projects/reacher_dynamics/render_rollout.py --models G2              # G2 only
    uv run python projects/reacher_dynamics/render_rollout.py --split ood --episode 44 --start 30

Nothing is simulated here: every arm is posed from stored angles (each model's rollout and the recorded truth) with
mj_forward, which only computes positions. Rollouts are made exactly as in eval_rollouts.py (G2 warms up on 0..t0-1).

Shape letters:
    T   steps per episode (100 actions, 101 states)
    H   rollout horizon in steps (default 50 = 1 s); start + H <= T
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen GPU rendering; must be set before mujoco is imported

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from physai.datasets.reacher_dynamics import DT, fingertip_xy
from physai.models.dynamics import ConstantVelocity, load_checkpoint

DATA_PATH = "data/reacher_dynamics_v1.npz"
OUT_DIR = Path("projects/reacher_dynamics/media")
REACHER_XML = Path(gymnasium.__file__).parent / "envs/mujoco/assets/reacher.xml"

# One entry per model that can be drawn. Colours match the plots (SERIES_COLORS in reacher_dynamics.py).
MODELS = {
    "B1": {"checkpoint": None, "color": (0.16, 0.47, 0.84), "label": "B1 constant velocity"},
    "M1": {"checkpoint": "checkpoints/mlp_full_s0.pt", "color": (0.92, 0.41, 0.20), "label": "M1 MLP, sees velocity"},
    "M2": {"checkpoint": "checkpoints/mlp_angles_s0.pt", "color": (0.11, 0.69, 0.48), "label": "M2 MLP, angles only"},
    "G2": {"checkpoint": "checkpoints/gru_angles_s0.pt", "color": (0.93, 0.63, 0.00), "label": "G2 GRU, angles + memory"},
}
TRUE_COLOR, TRUE_ALPHA = (1.0, 1.0, 1.0), 0.45   # see-through white ghost
ARM_HEIGHT = 0.01                                 # z of the first arm; each next arm sits 1.5 mm higher (no flicker)


def build_scene_xml(arms: list[str]) -> str:
    """Reacher XML with one arm per name in `arms` (bodies "M1_body0", ...) plus the ghost "true_body0..." on top.

    The original arm is used as a template and removed; every copy gets its own names, colour and height.
    """
    root = ET.parse(REACHER_XML).getroot()
    world = root.find("worldbody")
    world.remove(world.find("body[@name='target']"))   # no target: we only show the arm
    root.remove(root.find("actuator"))                # no motors: angles are set directly
    world.find("geom[@name='ground']").set("rgba", "0.16 0.18 0.22 1")  # dark floor, so the white ghost stands out
    for side in ("sideS", "sideE", "sideN", "sideW"):   # arena walls at +-30 cm: decoration only, the arm reaches 21 cm
        world.remove(world.find(f"geom[@name='{side}']"))
    template = world.find("body[@name='body0']")
    world.remove(template)

    for i, name in enumerate(arms + ["true"]):
        arm = copy.deepcopy(template)
        arm.set("pos", f"0 0 {ARM_HEIGHT + 0.0015 * i:.4f}")  # the ghost is last, so it is drawn on top
        rgba = (*TRUE_COLOR, TRUE_ALPHA) if name == "true" else (*MODELS[name]["color"], 1.0)
        for element in arm.iter():
            if element.get("name"):
                element.set("name", f"{name}_{element.get('name')}")
            if element.tag == "geom":
                element.set("rgba", " ".join(f"{c:.3f}" for c in rgba))
                element.set("contype", "0")
                element.set("conaffinity", "0")
        world.append(arm)

    # Top-down camera over the arm (reach 21 cm) and a light from above.
    ET.SubElement(world, "camera", name="top", pos="0 0 0.8", xyaxes="1 0 0 0 1 0", fovy="45")
    ET.SubElement(world, "light", pos="0 0 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    # Offscreen image buffer: MuJoCo's default is 640 x 480, too small for larger square frames.
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1024", offheight="1024")
    return ET.tostring(root, encoding="unicode")


@torch.no_grad()
def model_rollouts(models: list[str], split: str, episode: int, start: int,
                   horizon: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Each model open loop on one episode, exactly as in eval_rollouts.py.

    G2 warms up on the true steps 0..start-1; B1 / M1 / M2 have no memory and ignore the warm-up.
    Returns {model name: (H+1, 2) predicted angles} and the true angles (H+1, 2); row 0 = the true start state.
    """
    data = np.load(DATA_PATH)
    q, qd, a = (torch.as_tensor(data[f"{split}_{k}"][episode:episode + 1], dtype=torch.float32) for k in ("q", "qd", "a"))
    if not 10 <= start <= a.shape[1] - horizon:
        raise ValueError(f"need 10 <= start <= {a.shape[1] - horizon} for horizon {horizon}, got start {start}")
    predictions = {}
    for name in models:
        path = MODELS[name]["checkpoint"]
        model = ConstantVelocity() if path is None else load_checkpoint(path)[0]
        state = model.warmup(q[:, :start], qd[:, :start], a[:, :start])
        q_pred, _ = model.rollout(q[:, start], qd[:, start], a[:, start:start + horizon], state)
        predictions[name] = q_pred[0].numpy()
    return predictions, q[0, start:start + horizon + 1].numpy()


def add_trail(scene: mujoco.MjvScene, points: np.ndarray, rgba: tuple) -> None:
    """Small spheres at past fingertip positions (k, 3), drawn on top of the rendered scene."""
    for p in points:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.003, 0, 0]), p, np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
        scene.ngeom += 1


def draw_legend(draw: ImageDraw.ImageDraw, top: int, models: list[str], font) -> None:
    """Colour swatch + name for every arm, two entries per row, white text (the swatch carries the colour)."""
    entries = [(MODELS[n]["color"], 1.0, MODELS[n]["label"]) for n in models]
    entries.append((TRUE_COLOR, TRUE_ALPHA, "ground truth (transparent)"))
    for i, (color, alpha, label) in enumerate(entries):
        x, y = 12 + (i % 2) * 320, top + 10 + (i // 2) * 28
        fill = tuple(int(255 * (alpha * c + (1 - alpha) * 0.3)) for c in color)  # ghost: blended like on screen
        draw.rounded_rectangle([x, y + 3, x + 30, y + 17], radius=4, fill=fill, outline=(200, 200, 200))
        draw.text((x + 40, y), label, fill="white", font=font)


def render_frames(predictions: dict[str, np.ndarray], q_true: np.ndarray, title: str,
                  size: int = 640) -> list[np.ndarray]:
    """One image (size, size, 3) per step h = 0..H: all arms posed, fingertip trails, caption and legend."""
    names = list(predictions)
    model = mujoco.MjModel.from_xml_string(build_scene_xml(names))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, size, size)
    arms = names + ["true"]
    joints = {n: [model.joint(f"{n}_joint{j}").qposadr[0] for j in (0, 1)] for n in arms}
    tips = {n: model.body(f"{n}_fingertip").id for n in arms}
    angles = predictions | {"true": q_true}
    trail_rgba = {n: (*MODELS[n]["color"], 1.0) for n in names} | {"true": (*TRUE_COLOR, 0.8)}
    tip_error_cm = {n: np.linalg.norm(fingertip_xy(predictions[n]) - fingertip_xy(q_true), axis=-1) * 100
                    for n in names}  # (H+1,) per model

    font = ImageFont.load_default(size=17)
    legend_rows = (len(arms) + 1) // 2
    legend_top = size - (16 + 28 * legend_rows)
    trails = {n: [] for n in arms}
    frames = []
    for h in range(len(q_true)):
        for n in arms:
            data.qpos[joints[n]] = angles[n][h]
        mujoco.mj_forward(model, data)                 # positions only, no physics step
        for n in arms:
            trails[n].append(data.xpos[tips[n]].copy())

        renderer.update_scene(data, camera="top")
        for n in arms:
            add_trail(renderer.scene, np.array(trails[n]), trail_rgba[n])
        image = Image.fromarray(renderer.render())

        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, size, 62], fill=(20, 20, 20))
        draw.text((10, 6), title, fill="white", font=font)
        errors = "   ".join(f"{n} {tip_error_cm[n][h]:4.1f} cm" for n in names)
        draw.text((10, 34), f"h = {h:2d}/{len(q_true) - 1}  t = {h * DT:.2f} s   fingertip error: {errors}",
                  fill="white", font=font)
        draw.rectangle([0, legend_top, size, size], fill=(20, 20, 20))
        draw_legend(draw, legend_top, names, font)
        frames.append(np.asarray(image))
    renderer.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Render model rollouts against the ground truth as an MP4.")
    parser.add_argument("--models", nargs="+", default=["M1", "G2"], choices=list(MODELS),
                        help="one solid arm per model, drawn in this order")
    parser.add_argument("--split", default="test", choices=["test", "ood"])
    parser.add_argument("--episode", type=int, default=85)
    parser.add_argument("--start", type=int, default=50, help="first predicted step t0 (G2 warms up on 0..t0-1)")
    parser.add_argument("--horizon", type=int, default=50, help="steps to predict open loop (50 = 1 s)")
    parser.add_argument("--fps", type=int, default=10, help="10 fps = 5x slow motion (one step is 0.02 s)")
    args = parser.parse_args()

    predictions, q_true = model_rollouts(args.models, args.split, args.episode, args.start, args.horizon)
    title = f"Open-loop rollout, {args.split} episode {args.episode}, start t0 = {args.start}"
    frames = render_frames(predictions, q_true, title)
    frames += [frames[-1]] * args.fps                  # hold the last frame for 1 s

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(n.lower() for n in args.models)
    path = OUT_DIR / f"rollout_{tag}_{args.split}_ep{args.episode}_t{args.start}.mp4"
    # H.264 with yuv420p pixels: the format LinkedIn, browsers and phones all play.
    imageio.mimwrite(path, frames, fps=args.fps, codec="libx264", quality=8, pixelformat="yuv420p")
    print(f"saved {path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
