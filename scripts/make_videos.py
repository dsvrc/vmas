#!/usr/bin/env python
"""Render evaluation videos from a finished run's checkpoints.

    python scripts/make_videos.py <run folder or checkpoint.pt> [...] --out videos

RUN THIS ON A MACHINE WITH A DISPLAY -- your laptop.  Not on the cluster: VMAS
draws through pyglet, pyglet needs an X/EGL context even for an invisible
window, and a compute node has neither.  That is why the sweeps keep
`experiment.render=false` and why this script exists: the server produces
CHECKPOINTS and nothing else, and the video is made here afterwards by
replaying the trained policy for one episode.  No training happens.

WHAT TO COPY DOWN, per run, is two things:

    <run>/config.pkl                           the task + its kwargs + configs
    <run>/checkpoints/checkpoint_<frames>.pt    the weights

`scripts/pack_checkpoints.sh` on the server collects exactly those into one
tarball.  Unpack it anywhere and point this script at the run folder.

WHAT THIS NEEDS INSTALLED: torch, torchrl, tensordict, vmas and this repo --
what training needed, and nothing more.  The frames are written by whichever
writer happens to be present (imageio-ffmpeg, OpenCV, Pillow), falling back to
a PNG sequence and then to a .npz of raw frames, so a missing video codec costs
you the container format and never the render.

Each argument is either

  * a run folder   .../mappo_balance_mlp__<hash>_<date>/   (the one holding
    config.pkl), in which case its newest checkpoint is used, or
  * a checkpoint   .../checkpoints/checkpoint_3000000.pt   directly.

`--all` renders EVERY checkpoint under the folder rather than the newest, which
is the before/after of training; the sweeps keep all of them on purpose.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))


def frames_of(path: pathlib.Path) -> int:
    """The frame count in a ``checkpoint_<frames>.pt`` name, or -1."""
    stem = path.stem.rsplit("_", 1)[-1]
    return int(stem) if stem.isdigit() else -1


def checkpoints_of(folder: pathlib.Path) -> list:
    """Every checkpoint under ``folder``, oldest first."""
    found = sorted(folder.glob("checkpoints/checkpoint_*.pt"), key=frames_of)
    if not found:
        #  The folder may have been given a level or two up: a row directory
        #  holding seed_<N>/<run>/checkpoints/.
        found = sorted(folder.glob("**/checkpoints/checkpoint_*.pt"), key=frames_of)
    if not found:
        raise SystemExit(
            f"no checkpoints/checkpoint_*.pt under {folder}.\n"
            "A video needs the trained weights. The launchers in this repo do "
            "write them (checkpoint_at_end, plus CKPT_EVERY during the run), "
            "so if there are none: check you are pointing at the folder that "
            "holds config.pkl, and that what you copied down included the .pt "
            "files rather than only the csv logs."
        )
    return found


def resolve(target: str, want_all: bool) -> list:
    path = pathlib.Path(target).expanduser().resolve()
    if path.is_file() and path.suffix == ".pt":
        return [path]
    if not path.is_dir():
        raise SystemExit(f"not a folder or a .pt checkpoint: {path}")
    found = checkpoints_of(path)
    return found if want_all else [found[-1]]


def label_of(checkpoint: pathlib.Path) -> str:
    """A name for the output file, taken from the run's place in the tree.

    The layout the launchers produce is ``<row>/seed_<N>/<run>/checkpoints``,
    so the row and the seed are both in the path, and they are what you want on
    the file: `pact_sev3_seed0_3000000.mp4`, not a run hash.
    """
    parts = checkpoint.resolve().parts
    seed = next(
        (p for p in reversed(parts) if p.startswith("seed_") and p[5:].isdigit()),
        None,
    )
    if seed is not None:
        index = parts.index(seed)
        if index > 0:
            return f"{parts[index - 1]}_seed{seed[5:]}"
    return checkpoint.parent.parent.name


#  ---------------------------------------------------------------------
#  Writers, best first.  Each returns what it wrote, or None when the library
#  it needs is absent -- none of them is a hard dependency, and the last one
#  needs only numpy, so the render is never lost to a missing codec.
#  ---------------------------------------------------------------------
def write_imageio(frames, stem: pathlib.Path, fps: int):
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            return None
    out = stem.with_suffix(".mp4")
    try:
        #  macro_block_size=1 keeps VMAS's own frame size; the default silently
        #  resizes to a multiple of 16.
        imageio.mimwrite(str(out), frames, fps=fps, macro_block_size=1)
    except Exception as error:      # no ffmpeg plugin, or no usable codec
        print(
            f"   imageio could not write an mp4 ({error.__class__.__name__}), "
            "trying the next writer"
        )
        return None
    return out


def write_cv2(frames, stem: pathlib.Path, fps: int):
    try:
        import cv2
    except ImportError:
        return None
    out = stem.with_suffix(".mp4")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        return None
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return out


def write_gif(frames, stem: pathlib.Path, fps: int):
    try:
        from PIL import Image
    except ImportError:
        return None
    out = stem.with_suffix(".gif")
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / max(fps, 1)),
        loop=0,
    )
    return out


def write_pngs(frames, stem: pathlib.Path, fps: int):
    try:
        from PIL import Image
    except ImportError:
        return None
    folder = stem.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(folder / f"{i:05d}.png")
    pattern = "%05d.png"
    print(
        f"   PNG sequence. Join it with: ffmpeg -framerate {fps} "
        f"-i {folder}/{pattern} -pix_fmt yuv420p {stem}.mp4"
    )
    return folder


def write_npz(frames, stem: pathlib.Path, fps: int):
    import numpy as np

    out = stem.with_suffix(".npz")
    np.savez_compressed(out, frames=np.stack(frames), fps=fps)
    print(
        "   raw frames only -- no video writer was available. The render "
        "worked; `pip install imageio imageio-ffmpeg` and rerun for an mp4."
    )
    return out


WRITERS = (write_imageio, write_cv2, write_gif, write_pngs, write_npz)


def save(frames, stem: pathlib.Path, fps: int) -> pathlib.Path:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for writer in WRITERS:
        out = writer(frames, stem, fps)
        if out is not None:
            return out
    raise SystemExit("no writer succeeded, which numpy should have made impossible")


def render_one(checkpoint: pathlib.Path, args) -> None:
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    from benchmarl.experiment import Experiment

    outdir = pathlib.Path(args.out).resolve()
    scratch = outdir / "_reload"
    scratch.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    device = "cuda" if cuda else "cpu"

    patch = {
        #  Where the reloaded experiment may put its own files.  The pickled
        #  config still names the SERVER's save_folder, which does not exist
        #  here, and BenchMARL mkdirs it with parents=False.
        "save_folder": str(scratch),
        #  A run trained on the cluster holds cuda tensors, and torch.load
        #  raises on a machine with no GPU unless they are mapped.
        "restore_map_location": None if cuda else {"cuda:0": "cpu"},
        "sampling_device": device,
        "train_device": device,
        "buffer_device": device,
        #  The rollout and the frame capture are driven below, so BenchMARL's
        #  own render path stays off -- with it, the frames would go to the
        #  torchrl logger's video writer, which wants torchvision and av.
        "render": False,
        "evaluation": True,
        "loggers": [],
        "create_json": False,
        #  VMAS renders WORLD 0 only, so the test env needs exactly one world:
        #  more evaluation episodes would buy a less noisy mean, not more film.
        "evaluation_episodes": 1,
        "evaluation_deterministic_actions": args.deterministic,
        #  collect_with_grad skips building the 300-worker TRAINING collector
        #  and skips loading its state, which is what makes this a laptop-sized
        #  job and what makes the env width below safe to change.
        #  init_random_frames has to go with it: the two together raise.
        "collect_with_grad": True,
        "on_policy_n_envs_per_worker": 1,
        "off_policy_n_envs_per_worker": 1,
        "off_policy_init_random_frames": 0,
        #  Nothing about a replay should be written back as a checkpoint.
        "checkpoint_interval": 0,
        "checkpoint_at_end": False,
        "exclude_buffer_from_checkpoint": True,
    }

    label = f"{label_of(checkpoint)}_{frames_of(checkpoint)}"
    print(f"\n== {label}")
    print(f"   {checkpoint}")
    experiment = Experiment.reload_from_file(str(checkpoint), experiment_patch=patch)
    frames = []
    try:
        env = experiment.test_env
        if args.static:
            try:
                env.set_seed(experiment.seed)
            except NotImplementedError:
                print("   (the env does not support set_seed; --static ignored)")

        def callback(env, td):
            frames.append(
                experiment.task.__class__.render_callback(experiment, env, td)
            )

        exploration = (
            ExplorationType.DETERMINISTIC
            if args.deterministic
            else ExplorationType.RANDOM
        )
        with set_exploration_type(exploration), torch.no_grad():
            env.rollout(
                max_steps=experiment.max_steps,
                policy=experiment.policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
    finally:
        experiment.close()

    if not frames:
        raise SystemExit(
            "the rollout produced no frames, so render_callback returned "
            "nothing -- that is the renderer, not the writer."
        )
    out = save(frames, pathlib.Path(args.out).resolve() / label, args.fps)
    print(f"   {len(frames)} frames -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("targets", nargs="+", help="run folders or checkpoints")
    parser.add_argument("--out", default="videos", help="where the videos go")
    parser.add_argument(
        "--all",
        action="store_true",
        help="render every checkpoint under the folder, not just the newest",
    )
    parser.add_argument("--fps", type=int, default=20, help="video frame rate")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="auto uses a GPU only if there is one",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="sample actions instead of taking the distribution mode",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="seed the env, so one checkpoint always gives the same clip",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be rendered and exit (needs no torchrl)",
    )
    args = parser.parse_args()

    selected = []
    for target in args.targets:
        selected.extend(resolve(target, args.all))

    if args.dry_run:
        for checkpoint in selected:
            print(f"{label_of(checkpoint)}_{frames_of(checkpoint)}  <-  {checkpoint}")
        return 0

    for checkpoint in selected:
        render_one(checkpoint, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
