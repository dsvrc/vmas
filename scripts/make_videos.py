#!/usr/bin/env python
"""Render evaluation videos from a finished run's checkpoint.

    python scripts/make_videos.py <run folder or checkpoint.pt> [...]

Why the `videos/` folder of your existing runs is EMPTY: every launcher in this
repo passes ``experiment.render=false`` (see scripts/baselines_common.sh's
COMMON array), because rendering every evaluation during a 3 M-frame sweep
costs wall-clock and nobody watches it.  The folder is created by the CSV
logger regardless, which is why it exists and is empty.

So a video is not something you recover from a finished run's logs -- there are
no frames in them.  It has to be re-rendered by replaying the trained policy,
which is what this script does: reload the experiment from its checkpoint with
``render=true``, run ONE evaluation, write the video, and stop.  No training
happens and nothing in the run folder is overwritten except the new video.

Each argument is either

  * a run folder   .../mappo_balance_mlp__<hash>_<date>/   (the one holding
    config.pkl), in which case the newest checkpoint under it is used, or
  * a checkpoint   .../checkpoints/checkpoint_3000000.pt   directly.

HEADLESS MACHINES.  VMAS renders through pyglet, which needs a display.  On a
cluster login or compute node, run this under xvfb:

    xvfb-run -a python scripts/make_videos.py <run folder>

If ``xvfb-run`` is not available, ``PYGLET_HEADLESS=1`` works on some builds.
The script says which of the two it is using, and fails with that advice
rather than a pyglet traceback.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))


def newest_checkpoint(folder: pathlib.Path) -> pathlib.Path:
    """The highest-frame checkpoint under ``folder``."""
    checkpoints = list(folder.glob("checkpoints/checkpoint_*.pt"))
    if not checkpoints:
        #  The run folder may have been given one level up or down.
        checkpoints = list(folder.glob("**/checkpoints/checkpoint_*.pt"))
    if not checkpoints:
        raise SystemExit(
            f"no checkpoints/checkpoint_*.pt under {folder}.\n"
            "A video needs the trained weights, and they are only written when "
            "the run had experiment.checkpoint_at_end=true (the launchers in "
            "this repo do set it). Check you are pointing at the folder that "
            "holds config.pkl."
        )

    def frames(path: pathlib.Path) -> int:
        stem = path.stem.rsplit("_", 1)[-1]
        return int(stem) if stem.isdigit() else -1

    return max(checkpoints, key=frames)


def resolve(target: str) -> pathlib.Path:
    path = pathlib.Path(target).expanduser().resolve()
    if path.is_file() and path.suffix == ".pt":
        return path
    if not path.is_dir():
        raise SystemExit(f"not a folder or a .pt checkpoint: {path}")
    return newest_checkpoint(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", help="run folders or checkpoints")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help=(
            "evaluation episodes, which is also the test environment's batch "
            "size. VMAS renders WORLD 0 only, so this does not give you more "
            "video -- it gives you a noisier mean alongside it. Keep it at 1 "
            "for a clean, fast render."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="act at the distribution mode instead of sampling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be rendered and exit",
    )
    args = parser.parse_args()

    #  A GL CONTEXT, NOT A MONITOR.  pyglet needs an X connection to make an
    #  OpenGL context even for an invisible window, so "I do not need to watch
    #  it" does not remove the requirement -- the driver needs a surface to
    #  draw on.  Nothing is shown either way; the frames go to an mp4.
    if not os.environ.get("DISPLAY"):
        if shutil.which("xvfb-run"):
            print(
                "no DISPLAY -> re-running under xvfb-run (an in-memory "
                "display; nothing is shown, the mp4 is still written)"
            )
            os.execvp(
                "xvfb-run",
                [
                    "xvfb-run",
                    "-a",
                    "--server-args=-screen 0 1400x900x24",
                    sys.executable,
                    *sys.argv,
                ],
            )
        print(
            "no DISPLAY and no xvfb-run -> trying pyglet's EGL headless path.",
            "  If this fails with a GL/EGL error, get Xvfb onto the node:",
            "      module load xorg-server",
            "      conda install -c conda-forge xorg-x11-server-xvfb-cos7-x86_64",
            "  or start one by hand, which is what VMAS's own render()",
            "  docstring prescribes:",
            "      Xvfb :99 -screen 0 1400x900x24 &  export DISPLAY=:99.0",
            sep="\n",
        )
        os.environ["PYGLET_HEADLESS"] = "1"
    else:
        print(f"rendering onto DISPLAY={os.environ['DISPLAY']}")

    #  Imported HERE, not at module scope, so --dry-run works on a machine
    #  without torchrl -- which is the machine you check the paths on
    #  before submitting the job that renders them.
    if not args.dry_run:
        from benchmarl.experiment import Experiment

    for target in args.targets:
        checkpoint = resolve(target)
        folder = checkpoint.parent.parent
        print(f"\n== {folder.name}\n   checkpoint {checkpoint.name}")
        if args.dry_run:
            continue
        patch = {
            #  THE point of this script.  Everything else is left exactly as
            #  the run had it, so the policy being filmed is the policy that
            #  produced the numbers.
            "render": True,
            "evaluation": True,
            "loggers": ["csv"],
            "evaluation_episodes": args.episodes,
            "evaluation_deterministic_actions": args.deterministic,
        }
        experiment = Experiment.reload_from_file(
            str(checkpoint), experiment_patch=patch
        )
        try:
            experiment.evaluate()
        finally:
            experiment.close()
        videos = list(folder.glob("**/videos/*"))
        if videos:
            for video in sorted(videos):
                print(f"   wrote {video}")
        else:
            print(
                "   NO video file appeared. The evaluation ran, so this is the "
                "writer, not the policy: torchrl's CSV logger needs torchvision "
                "and a video backend (`av`). `pip install av torchvision` and "
                "rerun."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
