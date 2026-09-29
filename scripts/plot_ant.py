#!/usr/bin/env python
"""Plot the MAMuJoCo Ant-v2-4x2 eval return, mean +- std over seeds.

    python scripts/plot_ant.py --root <mamujoco_ns/Ant-v2-4x2> --out plots

One figure.  The styling -- fonts, palette, markers, bands, legend -- is
IMPORTED from ``plot_eval.py`` rather than copied, so the VMAS figures and this
one cannot drift apart.

The tree is one directory per arm, holding one ``progress*.txt`` per seed:

    <root>/<algo>/<arm>/progress.txt      <- seed 0
                        progress_1.txt    <- seed 1
                        ...

each a two-column ``step,value`` file with no header.  B0 and `mappo_NoNS`
are in the tree but are NOT plotted; see ``EXCLUDED``.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import plot_eval as sty  # noqa: E402  -- imports the rcParams and the palette
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

#: ``<algo>/<arm>`` under the root -> how it is drawn.  ``role`` is "ours" for
#: PACT, "ref" for the blind rule, "base" for a baseline.
ARMS = [
    ("mappo/blind_mappo", "MAPPO", "ref"),
    ("pact/pact", "PACT (ours)", "ours"),
    ("happo/blind_happo", "HAPPO", "base"),
    ("haa2c/blind_mappo", "HAA2C", "base"),
    ("hatrpo/blind_hatrpo", "HATRPO", "base"),
]

#: Present in the tree and deliberately NOT drawn, on request.  `mappo_NoNS`
#: is the sigma = 0 arm (it is the same five runs as the old `mappo/b0`), and
#: B0 is the same thing under its other name.  Listed rather than deleted so
#: the omission is visible in the script and reversible in one line.
EXCLUDED = {
    "mappo_NoNS": "the sigma = 0 arm (B0) -- excluded on request",
    "mappo/b0": "the sigma = 0 arm (B0) -- excluded on request",
}

REF_STYLE = {
    "MAPPO": dict(color="#7f7f7f", linestyle=":", lw=1.0, marker=None, ms=0),
}


def read_progress(path: pathlib.Path):
    """``(steps, values)`` from a two-column ``step,value`` file."""
    steps, values = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace("	", ",").split(",")
        if len(parts) < 2:
            continue
        try:
            steps.append(float(parts[0]))
            values.append(float(parts[1]))
        except ValueError:      # a header line, if one is ever written
            continue
    if not steps:
        return None
    return np.asarray(steps), np.asarray(values)


def collect(root: pathlib.Path):
    """``{label: {seed: (steps, values)}}``.

    A seed is the ``progress*.txt`` index: ``progress.txt`` is 0,
    ``progress_1.txt`` is 1, and so on.  They are separate runs of the same
    arm, which is all the aggregation needs.
    """
    found = defaultdict(dict)
    notes = []
    for rel, label, _role in ARMS:
        arm_dir = root / rel
        if not arm_dir.is_dir():
            notes.append(f"{rel}: not in the tree")
            continue
        files = sorted(arm_dir.glob("progress*.txt"))
        if not files:
            notes.append(f"{rel}: no progress*.txt")
            continue
        for path in files:
            match = re.search(r"progress_(\d+)", path.name)
            seed = int(match.group(1)) if match else 0
            curve = read_progress(path)
            if curve is None:
                notes.append(f"{rel}/{path.name}: unreadable, skipped")
                continue
            found[label][seed] = curve
    for rel, why in EXCLUDED.items():
        if (root / rel).exists():
            notes.append(f"{rel}: NOT PLOTTED -- {why}")
    return found, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=r"C:\Users\chinnu\Downloads\mamujoco_ns\mamujoco_ns\Ant-v2-4x2",
        help="the Ant results tree",
    )
    parser.add_argument("--out", default="plots", help="where the figure goes")
    parser.add_argument("--title", default="MAMuJoCo Ant-v2 4x2")
    parser.add_argument("--xlabel", default="Number of timesteps")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 2

    found, notes = collect(root)
    if not found:
        print(f"no eval curves anywhere under {root}", file=sys.stderr)
        return 2

    print(f"== Ant-v2-4x2 under {root} ==")
    for _rel, label, _role in ARMS:
        if label not in found:
            continue
        seeds = sorted(found[label])
        last = max(c[0][-1] for c in found[label].values())
        print(f"  {label:<14} {len(seeds)} seed(s) {seeds}  last step {last:,.0f}")
    for note in notes:
        print(f"  {note}")

    longest = max(c[0][-1] for arm in found.values() for c in arm.values())
    grid = np.linspace(0.0, longest, 120)

    fig, ax = plt.subplots(figsize=(3.5, 2.9), layout="constrained")
    handles, colour = [], 0
    ours = None
    for _rel, label, role in ARMS:
        if label not in found:
            continue
        curves = found[label]
        mean, std, _ = sty.aggregate(curves, grid)
        if role == "ours":
            style = dict(sty.OURS_STYLE)
        elif role == "ref":
            style = dict(REF_STYLE[label])
        else:
            style = dict(
                color=sty.COLORS[colour % len(sty.COLORS)],
                marker=sty.MARKERS[colour % len(sty.MARKERS)],
                lw=1.0, ms=3.2,
            )
            colour += 1
        entry = (grid, mean, std, style, role == "ours", len(curves))
        if role == "ours":
            ours = entry               # drawn last, on top
        else:
            sty.draw(ax, *entry, markevery=6)
        handles.append(
            Line2D([], [], label=label, **{
                k: v for k, v in style.items() if k in
                ("color", "marker", "lw", "ms", "linestyle")
            }, **({} if role == "ours" else
                  {"mfc": "white", "mec": style["color"], "mew": 0.8}))
        )
    if ours is not None:
        sty.draw(ax, *ours, markevery=6)

    ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel("Evaluation return")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.02)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(6.5)

    #  Ours first in the key, as in the VMAS figures.
    handles.sort(key=lambda h: h.get_label() != "PACT (ours)")
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False, handlelength=2.0, columnspacing=1.4,
               handletextpad=0.5)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"eval_return_ant.{ext}", dpi=args.dpi)
    plt.close(fig)
    print(f"  wrote {outdir / 'eval_return_ant'}.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
