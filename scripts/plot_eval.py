#!/usr/bin/env python
"""Plot eval return vs training iteration, mean +- std over seeds.

    python scripts/plot_eval.py --root <vmas_results> --out <dir>

Reads every ``scalars/eval_reward_episode_reward_mean.csv`` under the results
tree, groups the curves by arm and seed, and draws two figures so that ~20
curves do not land on one axis.  Both figures carry the SAME two references --
B0 (no non-stationarity) and PACT -- so either one can be read on its own.

Layout it expects, and the two quirks in the tree it was written for:

  * the references live under ``<root>/{b0,pact,no_pact}/<same again>/...``;
  * the baselines under ``<root>/balance/balance/sigma<S>/s<N>/<arm>/...``,
    sometimes with a duplicated ``s<N>/s<N>/`` level and sometimes with the
    run folder repeated -- both are collapsed, and a genuine
    (arm, seed) collision is reported rather than silently averaged.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

CSV_NAME = "eval_reward_episode_reward_mean.csv"

#: Every top-level directory that holds a REFERENCE rather than a baseline.
#: `b0` is severity-independent (it is the no-non-stationarity run), so it is
#: the same curve on every figure; PACT and blind MAPPO have one directory per
#: severity.
#: Where the reference arms live, and which severity each one belongs to.
#: `b0` is the sigma = 0 arm: it is the same run for every panel, so it has no
#: severity of its own.  `pact_sev1` / `blind_sev1` are the sigma = 1
#: counterparts of `pact` / `no_pact` and are relabelled as those, one
#: severity down, so a panel asks for "pact" and gets the right one.
REFERENCE_DIRS = {
    "b0": ("b0", None),
    "pact": ("pact", "3.0"),
    "no_pact": ("no_pact", "3.0"),
    "pact_sev1": ("pact", "1.0"),
    "blind_sev1": ("no_pact", "1.0"),
}
REFERENCES = ["b0", "pact", "no_pact"]

#: EVERY arm is keyed by severity, because the results tree holds sigma = 1
#: and sigma = 3 runs of the same algorithm under the same folder name in
#: different roots.  Merging them would average two different experiments and
#: nothing downstream would notice.
def key(name, sigma):
    """The internal arm key: severity-qualified, except for sigma = 0's B0."""
    return "b0" if name == "b0" else f"{name}@{sigma}"


#: The two halves the baselines are split into.  The list is the CANONICAL
#: order for a set: an arm keeps its colour and marker whether or not it was
#: run at a given severity, so the sigma = 1 and sigma = 3 panels of one set
#: can share a key.
SETS = [
    (
        "set1",
        #  "liam" is deliberately absent: that sweep was stopped at iteration
        #  15 of 100 on cost, so its curve would end a seventh of the way
        #  across and read as a failure rather than as an unfinished run.
        #  "hasac" is dropped on request.  NOTE for the write-up: unlike the
        #  two below, this is a COMPLETE result (5 seeds, 100 iterations) that
        #  was removed because the method does badly here, not because the run
        #  is broken.  If the table claims to show the published baselines,
        #  that omission should be stated.
        ["happo", "mappo_gru", "ippo_gru", "mfac", "mfac_team", "ernie"],
    ),
    (
        "set2",
        #  "lcpo" is dropped on request.  Its sigma = 1 sweep is the broken
        #  configuration -- flat at the initial return on all five seeds,
        #  which is the trust region never engaging -- so leaving it out there
        #  is the honest call.  Its sigma = 3 curve is the GOOD `lcpo_1`
        #  rerun and reaches ~35; put "lcpo" back in this list to restore
        #  both panels at once.
        ["dr_sigma", "oracle_driver_blind", "eso", "rls_raw",
         "qcd_glr", "dedafp", "ipga"],
    ),
]

#: ONE style per arm, across BOTH sets.  A 1x4 figure carries a single key,
#: so two arms may not share a (colour, marker) pair -- with a per-set cycle
#: HASAC and DR would both be an orange square and the key would be wrong.
#: Cycling the marker independently of the colour gives 9 x 9 distinct pairs,
#: which is more than the fifteen arms need.
def _build_styles():
    out, index = {}, 0
    for _, arms in SETS:
        for arm in arms:
            out[arm] = dict(
                color=COLORS[index % len(COLORS)],
                marker=MARKERS[(index // len(COLORS) + index) % len(MARKERS)],
                lw=1.0,
                ms=3.2,
            )
            index += 1
    return out


#: Severity, and how it is written in a title.
SEVERITIES = ["3.0", "1.0"]
SEV_LABEL = {"3.0": r"$\sigma$ = 3", "1.0": r"$\sigma$ = 1"}

SUPERSEDES = {"lcpo_1": "lcpo"}

#: Short forms.  Each is either the paper's OWN name for the method or a
#: standard abbreviation of it, so the figure can be read against the text
#: without a glossary: R-MAPPO / R-IPPO are Yu et al.'s names for the recurrent
#: variants, ESO is the standard abbreviation for an extended state observer,
#: RLS for recursive least squares, DR for domain randomization.  Oracle-A(t)
#: is the information grant: a stock learner handed the driver A(t).
PRETTY = {
    "b0": "B0",
    "pact": "PACT",
    "no_pact": "MAPPO",
    "pact_sev1": "PACT",
    "blind_sev1": "MAPPO",
    "happo": "HAPPO",
    "hasac": "HASAC",
    "mappo_gru": "R-MAPPO",
    "ippo_gru": "R-IPPO",
    "mfac": "MF-AC",
    "mfac_team": "MF-AC-T",
    "liam": "LIAM",
    "ernie": "ERNIE",
    "lcpo": "LCPO",
    "dr_sigma": "DR",
    "oracle_driver_blind": "Oracle-A(t)",
    "eso": "ESO",
    "rls_raw": "RLS",
    "qcd_glr": "QCD+",
    "dedafp": "DEDA-FP",
    "ipga": "IPGA",
}

#: Times, and big enough to survive a two-column shrink.  DejaVu is the
#: fallback if Times New Roman is not installed, which is what a Linux box
#: without mscorefonts will hit.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    #  Type 42 so the text in the PDF stays text: a camera-ready check will
    #  reject Type 3.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
})

#: The reference script's palette and marker cycle, with ONE removal: its
#: "#7f7f7f" grey is the colour MAPPO-blind is drawn in below, and two curves
#: in the same grey on a results figure is a correctness problem rather than a
#: cosmetic one.  Everything else is in the reference's own order.
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
          "#e377c2", "#bcbd22", "#17becf", "#393b79"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">"]

#: PACT is "ours": the reference script's OURS_STYLE, verbatim.
OURS = "pact"
OURS_STYLE = dict(color="#d62728", marker="*", lw=1.6, ms=5.5)

#: The two arms that are neither ours nor a baseline.  Drawn thin and
#: unmarked so they read as rules rather than as competitors.
REF_STYLE = {
    "b0": dict(color="#000000", linestyle="--", lw=1.0, marker=None, ms=0),
    "no_pact": dict(color="#7f7f7f", linestyle=":", lw=1.0, marker=None, ms=0),
    "pact_sev1": dict(**OURS_STYLE),
    "blind_sev1": dict(color="#7f7f7f", linestyle=":", lw=1.0, marker=None, ms=0),
}


STYLES = _build_styles()


def read_curve(path: pathlib.Path):
    """``(steps, values)`` from a two-column BenchMARL scalar CSV."""
    steps, values = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                steps.append(float(row[0]))
                values.append(float(row[1]))
            except ValueError:  # a header line, if one is ever written
                continue
    return np.asarray(steps), np.asarray(values)


def classify(path: pathlib.Path, root: pathlib.Path):
    """``(arm_key, seed)`` for one CSV, or ``None`` if unrecognised.

    ``arm_key`` is severity-qualified -- see :func:`key`.
    """
    rel = path.relative_to(root).as_posix()

    for folder, (name, sigma) in REFERENCE_DIRS.items():
        if rel.startswith(f"{folder}/"):
            match = re.search(r"/seed_(\d+)/", rel)
            if match:
                return key(name, sigma), int(match.group(1))
            #  A reference run that predates the seed_<N> layout.  `pact` and
            #  `no_pact` hold BOTH: five seed_<N> folders from 26-09-22 and a
            #  handful of older top-level run folders, which are DIFFERENT
            #  runs, not copies.  Mixing them would report 13 "seeds" for pact
            #  and average two generations of the method together.  So the
            #  older ones are tagged and `collect` drops them for any
            #  reference that has a seed_<N> layout; `b0`, `pact_sev1` and
            #  `blind_sev1` have none, so their top-level runs are what they
            #  use.
            match = re.search(r"__([0-9a-f]{8})_", rel)
            tag = match.group(1) if match else rel
            return key(name, sigma), "legacy:" + tag

    #  Any tree, any nesting.  The seed folder is written THREE ways in this
    #  results tree -- "s0/", a bare "0/", and a "seed_0/" level inside the
    #  arm -- so the arm is taken as the folder immediately before
    #  "seed_<N>/", which is the one level BenchMARL itself writes and the
    #  only one that is stable.  The older layout, which has no "seed_<N>",
    #  falls through to the positional rule.
    sigma_match = re.search(r"sigma([\d.]+)/", rel)
    if sigma_match:
        sigma = sigma_match.group(1)
        tail = rel[sigma_match.end():]
        match = re.search(r"([A-Za-z0-9_]+)/seed_(\d+)/", tail)
        if match:
            return key(match.group(1), sigma), int(match.group(2))
        match = re.match(r"s?(\d+)/(?:s?\d+/)?([A-Za-z0-9_]+)/", tail)
        if match:
            return key(match.group(2), sigma), int(match.group(1))
    return None


def collect(root: pathlib.Path):
    """``{arm: {seed: (steps, values)}}``, de-duplicated."""
    found = defaultdict(dict)
    dupes, unknown = [], []
    for path in sorted(root.rglob(CSV_NAME)):
        what = classify(path, root)
        if what is None:
            unknown.append(path)
            continue
        arm, seed = what
        steps, values = read_curve(path)
        if steps.size == 0:
            continue
        if seed in found[arm]:
            #  The tree contains the same run under two nested paths.  Keep the
            #  LONGER curve; if they are the same length they are the same run
            #  and it does not matter which.
            old_steps, _ = found[arm][seed]
            if steps.size <= old_steps.size:
                continue
            dupes.append(f"{arm}/seed {seed}")
        found[arm][seed] = (steps, values)

    #  A rerun replaces the arm it supersedes, under the arm's own name, AT
    #  THE SAME SEVERITY.  The keys carry an "@<sigma>" suffix, so the map is
    #  applied to the bare name and the suffix is carried across; otherwise
    #  the substitution silently stops firing the moment severities are
    #  namespaced, and the superseded (flat) LCPO sweep comes back.
    replaced = {}
    for arm_key in list(found):
        name, _, sigma = arm_key.partition("@")
        if name not in SUPERSEDES:
            continue
        target = SUPERSEDES[name] + ("@" + sigma if sigma else "")
        replaced[target] = (len(found.get(target, {})), len(found[arm_key]))
        found[target] = found.pop(arm_key)

    #  Drop the pre-seed_<N> runs of any reference that also has seed_<N>
    #  folders -- see `classify`.
    dropped = {}
    for arm, curves in found.items():
        legacy = [s for s in curves if isinstance(s, str) and s.startswith("legacy:")]
        modern = [s for s in curves if not (isinstance(s, str) and s.startswith("legacy:"))]
        if legacy and modern:
            for seed in legacy:
                del curves[seed]
            dropped[arm] = len(legacy)
    return found, dupes, unknown, dropped, replaced


def aggregate(curves, grid):
    """Mean and std over seeds on a common grid, by linear interpolation.

    Each seed's curve is clipped to its own last logged step -- a seed that
    stopped early contributes nothing beyond where it stopped, rather than a
    flat extrapolation that would look like a converged run.
    """
    stack = []
    for steps, values in curves.values():
        order = np.argsort(steps)
        steps, values = steps[order], values[order]
        interp = np.interp(grid, steps, values)
        interp[grid > steps[-1]] = np.nan
        interp[grid < steps[0]] = np.nan
        stack.append(interp)
    stack = np.vstack(stack)
    with np.errstate(invalid="ignore"):
        count = np.sum(~np.isnan(stack), axis=0)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
    mean[count == 0] = np.nan
    std[count < 2] = 0.0
    return mean, std, count


def draw(ax, xs, mean, std, style, is_ours, n_seeds, markevery):
    """One curve, in the reference script's `draw` shape."""
    z = 10 if is_ours else 3
    if n_seeds > 1:
        ax.fill_between(
            xs, mean - std, mean + std,
            color=style["color"], alpha=0.16 if is_ours else 0.07,
            lw=0, zorder=z - 1,
        )
    ax.plot(
        xs, mean,
        color=style["color"], lw=style["lw"], zorder=z,
        linestyle=style.get("linestyle", "-"),
        marker=style["marker"], ms=style["ms"], markevery=markevery,
        mfc=style["color"] if is_ours else "white",
        mec=style["color"], mew=0.8,
    )


def panel(ax, found, grid, scale, sigma, canonical, title, xlabel,
          markevery, show_xlabel=True, show_ylabel=True):
    """Draw one axis and return its legend handles, ours first.

    ``canonical`` is the set's FULL arm list, not just the arms that were run
    at this severity: colours and markers are assigned from it, so an arm keeps
    its style whether or not it appears in a given panel and the two severity
    rows of one set can share a key.
    """
    styles = STYLES
    refs = [r for r in REFERENCES if key(r, sigma) in found]
    arms = [a for a in canonical if key(a, sigma) in found]
    xs = grid * scale

    #  References first (they sit under everything), then the baselines, then
    #  ours on top -- which is what the zorders already enforce, but the draw
    #  order keeps the overlaps clean too.
    for name in refs:
        if name == OURS:
            continue
        curves = found[key(name, sigma)]
        mean, std, _ = aggregate(curves, grid)
        draw(ax, xs, mean, std, REF_STYLE[name], False, len(curves), markevery)
    for arm in arms:
        curves = found[key(arm, sigma)]
        mean, std, _ = aggregate(curves, grid)
        draw(ax, xs, mean, std, styles[arm], False, len(curves), markevery)
    if OURS in refs:
        curves = found[key(OURS, sigma)]
        mean, std, _ = aggregate(curves, grid)
        draw(ax, xs, mean, std, OURS_STYLE, True, len(curves), markevery)

    if title:
        ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("Evaluation return")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.02)
    if scale != 1:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(6.5)

    handles = []
    if OURS in refs:
        handles.append(Line2D([], [], label=f"{PRETTY[OURS]} (ours)",
                              **OURS_STYLE))
    for name in refs:
        if name == OURS:
            continue
        st = REF_STYLE[name]
        handles.append(Line2D([], [], color=st["color"], lw=st["lw"],
                              linestyle=st.get("linestyle", "-"),
                              label=PRETTY.get(name, name)))
    for arm in arms:
        st = styles[arm]
        handles.append(Line2D([], [], color=st["color"], marker=st["marker"],
                              lw=st["lw"], ms=st["ms"], mfc="white",
                              mec=st["color"], mew=0.8,
                              label=PRETTY.get(arm, arm)))
    return handles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=r"C:\Users\chinnu\Downloads\vmas_results",
        help="the results tree",
    )
    parser.add_argument("--out", default="plots", help="where the figures go")
    parser.add_argument(
        "--xlabel", default="Number of timesteps", help="x-axis label"
    )
    parser.add_argument(
        "--frames-per-iter",
        type=int,
        default=30000,
        help=(
            "environment frames per logged iteration, so the x axis reads in "
            "TIMESTEPS rather than iterations. 30000 is what every run in this "
            "tree logged (on_policy_collected_frames_per_batch in "
            "hparams0.txt). Pass 0 to plot the raw iteration index instead."
        ),
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        print(f"no such directory: {root}", file=sys.stderr)
        return 2

    found, dupes, unknown, dropped, replaced = collect(root)
    if not found:
        print(f"no {CSV_NAME} anywhere under {root}", file=sys.stderr)
        return 2

    print(f"== {sum(len(v) for v in found.values())} curves under {root} ==")
    for arm in sorted(found):
        seeds = sorted(found[arm], key=str)
        last = max(curve[0][-1] for curve in found[arm].values())
        print(f"  {arm:<22} {len(seeds)} seed(s) {seeds}  last step {last:.0f}")
    if dupes:
        print("  de-duplicated (same run under two paths):", ", ".join(dupes))
    for older, (n_old, n_new) in sorted(replaced.items()):
        base, _, sigma = older.partition("@")
        newer = next(k for k, v in SUPERSEDES.items() if v == base)
        newer += "@" + sigma if sigma else ""
        print(
            f"  {older}: REPLACED by {newer} ({n_new} seeds); the earlier "
            f"{n_old}-seed sweep is not plotted"
        )
    for arm, n in sorted(dropped.items()):
        print(
            f"  {arm}: IGNORED {n} older top-level run(s) -- this arm has a "
            "seed_<N> layout and those are a different generation"
        )
    if unknown:
        print(f"  {len(unknown)} path(s) not recognised, e.g. {unknown[0]}")

    last_common = min(
        max(curve[0][-1] for curve in found[arm].values()) for arm in found
    )
    longest = max(
        max(curve[0][-1] for curve in found[arm].values()) for arm in found
    )
    grid = np.linspace(0.0, longest, 120)
    markevery = 6
    scale = args.frames_per_iter or 1
    print(
        f"  x grid 0 .. {longest:.0f} (every arm reaches at least "
        f"{last_common:.0f})"
    )

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    def title_of(sigma, set_name):
        return f"VMAS-balance {SEV_LABEL[sigma]} ({set_name})"

    def has_data(sigma, canonical):
        return any(key(a, sigma) in found for a in canonical) or any(
            key(r, sigma) in found for r in REFERENCES
        )

    def save(fig, stem):
        for ext in ("pdf", "png"):
            fig.savefig(outdir / f"eval_return_{stem}.{ext}", dpi=args.dpi)
        print(f"  wrote {outdir / ('eval_return_' + stem)}.{{pdf,png}}")

    def stem_of(sigma, set_name):
        return f"sigma{sigma.rstrip('0').rstrip('.')}_{set_name}"

    # ---- one figure per (severity, set), single column ------------------
    for sigma in SEVERITIES:
        for set_name, canonical in SETS:
            if not has_data(sigma, canonical):
                print(f"  {title_of(sigma, set_name)}: nothing in the tree")
                continue
            fig, ax = plt.subplots(figsize=(3.5, 2.9), layout="constrained")
            handles = panel(ax, found, grid, scale, sigma, canonical,
                            title_of(sigma, set_name), args.xlabel, markevery)
            fig.legend(handles=handles, loc="outside lower center", ncol=3,
                       frameon=False, handlelength=2.0, columnspacing=1.4,
                       handletextpad=0.5)
            save(fig, stem_of(sigma, set_name))
            plt.close(fig)

    # ---- all four panels in ONE ROW -------------------------------------
    #  1 x 4 at the full text width, which is the reference script's own
    #  geometry.  One key for the whole figure: that is only unambiguous
    #  because STYLES is global, so no two arms share a (colour, marker) pair
    #  even though they live in different sets.
    order = [(sigma, set_name, canonical)
             for sigma in SEVERITIES
             for set_name, canonical in SETS
             if has_data(sigma, canonical)]
    if len(order) >= 2:
        fig, axs = plt.subplots(
            1, len(order), figsize=(7.16, 3.4), layout="constrained"
        )
        axs = np.atleast_1d(axs)
        merged, seen = [], set()
        for column, (ax, (sigma, set_name, canonical)) in enumerate(
            zip(axs, order)
        ):
            for handle in panel(
                ax, found, grid, scale, sigma, canonical,
                f"({'abcd'[column]}) {title_of(sigma, set_name)}",
                args.xlabel, markevery,
                #  every panel is the same metric, so one y label on the left
                #  is enough and the other three are wasted width
                show_ylabel=(column == 0),
            ):
                if handle.get_label() not in seen:
                    seen.add(handle.get_label())
                    merged.append(handle)
        fig.legend(handles=merged, loc="outside lower center", ncol=6,
                   frameon=False, handlelength=2.0, columnspacing=1.4,
                   handletextpad=0.5)
        save(fig, "all")
        plt.close(fig)

    claimed = {key(r, sigma) for r in REFERENCES for sigma in SEVERITIES}
    claimed |= {key(a, sigma) for _, c in SETS for a in c for sigma in SEVERITIES}
    claimed.add("b0")
    extra = sorted(set(found) - claimed)
    if extra:
        print(f"  NOT PLOTTED (no panel claims them): {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
