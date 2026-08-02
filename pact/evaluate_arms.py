#  PHASE 2 -- the arm report: every arm, every phase of the driver, one table.
#
#  A single scalar hides the whole point of a periodic driver.  The trough is
#  easy for everyone (there the NS is switched off, so a blind policy scores
#  ~100%) and the peak is where the problem lives, so a cycle-average alone
#  understates every gap.  This script freezes the driver at a grid of phases,
#  evaluates each arm at each, and reports:
#
#     * the phase profile        R(A) for A = 0 .. 1
#     * PEAK   (A = 1)           the phase sigma* is defined at, and the headline
#     * CYCLE  (phase-weighted)  the collapse-and-recover average
#
#  Reporting both is the honest framing: a controller is judged by its worst
#  phase, and the cycle-average shows how much of the cycle is free.
#
#  Usage:
#      python pact/evaluate_arms.py \
#          --b0        runs/b0/checkpoints/checkpoint_3000000.pt \
#          --arm blind_ippo=runs/blind_ippo/checkpoints/checkpoint_3000000.pt \
#          --arm blind_mappo=runs/blind_mappo/checkpoints/checkpoint_3000000.pt \
#          --arm pact=runs/pact/checkpoints/checkpoint_3000000.pt \
#          --arm pact_ctde=runs/pact_ctde/checkpoints/checkpoint_3000000.pt \
#          --arm ceiling=runs/oracle/checkpoints/checkpoint_3000000.pt

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.rollout_utils import bootstrap_ci, evaluate, load_experiment, task_from  # noqa: E402


def phase_grid(n_points: int):
    """Driver values at uniformly spaced phases over half a cycle, with the
    trapezoid weights that turn a mean over them into the true cycle average.

    A(phi) = (1 - cos 2*pi*phi) / 2 is symmetric about phi = 1/2, so sampling
    phi in [0, 1/2] and half-weighting the endpoints reproduces the average over
    the whole cycle exactly.
    """
    phis = [k / (2.0 * n_points) for k in range(n_points + 1)]
    values = [0.5 * (1.0 - math.cos(2.0 * math.pi * p)) for p in phis]
    weights = [0.5] + [1.0] * (n_points - 1) + [0.5]
    total = sum(weights)
    return values, [w / total for w in weights]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="repeatable",
    )
    parser.add_argument(
        "--b0",
        type=str,
        default=None,
        help="checkpoint trained with ns_severity=0; used to normalise",
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--phase-points", type=int, default=4)
    args = parser.parse_args()

    arms = []
    for entry in args.arm:
        if "=" not in entry:
            parser.error(f"--arm expects NAME=CHECKPOINT, got {entry!r}")
        name, path = entry.split("=", 1)
        arms.append((name, path))
    if args.b0:
        arms.insert(0, ("B0 (no NS)", args.b0))
    if not arms:
        parser.error("give at least one --arm or --b0")

    drivers, weights = phase_grid(args.phase_points)

    results = {}
    for name, checkpoint in arms:
        experiment = load_experiment(checkpoint, device=args.device)
        try:
            # B0 is defined with the NS switched off entirely, not merely frozen
            # at the trough, so that it is a severity-independent reference.
            freeze_values = [0.0] if name.startswith("B0") else drivers
            row = []
            for driver in freeze_values:
                task = task_from(experiment, ns_freeze_driver=driver)
                returns, extras = evaluate(
                    experiment,
                    task,
                    num_envs=args.episodes,
                    seed=args.seed,
                    device=args.device,
                )
                row.append((returns, extras))
            results[name] = row
        finally:
            experiment.close()

    b0 = None
    if args.b0:
        b0 = float(results["B0 (no NS)"][0][0].mean())

    def pct(value):
        return "" if b0 is None else f" ({100 * value / b0:5.1f}%)"

    print("\n" + "=" * 96)
    print("PHASE PROFILE  (driver frozen; mean episode return, 95% bootstrap CI)")
    print("=" * 96)
    if b0 is not None:
        print(f"B0 = {b0:+.4f}   bar for PACT = {0.9 * b0:+.4f}   bar for blind = {0.3 * b0:+.4f}")
    for name, row in results.items():
        print(f"\n{name}")
        if len(row) == 1:
            returns = row[0][0]
            lo, hi = bootstrap_ci(returns)
            print(f"  NS off : {float(returns.mean()):+.4f} [{lo:+.4f}, {hi:+.4f}]")
            continue
        for driver, (returns, extras) in zip(drivers, row):
            lo, hi = bootstrap_ci(returns)
            mean = float(returns.mean())
            beta = extras.get("pact_beta")
            beta_note = f"  beta={beta:5.3f}" if beta is not None else ""
            print(
                f"  A={driver:4.2f} : {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]{pct(mean)}"
                f"{beta_note}"
            )

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"{'arm':<24} | {'PEAK (A=1)':>20} | {'CYCLE avg':>20}")
    print("-" * 72)
    for name, row in results.items():
        if len(row) == 1:
            mean = float(row[0][0].mean())
            print(f"{name:<24} | {mean:+10.4f}{pct(mean):>10} | {'—':>20}")
            continue
        peak = float(row[-1][0].mean())
        cycle = sum(w * float(r.mean()) for w, (r, _) in zip(weights, row))
        print(
            f"{name:<24} | {peak:+10.4f}{pct(peak):>10} | {cycle:+10.4f}{pct(cycle):>10}"
        )
    print(
        "\nPEAK is the headline: it is the phase sigma* is defined at and the phase a\n"
        "controller is judged by. CYCLE is diluted by the trough, where the driver is\n"
        "off and every arm scores ~100% -- report both."
    )


if __name__ == "__main__":
    main()
