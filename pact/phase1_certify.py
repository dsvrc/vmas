#  PHASE 1 -- certify that the non-stationarity is solvable, and find sigma*.
#
#  sigma* = the largest severity at which a *privileged, scripted* controller
#  holds the NS peak to >= 90% of the undisturbed baseline B0.  A learner can at
#  best match a controller with perfect knowledge, so if the privileged scripted
#  controller fails at sigma, no method can succeed at sigma -- the bottleneck
#  would be the environment, not information or optimisation.  Certify existence
#  before spending compute on a method.
#
#  Learning is OFF here.  A single policy trained with the NS disabled is rolled
#  through a probe that rewrites its action using the environment's true
#  deflection at a hand-set gain.  Because the probe changes no spec, the same
#  B0 checkpoint is reused at every point of the sweep; the whole thing is
#  minutes of evaluation.
#
#  ---------------------------------------------------------------------------
#  Reading the result -- four things, not just sigma*
#  ---------------------------------------------------------------------------
#  * return vs sigma        -- the frontier (where it crosses 0.9 * B0)
#  * saturation vs sigma    -- the bounded resource; should switch on at sigma*
#  * best_beta vs sigma     -- a crossover from "full" to "near zero" is the
#                              loop-gain fingerprint of a category-C NS, where
#                              compensating changes the very thing compensated
#  * the gain-tolerance band -- for this environment's *transform* channel the
#                              inverse is norm-preserving, so the actuator is
#                              nearly never the binding constraint.  What binds
#                              instead is the conditioning of the inverse: how
#                              wrong beta may be and still clear the bar.  That
#                              band, not saturation, is what defines sigma* here,
#                              and the script reports both.
#
#  Usage:
#      # 1. train B0 (the NS off).  Reuse this one checkpoint for the whole sweep.
#      python pact/run.py algorithm=ippo task=vmas_ns/navigation_pcw \
#             task.ns_severity=0 experiment.checkpoint_at_end=true \
#             experiment.render=false
#
#      # 2. certify
#      python pact/phase1_certify.py <path/to/checkpoint_XXXX.pt>

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact.rollout_utils import (  # noqa: E402
    bootstrap_ci,
    evaluate,
    load_experiment,
    task_from,
)

BAR = 0.90


def _probe_factory(task, beta):
    if beta is None:
        return None

    def factory(base_env):
        return [task.make_phase1_probe(base_env, beta=beta)]

    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str, help="B0 checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=40, help="envs per cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--severities", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    )
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="beta as a multiple of the true deflection",
    )
    parser.add_argument(
        "--tolerance-band",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0, 1.1, 1.2],
        help="beta multipliers used for the gain-tolerance read-out",
    )
    parser.add_argument(
        "--target-severity",
        type=float,
        default=None,
        help="severity Phase 2 will train at; defaults to the task yaml's ns_severity",
    )
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}")
    experiment = load_experiment(args.checkpoint, device=args.device)

    def run(severity, beta, freeze=1.0):
        task = task_from(
            experiment,
            ns_severity=severity,
            ns_freeze_driver=freeze,
            pact_enabled=False,
            pact_oracle=False,
        )
        return evaluate(
            experiment,
            task,
            num_envs=args.episodes,
            seed=args.seed,
            device=args.device,
            probe_factory=_probe_factory(task, beta),
        )

    try:
        # ------------------------------------------------------------------
        # B0 and the two confounding checks
        # ------------------------------------------------------------------
        b0_returns, _ = run(severity=0.0, beta=None, freeze=0.0)
        b0 = float(b0_returns.mean())
        bar = BAR * b0
        lo, hi = bootstrap_ci(b0_returns)
        print(f"\nB0 (no NS, no probe)  = {b0:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        print(f"bar = {BAR:.2f} * B0     = {bar:+.4f}")

        print("\nCHECK A -- transparency: the probe at driver = 0 must reproduce B0")
        trans_returns, _ = run(severity=0.0, beta=1.0, freeze=0.0)
        tlo, thi = bootstrap_ci(trans_returns)
        t_mean = float(trans_returns.mean())
        ok_a = abs(t_mean - b0) < 1e-6 or (tlo <= b0 <= thi)
        print(
            f"  probe(beta=1, driver=0) = {t_mean:+.4f}  [{tlo:+.4f}, {thi:+.4f}]  "
            f"{'PASS' if ok_a else 'FAIL -- the probe corrupts the action independently of the NS'}"
        )

        print("\nCHECK B -- it-works-when-it-should: compensation must recover at low sigma")
        low = args.severities[0]
        blind_low, _ = run(severity=low, beta=0.0)
        comp_low, _ = run(severity=low, beta=1.0)
        ok_b = float(comp_low.mean()) > float(blind_low.mean())
        print(
            f"  sigma={low}: blind {float(blind_low.mean()):+.4f} -> "
            f"compensated {float(comp_low.mean()):+.4f} "
            f"({100 * float(comp_low.mean()) / b0:.0f}% of B0)  "
            f"{'PASS' if ok_b else 'FAIL -- law or privileged signal is wrong'}"
        )
        if not (ok_a and ok_b):
            print(
                "\nOne or both confounding checks failed. A high-severity failure "
                "below would NOT be a property of the environment. Fix the probe first."
            )

        # ------------------------------------------------------------------
        # the sweep
        # ------------------------------------------------------------------
        print(
            "\nSWEEP (driver frozen at the PEAK; beta re-optimised per severity --\n"
            "fixing beta across the sweep is what understates sigma*)"
        )
        header = "  ".join(f"b={g:<5.2f}" for g in args.gains)
        print(f"\n{'sigma':>6} | {header} |  best_b |   max_b R  | sat  | >=bar")
        print("-" * (24 + 9 * len(args.gains)))

        sigma_star = None
        for severity in args.severities:
            cells, best, best_beta, best_sat = [], -1e18, None, 0.0
            for beta in args.gains:
                returns, extras = run(severity=severity, beta=beta)
                mean = float(returns.mean())
                cells.append(mean)
                if mean > best:
                    best, best_beta, best_sat = mean, beta, extras.get("sat_frac", 0.0)
            passed = best >= bar
            if passed:
                sigma_star = severity
            print(
                f"{severity:6.2f} | "
                + "  ".join(f"{c:+7.4f}" for c in cells)
                + f" | {best_beta:7.2f} | {best:+10.4f} | {100 * best_sat:3.1f}% | "
                + ("YES" if passed else "no")
            )

        print(f"\nsigma* (largest sigma with max_beta R >= {BAR:.2f}*B0) = {sigma_star}")
        if sigma_star == args.severities[-1]:
            print(
                "  sigma* is at or beyond the top of the grid: the channel inverse is\n"
                "  norm-preserving, so the actuator is not the binding constraint.\n"
                "  Read the gain-tolerance table below for the operational frontier."
            )

        # ------------------------------------------------------------------
        # gain tolerance: the frontier that actually binds for this channel
        # ------------------------------------------------------------------
        print(
            "\nGAIN-TOLERANCE BAND (% of B0 at beta = k * true deflection).\n"
            "The operational sigma* is the largest sigma whose whole band clears the\n"
            "bar, i.e. that survives a realistic tracking error in the one scalar."
        )
        head = "  ".join(f"{k:>6.2f}" for k in args.tolerance_band)
        print(f"{'sigma':>6} | {head}")
        print("-" * (9 + 8 * len(args.tolerance_band)))
        operational = None
        for severity in args.severities:
            cells = []
            for k in args.tolerance_band:
                returns, _ = run(severity=severity, beta=k)
                cells.append(100 * float(returns.mean()) / b0)
            if min(cells) >= 100 * BAR:
                operational = severity
            print(f"{severity:6.2f} | " + "  ".join(f"{c:6.1f}" for c in cells))
        print(f"\noperational sigma* (whole band >= {100 * BAR:.0f}%) = {operational}")

        # ------------------------------------------------------------------
        # the decision Phase 1 forces
        # ------------------------------------------------------------------
        # The severity Phase 2 will TRAIN at, which lives in the task yaml -- not
        # experiment.task.config, which is the B0 checkpoint's own config and is
        # therefore always 0 by construction.
        target = args.target_severity
        if target is None:
            target = float(task_from().config["ns_severity"])
        print(f"\nDECISION: ns_severity configured for training = {target}")
        frontier = operational if operational is not None else sigma_star
        if frontier is not None and target <= frontier:
            print(
                f"  {target} <= sigma* ({frontier}) -> WELL-POSED. Build PACT (Phase 2).\n"
                "  The scripted law just certified IS the method's target and its ceiling."
            )
        else:
            print(
                f"  {target} > sigma* ({frontier}) -> ILL-POSED. Redesign, then re-run:\n"
                "    (a) lower ns_severity to sigma* - 0.05   (one config value)\n"
                "    (b) attenuate the harmful channel (lower ns_gain)\n"
                "    (c) cap the driver so it never exceeds what the law can undo"
            )
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
