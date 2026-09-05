#!/usr/bin/env python
#  PART C -- the ceiling decomposition.  Run this FIRST, before any method code.
#
#      python pact2/ceiling.py
#      python pact2/ceiling.py --csv runs/ceiling.csv --agents 1 3 6 9 12
#
#  torch only.  No simulator, no training, minutes of work -- and it bounds
#  everything downstream.  ``PACT_PIPELINE_SPEC`` 11.1: "If the coordination gap
#  is small, this environment is a poor showcase; say so and pick another."
#
#  Under the dial each agent's loading exceeds its sigma=0 counterfactual by
#  ``Delta_i = u_i (1 - g)``, and every unit of that traces to a contributor
#  that partitions by WHO CAN MOVE IT:
#
#      irreducible              = Delta_fixed / Delta_total    nobody
#      own (free)               = Delta_own   / Delta_total    any policy, alone
#      COORDINATION GAP         = Delta_peer  / Delta_total    <- what PACT claims
#
#  Use it to CHOOSE the environment, not to excuse the outcome.

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch

from pact2._bootstrap import load_cores, surrogate_rollout

slc_core, _ = load_cores()


def decompose(params, n_agents: int, steps: int, batch: int, seed: int) -> Dict:
    """Roll the surrogate fleet, then attribute the excess with the declared
    operator.  The Phi trajectory comes from VMAS's own dynamics driven by a
    strong velocity servo, so the attribution is against a realistic operating
    scale rather than an assumed one."""
    roll = surrogate_rollout(
        slc_core, params, n_agents, steps=steps, batch=batch, seed=seed
    )
    op = roll["operator"]
    T, B = roll["phi"].shape[0], roll["phi"].shape[1]
    phi = roll["phi"].reshape(T * B, n_agents)
    g = roll["g_chan"].reshape(T * B, op.n_chan)
    d = slc_core.decompose_excess(phi, g, op)
    return {
        "irreducible": float(d["irreducible"]),
        "own_free": float(d["own_free"]),
        "coordination_gap": float(d["coordination_gap"]),
        "decentralized_ceiling": float(d["decentralized_ceiling"]),
        "non_coordinating_ceiling": float(d["non_coordinating_ceiling"]),
        "u_mean": float(roll["u"].mean()),
        "u_p95": float(roll["u"].reshape(-1).quantile(0.95)),
        "harm_mean": float(roll["c"].mean()),
        "speed_mean": float(roll["speed"].mean()),
        "g_mean": float(roll["g"].mean()),
        "phi_cv": float(phi.std(unbiased=False) / phi.mean()),
        "W_spread": op.spread(),
        "W_asym": op.asymmetry(),
        "dead_agents": float(op.summary()["agents_without_coupling"]),
    }


def _params(**kw):
    d = dict(severity=1.0, n_chan=3)
    d.update(kw)
    return slc_core.SlcParams(**d)


HEAD = (
    f"{'':22s} {'irred':>7s} {'own':>7s} {'PEER':>7s} "
    f"{'dec.ceil':>9s} {'u_mean':>7s} {'harm':>6s} {'g':>6s} {'phiCV':>6s}"
)


def row(label: str, r: Dict) -> str:
    return (
        f"{label:22s} {r['irreducible']:6.1%} {r['own_free']:6.1%} "
        f"{r['coordination_gap']:6.1%} {r['decentralized_ceiling']:8.1%} "
        f"{r['u_mean']:7.3f} {r['harm_mean']:6.1%} {r['g_mean']:6.3f} "
        f"{r['phi_cv']:6.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, nargs="+", default=[1, 3, 6, 9, 12])
    ap.add_argument("--severities", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--channels", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    rows: List[Dict] = []

    print("=" * 96)
    print("PART C -- the ceiling decomposition   (torch only, no simulator, no training)")
    print("=" * 96)

    print("\n1. SEVERITY.  The gap widens with sigma: the peer term scales as 1/g.")
    print("   POWER measured 6.3% -> 9.5% -> 12.7% across the same three rows.\n")
    print(HEAD)
    for s in args.severities:
        r = decompose(_params(severity=s), 6, args.steps, args.batch, args.seed)
        r.update(dict(sweep="severity", severity=s, n_agents=6, n_chan=3))
        rows.append(r)
        tag = "  (beyond-physical)" if s > 1.0 else ""
        print(row(f"sigma={s:<4g} N=6{tag}", r))

    print(
        "\n2. FLEET SIZE.  C.4's falsifiable prediction: the coordination gap"
        "\n   GROWS with N, because a finer partition leaves each agent less local"
        "\n   authority over its own binding link.  No competing credit-assignment"
        "\n   method predicts this, and it costs no training to test.\n"
    )
    print(HEAD)
    for n in args.agents:
        r = decompose(_params(severity=1.0), n, args.steps, args.batch, args.seed)
        r.update(dict(sweep="n_agents", severity=1.0, n_agents=n, n_chan=3))
        rows.append(r)
        note = "   <- N=1: the peer sum is EMPTY" if n == 1 else ""
        print(row(f"N={n:<3d} sigma=1{note}", r))

    print(
        "\n3. FREQUENCY PLAN.  The same knob from the other side: more channels"
        "\n   means fewer co-channel peers per agent, so more of the damage is"
        "\n   self-inflicted and the gap shrinks.  This is the environment being"
        "\n   MEASURED, not tuned -- report the whole grid, never one point.\n"
    )
    print(HEAD)
    for c in args.channels:
        p = _params(severity=1.0, n_chan=c)
        r = decompose(p, 6, args.steps, args.batch, args.seed)
        r.update(dict(sweep="n_chan", severity=1.0, n_agents=6, n_chan=c))
        rows.append(r)
        dead = int(r["dead_agents"])
        note = f"   <- {dead} agents with NO live coupling" if dead else ""
        print(row(f"n_chan={c:<2d} N=6{note}", r))

    ref = [r for r in rows if r["sweep"] == "n_agents" and r["n_agents"] == 6][0]
    print("\n" + "-" * 96)
    print(
        f"HEADLINE  N=6, sigma=1, 3 channels:  irreducible {ref['irreducible']:.1%} / "
        f"own {ref['own_free']:.1%} / PEER {ref['coordination_gap']:.1%}"
    )
    print(
        f"          POWER at sigma=1 measured 13.6 / 76.9 / 9.5.  The peer share is"
        f" {ref['coordination_gap'] / 0.095:.1f}x larger here, and the reason is"
    )
    print(
        "          structural, not tuned: contention on a shared channel is caused"
        "\n          by other people's traffic, whereas grid loading is dominated by"
        "\n          the agent's own injection."
    )
    print(
        f"\nOperator health: spread(std/mean)={ref['W_spread']:.2f} "
        f"(POWER 1.35), asymmetry={ref['W_asym']:.2f}, "
        f"std(Phi)/mean(Phi)={ref['phi_cv']:.3f} (POWER 0.28, floor 0.05)"
    )

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in rows for k in r})
        # Never append across schema changes: two runs with different column
        # counts in one file misalign every field in the second segment.
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
