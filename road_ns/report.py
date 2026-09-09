#!/usr/bin/env python
#  I.6 -- what every severity run must report, plus the I.5 ceiling.
#
#      python road_ns/report.py
#      python road_ns/report.py --csv runs/road_ns_ceiling.csv
#
#  NS-4.1: commit this BEFORE any method code runs. Retuning after seeing a
#  method underperform plants the problem; the committed output is what makes
#  the decomposition a prediction rather than an explanation.

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_ns.ceiling import decompose, fleet_scan  # noqa: E402
from road_ns.dial import DialParams, dial_g, driver_A, sensitivity  # noqa: E402
from road_ns.structure import load_structure  # noqa: E402


def dial_table(struct, sigma: float) -> dict:
    p = DialParams(severity=sigma)
    s = sensitivity(struct, p)
    steps = torch.arange(0, p.period)
    a = driver_A(steps, p)
    g = dial_g(a, s, p)  # (P, A)
    dry = int((a == 0.0).sum())
    return {
        "sigma": sigma,
        "capacity_removed": float(1.0 - g.mean()),
        "peak_to_trough": float(g.max() / g.min()),
        "placebo_days": dry,
        "cycle": p.period,
        "g_min": float(g.min()),
        "g_max": float(g.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--severities", type=float, nargs="+", default=[0.5, 1.0, 3.0])
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    struct = load_structure()
    routes = struct.routes()
    W = struct.coupling(
        torch.randint(0, len(routes), (args.agents,),
                      generator=torch.Generator().manual_seed(0)).tolist(),
        routes,
    )

    print("=" * 88)
    print("road_ns -- I.6 mandatory report   (offline: no policy, no training, no simulator)")
    print("=" * 88)
    print(struct.banner())
    print(
        f"routes: {len(routes)}   operator: spread={struct.spread(W):.3f} "
        f"asym={struct.asymmetry(W):.3f} zero-diagonal="
        f"{bool(torch.all(W.diagonal() == 0))}"
    )

    print("\n-- the dial ------------------------------------------------------")
    print(
        f"  {'sigma':>6s} {'cap removed':>12s} {'peak/trough':>12s} "
        f"{'placebo':>9s} {'g range':>18s}"
    )
    rows = []
    for s in args.severities:
        d = dial_table(struct, s)
        rows.append(dict(kind="dial", **d))
        tag = "  <- beyond-physical" if s > 1.0 else ("  <- HCM anchor" if s == 1.0 else "")
        print(
            f"  {d['sigma']:>6g} {d['capacity_removed']:11.2%} "
            f"{d['peak_to_trough']:11.3f}x {d['placebo_days']:>4d}/{d['cycle']:<4d} "
            f"[{d['g_min']:.3f}, {d['g_max']:.3f}]{tag}"
        )

    print("\n-- the ceiling decomposition (I.5) -------------------------------")
    print(
        f"  {'fleet':>10s} {'irred':>7s} {'own':>7s} {'PEER':>7s} "
        f"{'dec.ceil':>8s} {'u_mean':>7s} {'g':>6s}"
    )
    for s in args.severities:
        p = DialParams(severity=s)
        for d in fleet_scan(struct, routes, p, n_total=args.agents):
            rows.append(
                dict(
                    kind="ceiling",
                    sigma=s,
                    n_controllable=d.n_controllable,
                    n_background=d.n_background,
                    irreducible=d.irreducible,
                    own=d.own,
                    peer=d.peer,
                    decentralized_ceiling=d.decentralized_ceiling,
                    u_mean=d.u_mean,
                    u_p95=d.u_p95,
                    g_mean=d.g_mean,
                )
            )
            print(f"  s={s:<4g} {d.row()}")

    ref = next(
        r
        for r in rows
        if r["kind"] == "ceiling"
        and r["sigma"] == 1.0
        and r["n_controllable"] == max(1, int(0.4 * args.agents))
    )
    print("\n" + "-" * 88)
    print(
        f"HEADLINE  sigma=1, {ref['n_controllable']}/{args.agents} controllable:  "
        f"irreducible {ref['irreducible']:.1%} / own {ref['own']:.1%} / "
        f"PEER {ref['peer']:.1%}"
    )
    print(
        "          URB reference at the same 40% controllable share: 42.6 / 16.5 / 40.8"
    )
    print(
        f"          loading u_mean={ref['u_mean']:.3f} -- I.6 requires this: a medium "
        "far below its\n          limit cannot express a capacity loss however large "
        "sigma grows."
    )

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in rows for k in r})
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
