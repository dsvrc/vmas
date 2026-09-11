#!/usr/bin/env python
#  Build-order STEP 7 -- the maximum-excitation probe, plus the operating point.
#
#      python road_ns/probe.py                       # the default ladder
#      python road_ns/probe.py --agents 16 40 80 --severities 0 1 3
#      python road_ns/probe.py --csv runs/road_ns_probe.csv
#
#  III.3 puts this before the sweep and calls it the step people skip:
#
#      "A probe with uniformly random actions is the best case for
#       identification -- maximum excitation -- so a fit gain near zero there is
#       decisive, and it costs a fiftieth of the full run."
#
#  It answers three questions, none of which needs a learning framework:
#
#    1. Does the dial BITE?  I.6 requires the loading distribution and the
#       capacity removed to be reported before any result is quoted, and Q7 of
#       the III.1 questionnaire is "is the medium loaded enough for a capacity
#       loss to matter".  A severity that moves the domain metric by less than
#       seed noise is a dead experiment, and finding that out here costs
#       minutes.
#
#    2. Is there anything to IDENTIFY?  II.9 gate 5 (channels not inert) and
#       gate 6 (rolling fit gain above a floor).  Fit gain is scored against an
#       intercept-only null, because raw R^2 is inflated by the per-agent
#       intercept memorising each agent's typical residual (II.10).
#
#    3. What is ALPHA here?  III.1's stated one-line procedure -- mean relative
#       excess divided by mean loading.  The shipped 2.28 is URB's figure and a
#       declared PLACEHOLDER; see the note under `--calibrate` for why this
#       instance cannot honestly calibrate it from a scripted driver.
#
#  Needs vmas (it drives the real scenario) but NOT torchrl or benchmarl.

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_ns.scenario import make_scenario  # noqa: E402


# ---------------------------------------------------------------------------
#  drivers
# ---------------------------------------------------------------------------


def pure_pursuit(sc, target: float = 0.9, look: int = 14, gain: float = 2.0):
    """A competent driver: aim at a point ahead on your own centre line.

    Not a policy and not a baseline -- a way to read the medium under behaviour
    that at least follows the road, so the domain metric (distance covered)
    means something.  A uniformly random driver spends most of its time off the
    road, where the severity has nothing to act on.
    """
    pos = sc._positions()
    idx = sc._route_of * sc.K + (sc._prog + look) % sc.K
    tgt = sc._point_flat[idx] - pos
    rot = torch.stack([a.state.rot[:, 0] for a in sc.world.agents], dim=1)
    c, s = rot.cos(), rot.sin()
    x = c * tgt[..., 0] + s * tgt[..., 1]
    y = -s * tgt[..., 0] + c * tgt[..., 1]
    steer = (torch.atan2(y, x) * gain).clamp(-sc.max_steering, sc.max_steering)
    v = torch.full_like(steer, target)
    return [torch.stack([v[:, i], steer[:, i]], -1) for i in range(len(sc.world.agents))]


def uniform_random(sc):
    """III.3 step 7: MAXIMUM excitation.  The best case for identification."""
    B, N = sc.world.batch_dim, len(sc.world.agents)
    v = torch.rand(B, N, device=sc._device) * sc.max_speed
    st = (torch.rand(B, N, device=sc._device) * 2 - 1) * sc.max_steering
    return [torch.stack([v[:, i], st[:, i]], -1) for i in range(N)]


# ---------------------------------------------------------------------------
#  the run
# ---------------------------------------------------------------------------


def build(n_agents: int, sigma: float, envs: int, seed: int, pact: bool = False, **over):
    from vmas import make_env

    sc = make_scenario(pact=pact, host="lanelet_flow")
    kw = dict(
        num_envs=envs,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        max_steps=10**9,  # the probe controls its own horizon
        clamp_actions=True,
        n_agents=n_agents,
        n_nearing_agents_observed=4,
        ns_severity=sigma,
        ns_period=100,
        ns_wet_fraction=0.5,
        ns_alpha=2.28,
        ns_mean_preserve=False,
        ns_observe_loading=True,
        ns_route_set="loops",
        ns_exclude_self=True,
        pact_enabled=pact,
        pact_trust=0.9,
        pact_kappa=1.0,
        pact_mu=0.999,
        pact_p0=10.0,
        pact_y_clip=10.0,
        pact_warmup=50,
        pact_shift_mode="centred",
        pact_shift_clip=0.5,
    )
    kw.update(over)
    env = make_env(scenario=sc, **kw)
    env.reset()
    return sc, env


def roll(sc, env, steps: int, driver, collect_psi: bool = False) -> Dict[str, object]:
    """One rollout.  Returns the medium's read-out plus, optionally, the rows
    the estimator would have been fed."""
    N = len(sc.world.agents)
    B = sc.world.batch_dim
    dist = torch.zeros(B, N)
    harm, u, speed = [], [], []
    psi_rows: List[torch.Tensor] = []
    y_rows: List[torch.Tensor] = []

    for _ in range(steps):
        prev = sc._prog.clone()
        prev_route = sc._route_of.clone()
        env.step(driver(sc))
        K = sc.K
        step_idx = (sc._prog.long() - prev.long() + K // 2) % K - K // 2
        d = step_idx.float() * sc._spacing[sc._route_of]
        # a respawned / rerouted vehicle did not "travel" the discontinuity
        d = torch.where(prev_route == sc._route_of, d, torch.zeros_like(d))
        dist += d
        harm.append(sc._harm.mean())
        u.append(sc._u.mean())
        speed.append(
            torch.stack([a.state.vel.norm(dim=-1) for a in sc.world.agents], 1).mean()
        )
        if collect_psi and hasattr(sc, "basis"):
            psi_rows.append(sc.basis.design(sc._ns_route, sc._ref, sc._scale).reshape(-1, sc._dim))
            y_rows.append((sc._harm_prev - 1.0).reshape(-1))

    out: Dict[str, object] = dict(
        distance=float(dist.mean()),
        harm=float(torch.stack(harm).mean()),
        harm_peak=float(torch.stack(harm).max()),
        u=float(torch.stack(u).mean()),
        speed=float(torch.stack(speed).mean()),
        offroad=float(sc._offroad.float().mean()),
        collided=float(sc._collided.float().mean()),
    )
    if collect_psi and psi_rows:
        out["psi"] = torch.cat(psi_rows)
        out["y"] = torch.cat(y_rows)
    return out


def fit_gain(psi: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """II.10: score the reduction against an INTERCEPT-ONLY null.

    Raw R^2 is inflated by the intercept column memorising the typical residual,
    so the number that means "the peer channels explained something" is the
    improvement over a fit that has only that column.
    """
    live = psi.abs().sum(-1) > 0
    psi, y = psi[live], y[live]
    if psi.shape[0] < psi.shape[1] * 10:
        return dict(fit_gain=float("nan"), cond=float("nan"), n=int(psi.shape[0]))
    sol = torch.linalg.lstsq(psi, y.unsqueeze(-1)).solution
    sse_full = float((y - (psi @ sol).squeeze(-1)).pow(2).sum())
    null = psi[:, :1]
    sol0 = torch.linalg.lstsq(null, y.unsqueeze(-1)).solution
    sse_null = float((y - (null @ sol0).squeeze(-1)).pow(2).sum())
    gram = psi.T @ psi
    return dict(
        fit_gain=(sse_null - sse_full) / max(sse_null, 1e-30),
        cond=float(torch.linalg.cond(gram)),
        n=int(psi.shape[0]),
        x_std=float(psi[:, 1:].std()),
    )


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, nargs="+", default=[16, 40, 80])
    ap.add_argument("--severities", type=float, nargs="+", default=[0.0, 1.0, 3.0])
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="III.1's alpha procedure: mean(realized/nominal - 1) / mean(u). "
        "Honest only under a driver that actually slows for traffic -- see the "
        "note this prints.",
    )
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []
    width = 92
    print("=" * width)
    print("road_ns -- build-order step 7: maximum-excitation probe and operating point")
    print(f"host lanelet_flow   envs={args.envs}   steps={args.steps}   seed={args.seed}")
    print("=" * width)

    # -- 1. does the dial bite? --------------------------------------------
    print("\n-- severity ladder, pure-pursuit driver (Q7 / I.6) ----------------")
    print(
        f"  {'N':>4} {'sigma':>6} {'mean harm':>10} {'peak harm':>10} {'mean u':>8} "
        f"{'distance':>10} {'% of B0':>8}"
    )
    for n in args.agents:
        base = None
        for sg in args.severities:
            sc, env = build(n, sg, args.envs, args.seed)
            r = roll(sc, env, args.steps, pure_pursuit)
            if base is None:
                base = r["distance"]
            pct = 100.0 * r["distance"] / max(base, 1e-12)
            rows.append(dict(kind="ladder", n_agents=n, sigma=sg, pct_of_b0=pct, **r))
            tag = ""
            if sg == 1.0:
                tag = "  <- HCM anchor"
            elif sg > 1.0:
                tag = "  <- BEYOND-PHYSICAL"
            print(
                f"  {n:>4} {sg:>6g} {r['harm']:>10.4f} {r['harm_peak']:>10.4f} "
                f"{r['u']:>8.3f} {r['distance']:>9.2f}m {pct:>7.1f}%{tag}"
            )
        print()

    # -- 2. is there anything to identify? ----------------------------------
    print("-- maximum excitation: uniform random actions (III.3 step 7) ------")
    print(
        f"  {'N':>4} {'sigma':>6} {'fit_gain':>9} {'cond(psi)':>10} {'x_std':>8} "
        f"{'rows':>8}   verdict"
    )
    for n in args.agents:
        for sg in [s for s in args.severities if s > 0]:
            sc, env = build(n, sg, args.envs, args.seed, pact=True, pact_enabled=True)
            r = roll(sc, env, max(60, args.steps // 4), uniform_random, collect_psi=True)
            if "psi" not in r:
                continue
            f = fit_gain(r["psi"], r["y"])
            rows.append(dict(kind="probe", n_agents=n, sigma=sg, **f))
            if f["fit_gain"] != f["fit_gain"]:
                verdict = "too few rows"
            elif f["fit_gain"] < 0.01:
                verdict = "DEAD -- gate 6 would abort; a full run cannot help"
            elif f["cond"] > 1e4:
                verdict = "predicts, but beta is NOT decomposable (gate 7 warns)"
            else:
                verdict = "live"
            print(
                f"  {n:>4} {sg:>6g} {f['fit_gain']:>9.4f} {f['cond']:>10.1f} "
                f"{f.get('x_std', float('nan')):>8.4f} {f['n']:>8}   {verdict}"
            )

    if args.calibrate:
        print("\n-- alpha (III.1) --------------------------------------------------")
        print(
            "  NOT calibrated here, and the reason is structural rather than a\n"
            "  missing feature.  alpha relates LOADING to DELAY in the stationary\n"
            "  medium, so it must be measured at sigma=0.  This host has no\n"
            "  car-following model -- agents are collide=False, exactly as\n"
            "  road_traffic has them -- so at sigma=0 congestion delays nobody\n"
            "  except through the proximity penalty, which only a trained policy\n"
            "  responds to.  Measuring alpha from a scripted driver would return\n"
            "  ~0 and would be a measurement of the driver, not of the medium.\n"
            "  Calibrate it from the first blind sigma=0 checkpoint -- collect\n"
            "  mean(v_commanded / v_realized - 1) and mean(u), then\n"
            "  road_ns.dial.calibrate_alpha -- and re-run the ladder before\n"
            "  quoting any headline number.  2.28 is URB's figure and is carried\n"
            "  as a declared placeholder until then."
        )

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{k: v for k, v in r.items() if not torch.is_tensor(v)} for r in rows]
        fields = sorted({k for r in rows for k in r})
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
