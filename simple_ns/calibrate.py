#!/usr/bin/env python
#  Pick the operating point BEFORE spending training compute.
#
#      python simple_ns/calibrate.py
#      python simple_ns/calibrate.py --hosts balance transport --envs 64
#
#  ANT_complete_story.md §2.2 asks the question this answers:
#
#      "At severity sigma, could *anything* recover?  Not 'can our algorithm
#       learn it' -- can any controller, even one handed the answer for free,
#       get back to normal?  If the answer is no, a failed training run tells
#       you nothing.  You would be measuring the environment, not the method."
#
#  So three controllers are run down the same severity ladder, on the host's OWN
#  shipped heuristic policy rather than on a random action stream:
#
#      blind      the heuristic, disturbed, no compensation        -> how far it falls
#      oracle     the heuristic, handed the TRUE disturbance       -> the ceiling
#      pact       the heuristic, with PACT estimating online       -> what is earned
#
#  A random action stream cannot answer this and will mislead: measured on
#  ``sampling``, a random policy's return RISES with severity, because being
#  shoved around spreads the agents out and accidentally improves coverage.
#  Whatever severity you pick must be picked against a controller that was
#  actually trying.
#
#  What to read off it:
#    * blind % of B0 -- how much headroom there is to recover.  If this is 98%
#      there is nothing to show, whatever the method does.
#    * oracle % of B0 -- the ceiling.  If the oracle cannot recover, sigma is
#      past sigma* and the row is about the environment, not the method.
#    * pact % of B0 -- what an online estimator actually earns of that ceiling.
#
#  Needs vmas.  Does NOT need torchrl or benchmarl.

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simple_ns.hosts import make_scenario  # noqa: E402

#: hosts that ship a heuristic, and a sensible fleet size for each
DEFAULT_HOSTS = {"balance": 4, "transport": 4, "navigation": 4}

NS = dict(
    ns_period=100,
    ns_wet_fraction=0.5,
    ns_loss_at_sigma1=0.14,
    ns_rho=0.9,
    ns_n_types=3,
    ns_recv_spread=0.6,
    ns_send_spread=0.8,
    ns_kernel_lambda=0.35,
    ns_y_clip=10.0,
    ns_observe_residual=True,
)
PACT = dict(pact_trust=0.9, pact_mu=0.999, pact_p0=10.0, pact_warmup=30)


def heuristic_for(host: str):
    mod = __import__(f"vmas.scenarios.{host}", fromlist=["HeuristicPolicy"])
    return mod.HeuristicPolicy(continuous_action=True)


def build(host, n, sigma, envs, seed, pact, direct=False, channels="full",
          oracle=False, **over):
    from vmas import make_env

    sc = make_scenario(pact=pact, host=host)
    kw = dict(
        num_envs=envs,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        max_steps=10**9,
        clamp_actions=True,
        n_agents=n,
        ns_severity=sigma,
        ns_direct=direct,
        pact_enabled=pact,
        pact_channels=channels,
        pact_oracle=oracle,
        **NS,
        **PACT,
    )
    kw.update(over)
    env = make_env(scenario=sc, **kw)
    env.reset()
    return sc, env


def roll(sc, env, policy, steps: int) -> Dict[str, float]:
    """One rollout under the host's own heuristic policy.

    The free-answer controller is an ARM (``pact_oracle=True``), not something
    applied out here.  From outside the environment the best available answer is
    one step stale, and that understates the ceiling badly enough that PACT
    appeared to BEAT it -- measured 78.9% of B0 against the oracle's 65.0% at
    sigma = 2, which is not a possible result.
    """
    N = len(sc.world.agents)
    obs = [sc.observation(a) for a in sc.world.agents]
    total = torch.zeros(N, sc.world.batch_dim)
    loads, errs, clips = [], [], []
    for _ in range(steps):
        acts = [
            policy.compute_action(obs[i], u_range=a.u_range)
            for i, a in enumerate(sc.world.agents)
        ]
        obs, rew, _, _ = env.step(acts)
        total += torch.stack(rew)
        loads.append(sc._load.abs().mean())
        clips.append(sc._clipped.mean())
        if hasattr(sc, "_pred"):
            errs.append((sc._load - sc._pred).abs().mean())
    return dict(
        ret=float(total.mean()),
        load=float(torch.stack(loads).mean()),
        clip=float(torch.stack(clips).mean()),
        err=float(torch.stack(errs).mean()) if errs else float("nan"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hosts", nargs="+", default=sorted(DEFAULT_HOSTS))
    ap.add_argument("--severities", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []
    print("=" * 96)
    print("simple_ns -- operating point, on each host's OWN heuristic policy")
    print(f"envs={args.envs} steps={args.steps} seeds={args.seeds}")
    print("=" * 96)

    for host in args.hosts:
        n = DEFAULT_HOSTS.get(host, 4)
        pol = heuristic_for(host)
        print(f"\n-- {host} (N={n}) " + "-" * (74 - len(host)))
        print(f"  {'sigma':>6} {'blind':>10} {'%B0':>7} {'oracle':>10} {'%B0':>7} "
              f"{'pact':>10} {'%B0':>7} {'|load|':>8} {'err':>7} {'clip':>6}")
        base = None
        for sg in args.severities:
            acc = {k: [] for k in ("blind", "oracle", "pact", "load", "err", "clip")}
            for sd in args.seeds:
                sc, env = build(host, n, sg, args.envs, sd, pact=False)
                b = roll(sc, env, pol, args.steps)
                sc, env = build(host, n, sg, args.envs, sd, pact=True, oracle=True)
                o = roll(sc, env, pol, args.steps)
                sc, env = build(host, n, sg, args.envs, sd, pact=True)
                q = roll(sc, env, pol, args.steps)
                acc["blind"].append(b["ret"]); acc["oracle"].append(o["ret"])
                acc["pact"].append(q["ret"]); acc["load"].append(b["load"])
                acc["err"].append(q["err"]); acc["clip"].append(q["clip"])
            m = {k: sum(v) / len(v) for k, v in acc.items()}
            if base is None:
                base = m["blind"]
            pc = lambda v: 100.0 * v / base if abs(base) > 1e-12 else float("nan")  # noqa: E731
            rows.append(dict(host=host, n_agents=n, sigma=sg, b0=base, **m))
            print(f"  {sg:>6g} {m['blind']:>10.2f} {pc(m['blind']):>6.1f}% "
                  f"{m['oracle']:>10.2f} {pc(m['oracle']):>6.1f}% "
                  f"{m['pact']:>10.2f} {pc(m['pact']):>6.1f}% "
                  f"{m['load']:>8.4f} {m['err']:>7.4f} {m['clip']:>5.1%}", flush=True)

    print("\n" + "=" * 96)
    print("Read it like this:")
    print("  blind %B0 near 100  -> nothing to recover at this sigma; raise it.")
    print("  oracle %B0 far below 100 -> past sigma*; the ROW is about the")
    print("     environment, not the method, and must be labelled that way.")
    print("  pact between the two -> what an online estimator earned of the")
    print("     ceiling the oracle defines.  That fraction is the result.")
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
