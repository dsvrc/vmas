#!/usr/bin/env python
#  PHASE 0 -- calibration on the surrogate.  torch only, no simulator.
#
#      python pact2/calibrate.py
#      python pact2/calibrate.py --csv runs/slc_calibration.csv
#
#  Settles the questions that are about the ENVIRONMENT, before any GPU time:
#
#    * the T4 inverted-U in ``max_trust`` -- evidence, not tuning
#    * the ``mu`` tracking floor
#    * ``delta`` vs ``level`` mode, and whether the driver feedforward earns its
#      place at a one-step-stale sensor
#    * the A.6 contrast: with the loop cut, T4 is predicted NOT to apply
#
#  It cannot settle the learning questions and does not try to.  The servo it
#  drives is a *strong* stand-in for a blind policy -- high-gain velocity
#  feedback is the controller class most able to reject an actuator
#  disturbance -- so every number here is optimistic for blind, i.e.
#  conservative for the design.

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

from pact2._bootstrap import load_cores, surrogate_rollout

slc_core, pact_core = load_cores()


def _params(**kw):
    d = dict(severity=1.0, n_chan=3)
    d.update(kw)
    return slc_core.SlcParams(**d)


def arm(
    label: str,
    slc_kw: Optional[Dict] = None,
    pact_kw: Optional[Dict] = None,
    n_agents: int = 6,
    steps: int = 1500,
    batch: int = 8,
    seed: int = 0,
    warm: int = 500,
) -> Dict:
    """One rollout.  ``pact_kw=None`` is the blind arm."""
    p = _params(**(slc_kw or {}))
    comp = None
    if pact_kw is not None:
        op = slc_core.build_operator(p, n_agents, torch.device("cpu"))
        comp = pact_core.PactCompensator(
            pact_core.PactParams(**pact_kw), p, op, batch, torch.device("cpu")
        )
    r = surrogate_rollout(
        slc_core, p, n_agents, steps=steps, batch=batch, seed=seed, compensator=comp
    )
    sl = slice(warm, None)  # a FIXED window: whole-run means have survivorship bias
    fin = torch.isfinite(r["u_hat"][sl])
    return dict(
        arm=label,
        # Task completion, not speed: speed is exactly what raises Phi, so
        # scoring on it would reward the compensator for congesting the medium.
        reached=float(r["reached"][sl].mean()),
        track_err=float(r["goal_dist"][sl].mean()),
        u_mean=float(r["u"][sl].mean()),
        u_err=float((r["u_hat"][sl] - r["u"][sl]).abs()[fin].mean()),
        speed=float(r["speed"][sl].mean()),
        harm=float(r["c"][sl].mean()),
        delta_abs=float(r["delta"][sl].mean()),
        delta_nonzero=float((r["delta"][sl] > 0).to(torch.float32).mean()),
        applied_trust=float(r["trust"][sl].mean()),
        peer_abs=float(r["peer_abs"][sl].mean()),
        ff_abs=float(r["ff_abs"][sl].mean()),
        clip_frac=float(r["clip"][sl].mean()),
        sat_frac=float(r["sat"][sl].mean()),
    )


BASE = dict(ready_updates=150, warmup_updates=50)


def table(title: str, note: str, rows: List[Dict], key: str, by: str = "reached") -> None:
    print(f"\n{title}")
    print(note)
    print(
        f"\n  {key:>12s} {'reached':>8s} {'trk_err':>8s} {'u_err':>9s} {'trust':>7s} "
        f"{'delta':>7s} {'peer':>8s} {'ff':>8s} {'clip':>6s} {'sat':>6s} {'harm':>6s}"
    )
    best = max(rows, key=lambda r: r["reached"]) if by == "reached" else min(
        rows, key=lambda r: r["u_err"]
    )
    for r in rows:
        mark = "  <- best" if r is best else ""
        print(
            f"  {r['arm']:>12s} {r['reached']:8.4f} {r['track_err']:8.4f} "
            f"{r['u_err']:9.5f} "
            f"{r['applied_trust']:7.3f} {r['delta_abs']:7.4f} {r['peer_abs']:8.5f} "
            f"{r['ff_abs']:8.5f} {r['clip_frac']:6.2f} {r['sat_frac']:6.2f} "
            f"{r['harm']:6.1%}{mark}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=str, default=None)
    a = ap.parse_args()
    kw = dict(n_agents=a.agents, steps=a.steps, batch=a.batch, seed=a.seed)
    rows: List[Dict] = []

    print("=" * 100)
    print("PHASE 0 -- calibration on the surrogate   (torch only, no simulator)")
    print("=" * 100)

    # ---- 0. provisioning --------------------------------------------------
    prov = []
    for un in (0.20, 0.30, 0.40, 0.50, 0.60):
        stock_u = arm(
            "stock", slc_kw=dict(u_nominal=un, harm_at_nominal=un,
                                 harm_enabled=False), **kw
        )
        blind_u = arm("blind", slc_kw=dict(u_nominal=un, harm_at_nominal=un), **kw)
        pact_u = arm(
            f"{un:g}",
            slc_kw=dict(u_nominal=un, harm_at_nominal=un),
            pact_kw=dict(mode="delta", max_trust=0.3, **BASE),
            **kw,
        )
        pact_u["g3_depth"] = 1.0 - blind_u["reached"] / max(stock_u["reached"], 1e-9)
        prov.append(pact_u)
    rows += [dict(sweep="u_nominal", **r) for r in prov]
    print(
        "\n0. PROVISIONING.  The one number that trades G3 depth against whether\n"
        "   the channel inverse fits inside the action set.  harm_gain is pinned\n"
        "   to 1.0 by the anchor (delivered loop gain = channel idle fraction),\n"
        "   so u_nominal is the ONLY knob and it is chosen by measurement here.\n"
        "   Want: G3 depth large AND sat_frac low.  A correction that lives on\n"
        "   the action rail is a constant bias, not a compensation."
    )
    print(
        f"\n  {'u_nominal':>12s} {'G3 depth':>9s} {'reached':>8s} {'u_err':>9s} "
        f"{'delta':>7s} {'clip':>6s} {'sat':>6s} {'harm':>6s}"
    )
    for r in prov:
        print(
            f"  {r['arm']:>12s} {r['g3_depth']:9.1%} {r['reached']:8.4f} "
            f"{r['u_err']:9.5f} {r['delta_abs']:7.4f} {r['clip_frac']:6.2f} "
            f"{r['sat_frac']:6.2f} {r['harm']:6.1%}"
        )

    # ---- 1. the ladder ----------------------------------------------------
    ladder = [
        arm("stock", slc_kw=dict(harm_enabled=False), **kw),
        arm("blind", **kw),
        arm("ff", pact_kw=dict(mode="ff", **BASE), **kw),
        arm("pact", pact_kw=dict(mode="delta", max_trust=0.3, **BASE), **kw),
        arm("peer-only", pact_kw=dict(mode="delta", max_trust=0.3, ff_gain=0.0,
                                      own_gain=0.0, **BASE), **kw),
        arm("level-mode", pact_kw=dict(mode="level", max_trust=0.3, **BASE), **kw),
    ]
    rows += [dict(sweep="ladder", **r) for r in ladder]
    table(
        "1. THE LADDER.",
        "   'stock' is slc_harm_enabled=false: stock VMAS byte for byte, the\n"
        "   absolute reference B0.  'blind' is the same task under contention.\n"
        "   'ff' is the INFORMATION-MATCHED BASELINE -- it has the declared\n"
        "   operator, the driver model and its own stale sensor, everything PACT\n"
        "   has except peers' exertion.  The gap that matters is ff -> pact.",
        ladder,
        "arm",
    )

    # ---- 2. the T4 inverted-U --------------------------------------------
    trust_rows = []
    for t in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0):
        r = arm(f"{t:g}", pact_kw=dict(mode="delta", max_trust=t, **BASE), **kw)
        trust_rows.append(r)
    rows += [dict(sweep="max_trust", **r) for r in trust_rows]
    table(
        "2. THE T4 GAIN CAP.",
        "   Phi reads the EXECUTED motion, so compensating feeds the medium it\n"
        "   compensates against: a commons.  T4 predicts an INTERIOR optimum --\n"
        "   not zero, not one.  This sweep is EVIDENCE, not tuning: calibrate on\n"
        "   one seed and validate on held-out seeds, and report the whole curve.\n"
        "   Scored on u_err, the MECHANISM metric: the surrogate's fixed servo is\n"
        "   too insensitive for the task metric to resolve the difference, and a\n"
        "   flat task column here is a limit of the surrogate, not evidence of\n"
        "   no effect.",
        trust_rows,
        "max_trust",
        by="u_err",
    )

    # ---- 3. the A.6 contrast ---------------------------------------------
    noloop_rows = []
    for t in (0.0, 0.3, 0.8, 1.5, 2.0):
        r = arm(
            f"{t:g}",
            slc_kw=dict(phi_reads_executed=False),
            pact_kw=dict(mode="delta", max_trust=t, **BASE),
            **kw,
        )
        noloop_rows.append(r)
    rows += [dict(sweep="max_trust_noloop", **r) for r in noloop_rows]
    table(
        "3. THE CONTRAST (A.6): the same sweep with the loop CUT.",
        "   slc_phi_reads_executed=false makes Phi read the policy's INTENDED\n"
        "   action, so pushing back is not pushing.  T4 is predicted NOT to apply\n"
        "   and the inverted-U should flatten.  Confirming a prediction of NO\n"
        "   EFFECT is worth more than a fifth environment where everything works.",
        noloop_rows,
        "max_trust",
    )

    # ---- 4. mu ------------------------------------------------------------
    mu_rows = []
    for mu in (0.99, 0.995, 0.999, 0.9995, 0.9999):
        r = arm(f"{mu:g}", pact_kw=dict(mode="delta", max_trust=0.3, mu=mu, **BASE), **kw)
        mu_rows.append(r)
    rows += [dict(sweep="mu", **r) for r in mu_rows]
    table(
        "4. THE FORGETTING FACTOR.",
        "   RE-MEASURE per environment; the optimum follows the drift rate.\n"
        "   Aggressive forgetting buys nothing and injects noise straight into\n"
        "   the coefficient the inverse depends on.",
        mu_rows,
        "mu",
    )

    # ---- 5. the failure-mode ablations ------------------------------------
    abl = [
        arm("pact", pact_kw=dict(mode="delta", max_trust=0.3, **BASE), **kw),
        arm("r=2", pact_kw=dict(mode="delta", max_trust=0.3, r=2, **BASE), **kw),
        arm("mean-preserve", slc_kw=dict(mean_preserve=True),
            pact_kw=dict(mode="delta", max_trust=0.3, **BASE), **kw),
        arm("N=12", n_agents=12, pact_kw=dict(mode="delta", max_trust=0.3, **BASE),
            steps=a.steps, batch=a.batch, seed=a.seed),
        arm("N=12 ff", n_agents=12, pact_kw=dict(mode="ff", **BASE),
            steps=a.steps, batch=a.batch, seed=a.seed),
    ]
    rows += [dict(sweep="ablation", **r) for r in abl]
    table(
        "5. STRUCTURAL ABLATIONS.",
        "   r=2          : after per-channel normalisation both columns collapse\n"
        "                  to a weighted mean and go near-collinear.\n"
        "   N=12 vs ff   : the coordination gap is 75% at N=12 against 64% at\n"
        "                  N=6, so the ff -> pact margin should widen.\n"
        "\n"
        "   NOT here: the trace gate and the covariance-windup collapse are\n"
        "   EXCITATION-DEATH failures.  They bite only once a policy has\n"
        "   converged, and this surrogate runs a fixed servo that never stops\n"
        "   exciting -- a fixed-controller rollout would show them as no-ops and\n"
        "   silently bless the bug.  They are reproduced with excitation forced\n"
        "   off in pact2/selfcheck.py, and must be re-checked on the real\n"
        "   training curve late in the run.",
        abl,
        "ablation",
    )

    # ---- verdict ----------------------------------------------------------
    ff = next(r for r in ladder if r["arm"] == "ff")
    pact = next(r for r in ladder if r["arm"] == "pact")
    blind = next(r for r in ladder if r["arm"] == "blind")
    stock = next(r for r in ladder if r["arm"] == "stock")
    peak = min(trust_rows, key=lambda r: r["u_err"])
    interior = peak["arm"] not in (trust_rows[0]["arm"], trust_rows[-1]["arm"])
    noloop_peak = min(noloop_rows, key=lambda r: r["u_err"])
    n12 = next(r for r in abl if r["arm"] == "N=12")
    n12ff = next(r for r in abl if r["arm"] == "N=12 ff")

    print("\n" + "-" * 100)
    print("VERDICT (surrogate; the learning questions are still open)")
    print(
        f"  G3, does the dial hurt?      blind reaches "
        f"{blind['reached'] / stock['reached'] - 1:+.1%} as often as stock VMAS "
        f"({stock['reached']:.3f} -> {blind['reached']:.3f})"
    )
    span = stock["reached"] - blind["reached"]
    print(
        f"  ff arm recovers              "
        f"{(ff['reached'] - blind['reached']) / span if span else float('nan'):.1%} "
        "of that, using LOCAL information only"
    )
    print(
        f"  pact recovers                "
        f"{(pact['reached'] - blind['reached']) / span if span else float('nan'):.1%}"
    )
    print(
        f"  peer term cuts the residual loading error a further "
        f"{1 - pact['u_err'] / max(ff['u_err'], 1e-12):.1%}"
    )
    print(
        f"  correction split             base+ff+own (local) vs peer "
        f"(coordination) = {pact['ff_abs']:.5f} / {pact['peer_abs']:.5f} "
        "in loading units"
    )
    print(
        f"  margin widens with N         ff->pact u_err margin "
        f"{1 - pact['u_err'] / ff['u_err']:.1%} at N=6, "
        f"{1 - n12['u_err'] / n12ff['u_err']:.1%} at N=12 -- the direction the "
        "ceiling decomposition predicts"
    )
    print(
        f"  max_trust optimum at {peak['arm']}       "
        + (
            "<- INTERIOR, in u_err"
            if interior
            else "<- at the sweep edge; widen the sweep"
        )
    )
    print(
        f"  loop cut (A.6 contrast)      u_err optimum also at "
        f"{noloop_peak['arm']}"
    )
    print(
        "\n  HONEST LIMITS OF THIS TABLE:\n"
        "   * The task column is FLAT between ff and pact.  The surrogate runs a\n"
        "     fixed high-gain servo, so a better loading estimate changes the\n"
        "     command it issues only marginally.  The mechanism is measurable\n"
        "     here (u_err); whether it moves RETURN is a learning question and\n"
        "     needs the real environment.  Do not quote a task-metric win from\n"
        "     this script.\n"
        "   * The max_trust optimum above is an ESTIMATION optimum, not the T4\n"
        "     return inverted-U.  It is a defensible initialisation for the\n"
        "     Phase-1 sweep, which still has to be run on the real env and\n"
        "     validated on held-out seeds.\n"
        "   * The A.6 contrast is not clean on the surrogate: the two Phi\n"
        "     definitions differ by a small level shift as well as by the loop.\n"
        "     Read it on the real env, where the policy adapts to each."
    )

    if args_csv := a.csv:
        path = Path(args_csv)
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
