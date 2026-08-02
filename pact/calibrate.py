#  Phase 0 -- calibrate the non-stationarity to the policy's *real* operating scale.
#
#  Guide pitfall #4: sizing SEVERITY against an assumed action magnitude is how
#  a first calibration ends up doing nothing.  Rather than guess, this script
#  mirrors VMAS navigation's dynamics exactly (double integrator, per-step drag,
#  two substeps, dt = 0.1, unit mass, action box [-1, 1]^2) and drives it with a
#  *saturating velocity servo* -- a deliberately strong stand-in for a trained
#  blind policy, since a high-gain velocity servo is the controller class most
#  able to reject a rotational actuator disturbance by feedback alone.  Results
#  here are therefore optimistic for blind, i.e. conservative for the design.
#
#  It answers four questions before any GPU time is spent:
#    1. What operating scale (|Phi|, |x2|, |theta|) does the NS actually reach?
#       -> sets `ns_gain`.
#    2. At what severity does blind fall below 30% of B0 while scripted
#       compensation holds >= 90%?  -> sets `ns_severity`.
#    3. Can a blind team escape simply by *exerting less*?  That escape needs no
#       knowledge of the driver, so if it works the environment fails guide
#       condition 7 regardless of how the disturbance is shaped.  -> sets the
#       task horizon.
#    4. How wide is the gain-tolerance band around beta = c?  For a transform
#       channel the binding constraint is the conditioning of the inverse, not
#       the actuator, so this -- not saturation -- is what defines sigma*.
#
#  Run:  python pact/calibrate.py
#
#  This is a *surrogate*, not the environment.  Its job is to put the constants
#  in the right decade; the real numbers come from Phase 1
#  (`pact/phase1_certify.py`) run against a trained policy.

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch


def _load_pcw_core():
    """Load the arithmetic core by path.

    Deliberately bypasses ``import benchmarl`` so this script runs on a plain
    torch install with no torchrl / vmas, which is the whole point of keeping
    the core dependency-free.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarl"
        / "environments"
        / "vmas_ns"
        / "pcw_core.py"
    )
    spec = importlib.util.spec_from_file_location("pcw_core", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules, so register first
    sys.modules["pcw_core"] = module
    spec.loader.exec_module(module)
    return module


_core = _load_pcw_core()
angular_impulse = _core.angular_impulse
channel_inverse = _core.channel_inverse
leak_step = _core.leak_step
peer_mean = _core.peer_mean
rotate = _core.rotate
wrap_angle = _core.wrap_angle

# ---- VMAS navigation constants (see vmas/simulator/core.py) -------------------
DT = 0.1
SUBSTEPS = 2
SUB_DT = DT / SUBSTEPS
DRAG = 0.25
MASS = 1.0
U_RANGE = 1.0
WORLD_SPAWN = 1.0
EPISODE_LEN = 100


def servo(
    pos: torch.Tensor,
    vel: torch.Tensor,
    goal: torch.Tensor,
    speed_scale: float = 1.0,
) -> torch.Tensor:
    """Saturating velocity servo: a strong stand-in for a trained blind policy.

    ``speed_scale`` throttles the reference speed, which is how the
    "exert less to escape" strategy is tested.
    """
    to_goal = goal - pos
    dist = to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v_max = 0.4 * speed_scale  # 0.4 = terminal speed at full thrust here
    speed = torch.minimum(4.0 * dist, torch.full_like(dist, v_max))
    desired_vel = to_goal / dist * speed
    return (10.0 * (desired_vel - vel)).clamp(-U_RANGE, U_RANGE)


def integrate(pos, vel, force):
    """One VMAS world step: drag applied once, then `SUBSTEPS` integrations."""
    vel = vel * (1.0 - DRAG)
    for _ in range(SUBSTEPS):
        vel = vel + (force / MASS) * SUB_DT
        pos = pos + vel * SUB_DT
    return pos, vel


def rollout(
    *,
    n_agents: int,
    batch: int,
    severity: float,
    gain: float,
    rho: float,
    beta_scale=None,
    beta_abs=None,
    seed: int,
    driver: float = 1.0,
    speed_scale: float = 1.0,
    episode_len: int = EPISODE_LEN,
):
    """One frozen-driver episode batch.

    Args:
        beta_scale: compensation gain relative to the truth, ``beta = k * c``,
            so ``1.0`` is the perfectly-informed scripted controller (the O1
            ceiling) and ``1.1`` is a 10% gain error.  ``None`` with no
            ``beta_abs`` means blind.
        beta_abs: an absolute gain, for testing a *phase-blind* fixed gain that
            cannot be written as a multiple of ``c`` (``c`` hits zero).
        driver: frozen driver value ``A``; Phase 1 freezes it at the peak.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, n_agents, 2)
    pos = (torch.rand(shape, generator=generator) * 2 - 1) * WORLD_SPAWN
    goal = (torch.rand(shape, generator=generator) * 2 - 1) * WORLD_SPAWN
    vel = torch.zeros(shape)
    x2 = torch.zeros(batch, n_agents)
    c = driver * severity

    d0 = (pos - goal).norm(dim=-1)
    stats = {k: [] for k in ("phi", "x2", "theta", "sat")}

    for _ in range(episode_len):
        theta = c * x2
        a = servo(pos, vel, goal, speed_scale=speed_scale)

        if beta_scale is None and beta_abs is None:
            u = a
            sat = torch.zeros(batch, n_agents)
        else:
            beta = beta_abs if beta_abs is not None else beta_scale * c
            raw = channel_inverse(a, beta * x2)
            u = raw.clamp(-U_RANGE, U_RANGE)
            sat = (raw != u).any(dim=-1).float()

        # The message is formed from the position the command was issued FROM,
        # matching the scenario, which reads agent.state.pos inside process_action.
        message = angular_impulse(pos, u)
        phi = peer_mean(message)

        delivered = rotate(u, theta)
        pos, vel = integrate(pos, vel, delivered)
        x2 = leak_step(x2, phi, rho=rho, gain=gain)

        stats["phi"].append(phi)
        stats["x2"].append(x2)
        stats["theta"].append(wrap_angle(theta).abs())
        stats["sat"].append(sat)

    dT = (pos - goal).norm(dim=-1)
    # Exactly VMAS navigation's telescoped position-shaping return.
    episode_return = float((d0 - dT).mean())
    return episode_return, {k: torch.stack(v) for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    # Defaults are the SHIPPED constants, so a bare run reproduces the tables in
    # pact/README.md §2.  See benchmarl/conf/task/vmas_ns/navigation_pcw.yaml.
    parser.add_argument("--gain", type=float, default=20.0)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument(
        "--severities", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    parser.add_argument(
        "--speed-scales", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.35, 0.25]
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[100, 70, 50, 40, 30])
    args = parser.parse_args()

    common = dict(
        n_agents=args.n_agents,
        batch=args.batch,
        gain=args.gain,
        rho=args.rho,
        seed=args.seed,
        episode_len=args.horizon,
    )

    b0, b0_stats = rollout(severity=0.0, beta_scale=None, **common)
    print(
        f"\nVMAS-navigation surrogate | N={args.n_agents} "
        f"| G={args.gain} rho={args.rho} horizon={args.horizon}"
    )
    print(f"B0 (no NS)                = {b0:+.4f}")
    print(f"  PACT must clear (>=90%) = {0.90 * b0:+.4f}")
    print(f"  blind must stay (<=30%) = {0.30 * b0:+.4f}")
    print(
        f"operating scale: |Phi| rms = {float(b0_stats['phi'].pow(2).mean().sqrt()):.4f}, "
        f"|x2| rms = {float(b0_stats['x2'].pow(2).mean().sqrt()):.4f} rad per unit c"
    )

    # ---------------------------------------------------------------- 1 + 2
    print("\n[1] severity sweep (frozen at driver peak)")
    print(f"{'sigma':>6} | blind %B0 | oracle %B0 | orac sat | |th| mean | |th| p95 | >90deg")
    print("-" * 82)
    rows = []
    for sigma in args.severities:
        blind, bs = rollout(severity=sigma, beta_scale=None, **common)
        orac, os_ = rollout(severity=sigma, beta_scale=1.0, **common)
        rows.append((sigma, 100 * blind / b0, 100 * orac / b0))
        print(
            f"{sigma:6.2f} | {100 * blind / b0:9.1f} | {100 * orac / b0:10.1f} | "
            f"{100 * float(os_['sat'].mean()):7.1f}% | "
            f"{float(bs['theta'].mean()):9.3f} | {float(bs['theta'].quantile(0.95)):8.3f} | "
            f"{100 * float((bs['theta'] > math.pi / 2).float().mean()):5.1f}%"
        )
    usable = [s for s, b, o in rows if b <= 30.0 and o >= 90.0]
    print(f"  -> blind<=30% and oracle>=90% at sigma in {usable or 'NONE'}")

    # ------------------------------------------------------------------- 3
    sigma_t = args.severities[-1]
    print(
        f"\n[3] can a blind team escape by exerting less?  (sigma={sigma_t}, "
        "best blind over speed scales; each cell is % of that horizon's own B0)"
    )
    header = "  ".join(f"{s:>6.2f}" for s in args.speed_scales)
    print(f"{'horizon':>8} | {'B0':>7} | {header} |   best")
    print("-" * (24 + 8 * len(args.speed_scales) + 9))
    for horizon in args.horizons:
        kw = dict(common)
        kw["episode_len"] = horizon
        b0_h, _ = rollout(severity=0.0, beta_scale=None, **kw)
        cells, best = [], -1e9
        for scale in args.speed_scales:
            ret, _ = rollout(
                severity=sigma_t, beta_scale=None, speed_scale=scale, **kw
            )
            pct = 100 * ret / b0_h
            cells.append(pct)
            best = max(best, pct)
        flag = "  <- ESCAPES" if best > 30.0 else "  ok"
        print(
            f"{horizon:8d} | {b0_h:+7.4f} | "
            + "  ".join(f"{c:6.1f}" for c in cells)
            + f" | {best:6.1f}{flag}"
        )
    print(
        "  A horizon whose row 'ESCAPES' is unusable: the team recovers by\n"
        "  moderating exertion, which needs no knowledge of the driver."
    )

    # ------------------------------------------------------------------- 4
    print(
        f"\n[4] gain-tolerance band around beta = c  (the binding constraint for a\n"
        f"    norm-preserving transform channel; % of B0 at beta = k*c)"
    )
    ks = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]
    print(f"{'sigma':>6} | " + "  ".join(f"{k:>5.2f}" for k in ks))
    print("-" * (9 + 7 * len(ks)))
    for sigma in args.severities:
        if sigma == 0.0:
            continue
        cells = []
        for k in ks:
            ret, _ = rollout(severity=sigma, beta_scale=k, **common)
            cells.append(100 * ret / b0)
        print(f"{sigma:6.2f} | " + "  ".join(f"{c:5.1f}" for c in cells))
    print("  sigma* = the largest sigma whose band still clears 90% under a")
    print("  realistic gain error (the +-10% columns).")

    # ------------------------------------------------------------------- 4b
    print(
        f"\n[4b] why beta must TRACK the phase (sigma={sigma_t}): a gain that is\n"
        "     right at the peak injects a disturbance of its own in the trough,\n"
        "     where c=0 but x2 does not vanish.  % of B0:"
    )
    print(f"{'driver A':>9} | {'c=A*sig':>8} | {'blind':>7} | {'fixed b=sig':>12} | {'b=c':>7}")
    print("-" * 56)
    for driver in (0.0, 0.25, 0.5, 0.75, 1.0):
        kw = dict(common)
        blind, _ = rollout(severity=sigma_t, driver=driver, **kw)
        # A phase-blind policy can only pick one gain.  Picking the peak-optimal
        # one means carrying beta = sigma into every phase, trough included.
        fixed, _ = rollout(
            severity=sigma_t, beta_abs=sigma_t, driver=driver, **kw
        )
        perfect, _ = rollout(
            severity=sigma_t, beta_scale=1.0, driver=driver, **kw
        )
        print(
            f"{driver:9.2f} | {driver * sigma_t:8.3f} | {100 * blind / b0:6.1f}% | "
            f"{100 * fixed / b0:11.1f}% | {100 * perfect / b0:6.1f}%"
        )
    print(
        "     A fixed gain below blind in the trough is the negative result the\n"
        "     PACT ladder predicts: it is what forces recurrence / a CTDE critic."
    )

    # -------------------------------------------------- irreducibility check
    print(f"\n[5] N-ablation at sigma={sigma_t} (irreducibility certificate)")
    for n in (1, 2, 3, 4):
        kw = dict(common)
        kw["n_agents"] = n
        ret, st = rollout(severity=sigma_t, beta_scale=None, **kw)
        ref, _ = rollout(severity=0.0, beta_scale=None, **kw)
        note = "   <- must be EXACTLY 0.0" if n == 1 else ""
        print(
            f"  N={n}: NS return={ret:+.4f}  no-NS={ref:+.4f}  "
            f"|theta| max={float(st['theta'].max()):.6f}{note}"
        )


if __name__ == "__main__":
    main()
