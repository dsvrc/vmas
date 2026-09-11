#!/usr/bin/env python
#  Offline conformance.  torch only -- no vmas, no torchrl, seconds to run.
#
#      python simple_ns/conformance.py
#
#  The two decision-procedure checks come first, because they are what the
#  classification claim rests on: which cell an instance is in is settled by
#  MEASUREMENT here, not by argument in the paper.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simple_ns.coupling import Coupling  # noqa: E402
from simple_ns.driver import (  # noqa: E402
    DialParams,
    beta_star,
    class_constants,
    cycle_mean_A,
    driver_A,
)

RESULTS: List[Tuple[str, bool, str]] = []
N = 6


def check(name: str):
    def wrap(fn: Callable[[], str]):
        try:
            RESULTS.append((name, True, fn() or ""))
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        return fn

    return wrap


def fleet(n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(4, n, 2, generator=g) * 2 - 1
    u = torch.rand(4, n, 2, generator=g) * 2 - 1
    return pos, u


def load_of(c: Coupling, p: DialParams, pos, u, a_val=1.0, steps=4):
    Q = torch.zeros(pos.shape[0], c.n, c.r, 2)
    for _ in range(steps):
        Q, _, ehat, x = c.step_channels(pos, u, Q)
    beta = beta_star(torch.full((pos.shape[0],), a_val), p)
    return (beta.unsqueeze(1) * x).sum(-1)


# ===========================================================================
#  the decision procedure
# ===========================================================================


@check("test_lone_agent_feels_nothing")
def _lone():
    """The (B) vs (C) test.  A lone agent must read EXACTLY zero at any severity.

    NS_design_guide.md 2.1: "C == 0 at N=1, so the single-agent projection of the
    world is exactly stationary. What is non-stationary is not the world -- it is
    how agents matter to each other."  If a single agent suffers, the driver is
    reaching it directly and the instance is category B in disguise.
    """
    pos, u = fleet(1)
    for sigma in (0.5, 1.0, 3.0, 100.0):
        p = DialParams(severity=sigma)
        c = Coupling(1, p)
        assert float(load_of(c, p, pos, u).abs().max()) == 0.0, (
            f"a lone agent read a non-zero load at sigma={sigma}"
        )
    return "N=1 reads exactly 0.0 at sigma in {0.5, 1, 3, 100} -- category C"


@check("test_frozen_partners_still_drift")
def _frozen():
    """The (A) vs (C) test.  With teammates frozen at a fixed policy the
    disturbance must still drift, or it is a learning artefact rather than a
    property of the task."""
    p = DialParams(severity=1.0)
    c = Coupling(N, p)
    pos, u = fleet()
    vals = [float(load_of(c, p, pos, u, a_val=float(driver_A(torch.tensor([t]), p)))
                  .abs().mean()) for t in range(0, p.period, 5)]
    assert max(vals) > 2 * (min(vals) + 1e-12), (
        f"frozen partners gave a flat load {min(vals):.4g}..{max(vals):.4g}"
    )
    return (
        f"partners frozen, load still swings {min(vals):.4f} -> {max(vals):.4f} "
        "over one cycle -- not category A"
    )


# ===========================================================================
#  the dial
# ===========================================================================


@check("test_identity_at_zero_is_exact")
def _zero():
    """NS-2.1.  Not 'approximately'; equality, over the whole driver domain."""
    p = DialParams(severity=0.0)
    c = Coupling(N, p)
    pos, u = fleet()
    for t in range(p.period):
        a = float(driver_A(torch.tensor([t]), p))
        assert float(load_of(c, p, pos, u, a_val=a).abs().max()) == 0.0, t
    return f"sigma=0 gives exactly 0.0 at all {p.period} driver values"


@check("test_monotone_in_severity")
def _mono():
    p0 = DialParams()
    c = Coupling(N, p0)
    pos, u = fleet()
    prev = -1.0
    for sigma in (0.0, 0.5, 1.0, 2.0, 4.0):
        p = DialParams(severity=sigma)
        v = float(load_of(c, p, pos, u).abs().mean())
        assert v >= prev - 1e-12, f"load fell from {prev} to {v} at sigma={sigma}"
        prev = v
    return f"monotone in sigma at the driver peak, up to {prev:.4f}"


@check("test_placebo_regime_is_exactly_inert")
def _placebo():
    """NS-2.5.  Half of every cycle is EXACTLY quiet, at every severity."""
    p = DialParams(severity=3.0)
    c = Coupling(N, p)
    pos, u = fleet()
    dry = [t for t in range(p.period) if float(driver_A(torch.tensor([t]), p)) == 0.0]
    assert len(dry) >= p.period // 2 - 1, f"only {len(dry)} exactly-dry steps"
    for t in dry:
        a = float(driver_A(torch.tensor([t]), p))
        assert float(load_of(c, p, pos, u, a_val=a).abs().max()) == 0.0, t
    return f"{len(dry)}/{p.period} steps exactly inert at sigma=3, bit for bit"


@check("test_driver_is_a_function_of_time_alone")
def _exogenous():
    """NS-1.3.  A(t) depends on the clock and on nothing else."""
    p = DialParams()
    t = torch.arange(3 * p.period)
    a = driver_A(t, p)
    assert torch.equal(a[: p.period], a[p.period : 2 * p.period]), "not periodic"
    assert float(a.min()) == 0.0 and abs(float(a.max()) - 1.0) < 1e-6
    return (
        f"periodic, range [0, 1] exactly, cycle mean {cycle_mean_A(p):.4f} "
        "(half the cycle is exactly zero)"
    )


# ===========================================================================
#  the operator
# ===========================================================================


@check("test_operator_is_zero_diagonal_spread_and_asymmetric")
def _operator():
    """NS-1.2.  A flat proxy measured a fit gain of -0.0045 on the source
    implementation -- worse than an intercept-only null -- so all three
    properties are load-bearing rather than decorative."""
    p = DialParams()
    c = Coupling(N, p)
    pos, _ = fleet()
    st = c.operator_stats(pos)
    assert st["diag_max"] == 0.0, f"W has a non-zero diagonal: {st['diag_max']}"
    assert st["spread"] > 0.1, f"W is nearly flat: spread {st['spread']:.4f}"
    assert st["asymmetry"] > 0.01, f"W is nearly symmetric: {st['asymmetry']:.4f}"
    return (
        f"zero diagonal, spread {st['spread']:.3f}, asymmetry {st['asymmetry']:.3f}"
    )


@check("test_channels_equal_the_brute_force_definition")
def _gate1():
    """P-3.2, at startup rather than in a test somebody can skip."""
    p = DialParams()
    c = Coupling(N, p)
    pos, u = fleet()
    return c.verify(pos, u)


@check("test_r_is_independent_of_the_number_of_agents")
def _rank():
    """P-1.1.  If the reduction has a parameter per agent it is not this method."""
    p = DialParams()
    dims = {Coupling(n, p).r for n in (2, 5, 11, 40)}
    assert dims == {p.n_types}, dims
    return f"r = {p.n_types} at N in 2, 5, 11, 40 -- no parameter per agent"


@check("test_class_constants_are_declared_not_fitted")
def _declared():
    """P-1.2.  Deterministic in the class index alone: no RNG, no run data, so
    every arm and every seed gets the same operator."""
    p = DialParams()
    a1, b1 = class_constants(p)
    a2, b2 = class_constants(p)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert abs(float(a1.mean()) - 1.0) < 1e-6 and abs(float(b1.mean()) - 1.0) < 1e-6
    assert float((a1 - b1).abs().max()) > 0.1, "recv and send are not distinguishable"
    return (
        f"recv {[round(float(v),3) for v in a1]} (public) vs "
        f"send {[round(float(v),3) for v in b1]} (unknown), both mean 1"
    )


@check("test_centring_conditions_the_design_matrix")
def _centring():
    """P-3.3.  Uncentred channels against an intercept column measured a
    condition number of ~1.3e5 on the source implementation, at which the
    per-class split is unidentifiable even though prediction is fine."""
    p = DialParams()
    c = Coupling(N, p)
    pos, u = fleet()
    Q = torch.zeros(pos.shape[0], c.n, c.r, 2)
    for _ in range(4):
        Q, _, _, x = c.step_channels(pos, u, Q)
    #  The reference is the host's own spawn geometry at run time; offline there
    #  is no host, so the fleet layout stands in for it.  What is being checked
    #  is that centring CONDITIONS the design matrix, which does not depend on
    #  which layout is used -- only on using the same one the channels come from.
    ref, scale = c.geometric_reference(pos, samples=64)
    raw = torch.cat([torch.ones_like(x[..., :1]), x], -1).reshape(-1, 1 + c.r)
    cen = c.design(x, ref, scale).reshape(-1, 1 + c.r)
    k_raw = float(torch.linalg.cond(raw.T @ raw))
    k_cen = float(torch.linalg.cond(cen.T @ cen))
    assert k_cen < k_raw, f"centring made it worse: {k_raw:.3g} -> {k_cen:.3g}"
    return f"condition number {k_raw:.3g} -> {k_cen:.3g}"


def main() -> int:
    width = 78
    print("=" * width)
    print("simple_ns -- offline conformance (torch only)")
    print("=" * width)
    failed = 0
    for name, ok, detail in RESULTS:
        failed += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"       {detail}")
    print("-" * width)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
