#!/usr/bin/env python
#  I.7 -- the conformance suite.  Fourteen offline checks.
#
#      python road_ns/conformance.py
#
#  No simulator, no learning framework.  Test names are the source
#  implementation's own, as I.7 instructs: port the test, not just the prose.
#  Each corresponds to a numbered requirement and each has failed at least once.

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_ns.ceiling import decompose, element_load, fleet_scan  # noqa: E402
from road_ns.dial import (  # noqa: E402
    DialParams,
    dial_g,
    driver_A,
    harm,
    loading,
    performance,
    sensitivity,
)
from road_ns.structure import load_structure, loop_assignment  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []
STRUCT = load_structure()
#: THE COMMITTED INSTANCE: road_traffic map_type=1 (full map), its own seven
#: reference loops, its own agent->loop assignment, 40 agents (path_to_loop's
#: maximum).  Chosen on the Part C decomposition BEFORE any method code ran
#: (NS-4.1): it is the only route set that satisfies NS-1.2's spread
#: requirement while keeping loading in URB's measured regime.
ROUTES = STRUCT.declared_routes("loops")
N_AGENTS = 16


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


def _fleet(n=N_AGENTS, seed=0):
    # The scenario's OWN assignment, not a random one. seed rotates which
    # slots are used so the checks see more than a single fleet.
    a = loop_assignment(40)
    return [a[(i + seed) % len(a)] for i in range(n)]


# ===========================================================================
#  operator
# ===========================================================================


@check("test_operator_is_zero_diagonal")
def _zero_diag():
    for n in (1, 2, 10, 20, 40):
        W = STRUCT.coupling(_fleet(n), ROUTES)
        assert torch.all(W.diagonal() == 0), f"N={n}"
    return "N in {1,2,10,20,40}: the own-effect never enters the coupling operator"


@check("test_operator_is_asymmetric_and_spread")
def _spread():
    W = STRUCT.coupling(_fleet(), ROUTES)
    sp, asym = STRUCT.spread(W), STRUCT.asymmetry(W)
    assert sp > 0.3, (
        f"spread std/mean = {sp:.3f} is too flat. A flat geometric proxy "
        "measured fit gain -0.0045 in the source implementation -- worse than "
        "an intercept-only null."
    )
    assert asym > 0.0, f"operator came out exactly symmetric (asym={asym:.4f})"
    off = W[~torch.eye(N_AGENTS, dtype=torch.bool)]
    nz = off[off > 0]
    return (
        f"spread={sp:.3f} asym={asym:.3f} "
        f"weights span {float(nz.max() / nz.min()):.1f}x "
        f"({nz.numel()}/{off.numel()} pairs coupled)"
    )


@check("test_n1_gives_exactly_zero_peer_load")
def _n1():
    W = STRUCT.coupling(_fleet(1), ROUTES)
    assert W.shape == (1, 1) and float(W.abs().max()) == 0.0
    return "a lone agent reads exactly zero peer load, structurally"


# ===========================================================================
#  dial
# ===========================================================================


@check("test_dial_identity_at_zero_is_exact")
def _identity():
    p = DialParams(severity=0.0)
    s = sensitivity(STRUCT, p)
    a = driver_A(torch.arange(0, 4 * p.period), p)
    g = dial_g(a, s, p)
    assert torch.all(g == 1.0), (
        f"g deviated from 1 at sigma=0: min {float(g.min()):.17g}. This must be "
        "exact, not approximate -- it is what lets the same command produce the "
        "no-severity row."
    )
    return f"{g.numel()} (step, element) pairs, all exactly 1.0, over 4 full cycles"


@check("test_dial_is_monotone_and_never_generous")
def _monotone():
    a = driver_A(torch.arange(0, 200), DialParams())
    prev = None
    for sigma in (0.0, 0.5, 1.0, 2.0, 3.0):
        p = DialParams(severity=sigma)
        g = dial_g(a, sensitivity(STRUCT, p), p)
        assert float(g.max()) <= 1.0, f"g > 1 at sigma={sigma}"
        if prev is not None:
            bad = int((g > prev + 1e-7).sum())
            assert bad == 0, (
                f"sigma={sigma} made the task EASIER at {bad} points. Scaling a "
                "two-sided physical law by sigma made POWER strictly easier at "
                "sigma=2."
            )
        prev = g
    return "5 severities x 200 steps x 104 elements: monotone and never generous"


@check("test_placebo_regime_is_exactly_inert")
def _placebo():
    p0 = DialParams()
    a = driver_A(torch.arange(0, p0.period), p0)
    dry = a == 0.0
    assert int(dry.sum()) > 0, "no exactly-dry step exists"
    ref = None
    for sigma in (0.0, 1.0, 3.0, 10.0):
        p = DialParams(severity=sigma)
        g = dial_g(a[dry], sensitivity(STRUCT, p), p)
        assert torch.all(g == 1.0), f"placebo not inert at sigma={sigma}"
        ref = g if ref is None else ref
    return (
        f"{int(dry.sum())}/{p0.period} steps exactly dry, g == 1.0000 bit for bit "
        "at every sigma -- the rig switches itself off half the time"
    )


@check("test_dial_anchored_to_hcm")
def _anchor():
    p = DialParams(severity=1.0)
    assert abs(p.loss_at_sigma1 - 0.14) < 1e-12, "anchor is not the HCM 14%"
    s = sensitivity(STRUCT, p)
    g = dial_g(torch.tensor([1.0]), s, p)
    uniform = 1.0 - p.loss_at_sigma1
    return (
        f"sigma=1 anchored at the HCM heavy-rain capacity adjustment factor "
        f"(loss 0.14, g={uniform:.3f} at unit sensitivity); with per-element "
        f"s_a the peak spans g in [{float(g.min()):.3f}, {float(g.max()):.3f}]. "
        "sigma > 1 is a beyond-physical stress test."
    )


# ===========================================================================
#  harm
# ===========================================================================


@check("test_harm_is_identity_when_the_dial_is_off")
def _harm_identity():
    p = DialParams(severity=0.0)
    u = torch.rand(64, 8) * 0.5
    assert torch.all(harm(u, u, p) == 1.0)
    p1 = DialParams(severity=3.0)
    zero = torch.zeros(64, 8)
    assert torch.all(harm(zero, zero, p1) == 1.0), (
        "a lone agent (u=0) must read harm exactly 1.0 at any severity -- if a "
        "single agent suffers, the driver is adding to the loss somewhere"
    )
    return "harm == 1.0 exactly at sigma=0, and at u=0 for any sigma"


@check("test_harm_is_monotone_in_severity")
def _harm_monotone():
    prev = None
    for sigma in (0.0, 0.5, 1.0, 2.0):
        p = DialParams(severity=sigma)
        s = sensitivity(STRUCT, p)
        g = dial_g(torch.tensor([1.0]), s, p)
        u_nom = torch.full((1, 6), 0.20)
        u_der = u_nom / g[:, :6].clamp_min(1e-6)
        h = harm(u_der, u_nom, p)
        if prev is not None:
            assert float(h.mean()) >= float(prev.mean()) - 1e-7
        prev = h
    return f"harm rises with sigma; at sigma=2 it is {float(prev.mean()):.4f}x free-flow"


@check("test_sensor_uses_the_binding_element_not_the_mean")
def _binding():
    p = DialParams()
    load = torch.zeros(1, STRUCT.n_elements)
    route = ROUTES[0]
    for a in route:
        load[0, a] = 1.0
    load[0, route[-1]] = 40.0  # one badly congested element on the route
    g = torch.ones(1, STRUCT.n_elements)
    u, binding = loading(load, g, STRUCT, [route])
    ratios = load[0, list(route)] / STRUCT.capacity[list(route)]
    assert abs(float(u[0, 0]) - float(ratios.max())) < 1e-6, "sensor is not the max"
    assert float(u[0, 0]) > float(ratios.mean()) * 1.5, "max did not dominate the mean"
    assert int(binding[0, 0]) == route[-1]
    return (
        f"max {float(ratios.max()):.3f} vs mean {float(ratios.mean()):.3f} on the "
        "same route -- averaging would dilute the signal toward zero"
    )


# ===========================================================================
#  ceiling
# ===========================================================================


@check("test_ceiling_shares_are_a_partition")
def _partition():
    p = DialParams()
    fleet = _fleet()
    d = decompose(STRUCT, ROUTES, fleet[:16], fleet[16:], p)
    total = d.irreducible + d.own + d.peer
    assert abs(total - 1.0) < 1e-5, f"shares sum to {total}"
    assert min(d.irreducible, d.own, d.peer) >= 0.0
    return f"irreducible {d.irreducible:.1%} + own {d.own:.1%} + peer {d.peer:.1%} = 1"


@check("test_all_controllable_has_no_irreducible_share")
def _all_controllable():
    p = DialParams()
    fleet = _fleet()
    d = decompose(STRUCT, ROUTES, fleet, [], p)
    assert d.irreducible < 1e-9, f"irreducible = {d.irreducible:.4g} with no background"
    return "with no background participants the irreducible share is exactly 0"


@check("test_coordination_gap_grows_with_fleet_share")
def _gap_grows():
    p = DialParams()
    rows = fleet_scan(STRUCT, ROUTES, p, n_total=N_AGENTS)
    gaps = [r.peer for r in rows]
    assert gaps[-1] > gaps[0] + 1e-6, f"gap did not grow: {gaps}"
    for a, b in zip(gaps, gaps[1:]):
        assert b >= a - 1e-6, f"gap fell: {gaps}"
    return "  ".join(
        f"{r.n_controllable}/{r.n_controllable + r.n_background}:{r.peer:.1%}"
        for r in rows
    )


@check("test_placebo_produces_no_excess_to_attribute")
def _placebo_no_excess():
    p = DialParams(severity=3.0)
    fleet = _fleet()
    d = decompose(STRUCT, ROUTES, fleet[:16], fleet[16:], p, driver_value=0.0)
    assert d.g_mean == 1.0, f"g = {d.g_mean} on a dry day"
    return "on a dry day there is no excess to attribute, at any severity"


# ===========================================================================


def main() -> int:
    width = 78
    print("=" * width)
    print("road_ns conformance   (I.7 -- offline, no simulator, no learning)")
    print(STRUCT.banner())
    print(f"instance: full-map loops, {len(ROUTES)} routes, N={N_AGENTS}, "
          f"hop counts {sorted({len(r) for r in ROUTES})}")
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
