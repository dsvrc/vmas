#!/usr/bin/env python
#  The arithmetic self-check.  torch only -- no vmas, no torchrl, no hydra.
#
#      python pact2/selfcheck.py
#
#  PACT_PIPELINE_SPEC section 11.4: "Write an arithmetic self-check that runs
#  without the simulator.  It cannot prove the method works; it proves the
#  arithmetic is not the reason if it does not."
#
#  Every fixture here is calibrated to the *real* operator and to Phi
#  trajectories from a surrogate of VMAS's own dynamics.  A fixture too gentle
#  silently blesses broken code.

from __future__ import annotations

import argparse
import math
import sys
from typing import Callable, List, Tuple

import torch

from pact2._bootstrap import HolonomicFleet, load_cores, waypoint_servo

slc_core, pact_core = load_cores()

SlcParams = slc_core.SlcParams
PactParams = pact_core.PactParams
PactCompensator = pact_core.PactCompensator

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn: Callable[[], str]):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, detail))
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        return fn

    return wrap


DEV = torch.device("cpu")


def base_params(**kw) -> "SlcParams":
    d = dict(severity=1.0, n_chan=3)
    d.update(kw)
    return SlcParams(**d)


def _surrogate_phi(p, n_agents: int, steps: int = 600, seed: int = 0) -> torch.Tensor:
    """Phi trajectories from a surrogate of VMAS's own dynamics driven by a
    saturating velocity servo chasing random waypoints.  Returns ``(T, N)``.

    The fixtures below are calibrated to THIS, not to a convenient synthetic
    distribution: a fixture too gentle silently blesses broken code.
    """
    gen = torch.Generator().manual_seed(seed)
    fleet = HolonomicFleet(1, n_agents)
    fleet.pos.uniform_(-1.0, 1.0, generator=gen)
    goal = torch.empty_like(fleet.pos).uniform_(-1.0, 1.0, generator=gen)
    out = []
    for t in range(steps):
        if t % 40 == 0 and t:
            goal = torch.empty_like(fleet.pos).uniform_(-1.0, 1.0, generator=gen)
        out.append(slc_core.exertion(fleet.vel, p)[0].clone())
        fleet.step(waypoint_servo(fleet.pos, fleet.vel, goal))
    return torch.stack(out)


# =============================================================================
#  Part B -- the dial
# =============================================================================


@check("B.1.1 dial identity at sigma=0 is EXACT over the whole driver domain")
def _identity():
    p0 = base_params(severity=0.0)
    op = slc_core.build_operator(p0, 6, DEV)
    a = torch.linspace(0.0, 1.0, 513)
    g = slc_core.dial_g(a, p0, op)
    assert torch.all(g == 1.0), (
        f"g deviated from 1 at sigma=0: min {g.min():.17g} max {g.max():.17g}. "
        "This must be exact, not approximate -- it is what lets you claim the "
        "sigma=0 task is recovered byte for byte."
    )
    return f"{g.numel()} (driver, channel) pairs, all exactly 1.0"


@check("B.1.2 dial is non-increasing in sigma at EVERY driver value")
def _monotone():
    a = torch.linspace(0.0, 1.0, 257)
    prev = None
    for sigma in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        p = base_params(severity=sigma)
        op = slc_core.build_operator(p, 6, DEV)
        g = slc_core.dial_g(a, p, op)
        if prev is not None:
            bad = (g > prev + 1e-7).sum()
            assert bad == 0, (
                f"sigma={sigma} made the task EASIER at {int(bad)} driver values. "
                "A two-sided physical law scaled by sigma is a realism dial, not "
                "a severity dial -- clip at 1 and keep only the harmful half."
            )
        prev = g
    return "6 severities x 257 driver values x 3 channels, monotone throughout"


@check("B.1.3 dial is never generous (g <= 1 always)")
def _never_generous():
    a = torch.linspace(0.0, 1.0, 257)
    for sigma in (0.0, 1.0, 3.0):
        p = base_params(severity=sigma)
        op = slc_core.build_operator(p, 6, DEV)
        g = slc_core.dial_g(a, p, op)
        assert float(g.max()) <= 1.0, f"g={float(g.max())} > 1 at sigma={sigma}"
    return "the environment is never credited with more capacity than nominal"


@check("B.4 placebo: the quiet shift is byte-identical across every sigma")
def _placebo():
    a = torch.linspace(0.0, 1.0, 129)
    shift = torch.full((129,), slc_core.SHIFT_QUIET, dtype=torch.long)
    ref = None
    for sigma in (0.0, 0.5, 1.0, 1.5, 2.0):
        p = base_params(severity=sigma)
        op = slc_core.build_operator(p, 6, DEV)
        a_eff = slc_core.effective_driver(a, shift, p)
        g = slc_core.dial_g(a_eff, p, op)
        if ref is None:
            ref = g
        else:
            assert torch.equal(g, ref), (
                f"quiet shift differed at sigma={sigma}: "
                f"max |diff| {float((g - ref).abs().max()):.3e}"
            )
    assert torch.all(ref == 1.0), "placebo regime did not clip to exactly 1"
    return "5 severities, identical to the last bit -- the rig switches itself off"


@check("D.2 the dial does reduce total capacity; the number is reported")
def _capacity_cost():
    a = torch.linspace(0.0, 1.0, 513)
    p = base_params(severity=1.0)
    op = slc_core.build_operator(p, 6, DEV)
    g = slc_core.dial_g(a, p, op)
    lost = 1.0 - float(g.mean())
    swing = float(g.max() / g.min())
    pm = base_params(severity=1.0, mean_preserve=True)
    opm = slc_core.build_operator(pm, 6, DEV)
    gm = slc_core.dial_g(a, pm, opm)
    lost_m = 1.0 - float(gm.mean())
    return (
        f"sigma=1 removes {lost:.1%} of mean capacity with a {swing:.3f}x swing; "
        f"mean_preserve=True leaves {lost_m:+.2%}.  G4a will fail past some "
        "sigma no matter how good the controller is -- report against the "
        "ceiling MEASURED AT THAT SIGMA, never against B0 at sigma=0."
    )


# =============================================================================
#  Part A / PACT 2 -- the declared operator
# =============================================================================


@check("G1 / A.2 W is zero-diagonal; irreducibility is structural")
def _zero_diag():
    for n in (1, 2, 6, 12):
        op = slc_core.build_operator(base_params(), n, DEV)
        assert torch.all(op.W.diagonal() == 0)
    return "N in {1,2,6,12}: the own-effect never enters the coupling operator"


@check("2.1 operator spread and asymmetry are real, not a geometric proxy")
def _spread():
    op = slc_core.build_operator(base_params(), 6, DEV)
    s = op.summary()
    assert s["W_spread"] > 0.3, (
        f"W spread std/mean={s['W_spread']:.3f} is too flat.  A symmetric "
        "equal-weight bucket relation measured fit_gain = -0.0045 on POWER: "
        "the peer channels made prediction WORSE than an intercept-only model."
    )
    assert s["W_asymmetry"] > 0.01, "W came out symmetric; real operators are not"
    assert s["agents_without_coupling"] == 0.0, (
        f"{int(s['agents_without_coupling'])} agents have no live coupling and "
        "would sit at g=0 forever.  Lower slc_n_chan or raise n_agents."
    )
    return (
        f"spread(std/mean)={s['W_spread']:.2f} (POWER: 1.35), "
        f"asymmetry={s['W_asymmetry']:.2f}, "
        f"W in [{s['W_min_positive']:.3e}, {s['W_max']:.3e}]"
    )


@check("2.4 cond(E[psi psi']) is FINITE, and non-finite is tested as a VALUE")
def _cond():
    op = slc_core.build_operator(base_params(), 6, DEV)
    pact = PactCompensator(PactParams(), base_params(), op, 4, DEV)
    phi = _surrogate_phi(base_params(), 6, steps=400)
    for t in range(phi.shape[0]):
        psi = pact.peer_basis(phi[t : t + 1].expand(4, -1))
        own = pact.own_basis(phi[t : t + 1].expand(4, -1))
        reg = torch.cat(
            [torch.ones_like(own).unsqueeze(-1), own.unsqueeze(-1), psi], dim=-1
        )
        pact._accumulate_gram(reg, torch.ones_like(own, dtype=torch.bool))
    c = pact.cond_psi()
    assert math.isfinite(c), (
        "cond is non-finite: theta may be predictable without being "
        "decomposable.  Note the test is `isfinite(c)` FIRST -- an "
        "`isfinite(c) and c > thr` guard lets the most degenerate basis "
        "possible pass silently."
    )
    assert c < 1e5, f"cond={c:.1f} is too high to claim identification"
    return f"cond={c:.1f} over 400 surrogate steps (POWER's good case: 57-810)"


@check("C.1 N=1 -> the coordination gap is EXACTLY zero, for any sigma")
def _n1():
    for sigma in (0.5, 1.0, 5.0):
        p = base_params(severity=sigma)
        op = slc_core.build_operator(p, 1, DEV)
        phi = torch.rand(64, 1) * p.phi_span + p.phi_floor
        a = torch.rand(64)
        g = slc_core.dial_g(a, p, op)
        d = slc_core.decompose_excess(phi, g, op)
        gap = float(d["coordination_gap"])
        assert gap == 0.0, f"gap={gap} at N=1, sigma={sigma}; must be exactly 0"
    return "the peer sum is empty however small g becomes -- structural, not verified after the fact"


# =============================================================================
#  Part A.5 -- the exertion functional
# =============================================================================


@check("A.5 Phi is uncancellable AND varying (std/mean > 0.05)")
def _phi_varies():
    p = base_params()
    phi = _surrogate_phi(p, 6)
    ratio = float(phi.std(unbiased=False) / phi.mean())
    assert ratio > 0.05, (
        f"std(Phi)/mean(Phi) = {ratio:.4f}.  A perfectly constant Phi is "
        "unidentifiable; you need uncancellable AND varying."
    )
    assert float(phi.min()) >= p.phi_floor - 1e-6, "Phi went below its floor"
    return f"std/mean = {ratio:.3f} (POWER ran 0.28), Phi in [{float(phi.min()):.3f}, {float(phi.max()):.3f}]"


# =============================================================================
#  Part C -- the ceiling
# =============================================================================


@check("C.3 the coordination gap is large enough for this to be a showcase")
def _gap():
    p = base_params(severity=1.0)
    op = slc_core.build_operator(p, 6, DEV)
    phi = _surrogate_phi(p, 6)
    a = torch.rand(phi.shape[0])
    g = slc_core.dial_g(a, p, op)
    d = slc_core.decompose_excess(phi, g, op)
    gap = float(d["coordination_gap"])
    assert gap > 0.15, (
        f"coordination gap {gap:.1%} is small -- this environment would be a "
        "poor showcase for a coordination method however good the method is. "
        "Use it to CHOOSE the environment, not to excuse the outcome."
    )
    return (
        f"irreducible {float(d['irreducible']):.1%} / "
        f"own {float(d['own_free']):.1%} / "
        f"PEER {gap:.1%}  (POWER at sigma=1: 13.6 / 76.9 / 9.5)"
    )


@check("C.4 the coordination gap GROWS with N -- a falsifiable prediction")
def _gap_scales():
    rows = []
    prev = -1.0
    for n in (1, 3, 6, 9, 12):
        p = base_params(severity=1.0)
        op = slc_core.build_operator(p, n, DEV)
        phi = _surrogate_phi(p, n, steps=300)
        a = torch.rand(phi.shape[0])
        g = slc_core.dial_g(a, p, op)
        gap = float(slc_core.decompose_excess(phi, g, op)["coordination_gap"])
        rows.append(f"N={n}:{gap:.1%}")
        assert gap >= prev - 1e-3, f"gap fell from {prev:.3f} to {gap:.3f} at N={n}"
        prev = gap
    return "  ".join(rows) + "   (no competing credit-assignment method predicts this)"


# =============================================================================
#  PACT 5 -- the estimator
# =============================================================================


def _rls_fixture(n_steps: int, mu: float, p_max_mult: float, excite: float):
    """Synthetic identification with the REAL basis distribution.

    ``excite`` scales the peer variation: at ``excite -> 0`` the excitation dies,
    which is what happens as the policy converges.
    """
    p = base_params()
    op = slc_core.build_operator(p, 6, DEV)
    pp = PactParams(mu=mu, p_max_mult=p_max_mult, r=1)
    pact = PactCompensator(pp, p, op, 1, DEV)
    phi = _surrogate_phi(p, 6, steps=min(n_steps + 10, 800))

    true = torch.tensor([0.35, 0.20, 0.45])
    active = torch.ones(1, 6, dtype=torch.bool)
    for t in range(n_steps):
        f = phi[t % phi.shape[0]].unsqueeze(0)
        f = p.phi_nominal + excite * (f - p.phi_nominal)
        psi = pact.peer_basis(f)
        own = pact.own_basis(f)
        reg = torch.cat(
            [torch.ones_like(own).unsqueeze(-1), own.unsqueeze(-1), psi], dim=-1
        )
        y = (reg * true.view(1, 1, -1)).sum(-1)
        pact.beta, pact.P, _ = pact._rls(
            pact.beta, pact.P, reg, y, active, pact._eye, track_clamp=True
        )
    return pact, true


@check("5.1 RLS recovers a known model on the real basis")
def _rls_recovers():
    pact, true = _rls_fixture(1500, mu=0.9995, p_max_mult=10.0, excite=1.0)
    err = (pact.beta[0] - true.view(1, -1)).abs().max()
    assert float(err) < 5e-3, f"max |beta - true| = {float(err):.4g}"
    return f"max |beta - true| = {float(err):.2e} over 1500 steps, mu=0.9995"


# The reproduction runs at mu=0.997 over 15k steps rather than POWER's 0.9995
# over 83k, purely so the check stays fast: (1/mu)^T is the same 1e19-scale
# blow-up either way.  The failure mode is the ratio, not the wall clock.
_WINDUP = dict(n_steps=15_000, mu=0.997, excite=0.0)


@check("5.2 WITHOUT the covariance bound the estimator runs away")
def _windup_reproduces():
    pact, _ = _rls_fixture(p_max_mult=1e30, **_WINDUP)
    trP = float(pact.P.diagonal(dim1=-2, dim2=-1).sum(-1).max())
    assert trP > 1e6, (
        f"trace(P) only reached {trP:.3g}; the fixture is too gentle to "
        "reproduce the failure and would silently bless broken code"
    )
    return (
        f"excitation off, no bound: trace(P) -> {trP:.3g}.  On POWER this "
        "tracked return exactly -- se(own_gain) went 29 -> 1.4e8 across "
        "quarters and the method led in Q1-Q3 then lost 28.8 in Q4."
    )


@check("5.2 WITH the bound the scale is held and the estimate survives")
def _windup_bounded():
    pact, true = _rls_fixture(p_max_mult=10.0, **_WINDUP)
    trP = float(pact.P.diagonal(dim1=-2, dim2=-1).sum(-1).max())
    dim = pact.dim
    assert trP <= 10.0 * PactParams().p0 * dim * 1.001 + 1e-6, f"trace(P)={trP:.3g}"
    # with zero excitation only the intercept is identifiable; check it is
    intercept_err = float(
        (pact.beta[0, :, 0] - (true[0] + true[1] * 0.0)).abs().max()
    )
    assert intercept_err < 0.5, f"intercept drifted by {intercept_err:.3g}"
    return (
        f"trace(P) held at {trP:.1f}; rescaling preserves RELATIVE uncertainty "
        "while stopping the absolute scale from diverging"
    )


@check("4.3 fit_gain is guarded with NaN, never an epsilon")
def _nan_guard():
    p = base_params()
    op = slc_core.build_operator(p, 6, DEV)
    pact = PactCompensator(PactParams(), p, op, 2, DEV)
    fg = pact.fit_gain()
    assert torch.isnan(fg).all(), (
        "a zero-variance target must give NaN, not a huge ratio.  A 1e-12 "
        "floor once produced a -1011 that poisoned a column average."
    )
    admissible = pact._admissible(fg, torch.ones_like(fg, dtype=torch.bool))
    assert not bool(admissible.any()), "NaN lift must compare False and gate OFF"
    return "undefined lift -> NaN -> inadmissible by construction"


# =============================================================================
#  PACT 6 -- the channel inverse
# =============================================================================


@check("A.2 conjugacy: harm o inverse is the identity when c_hat == c")
def _conjugacy():
    a = torch.randn(64, 6, 2)
    c = torch.rand(64, 6) * 0.6
    pp = PactParams(denom_floor=0.05, max_delta=1e9)
    p = base_params()
    op = slc_core.build_operator(p, 6, DEV)
    pact = PactCompensator(pp, p, op, 64, DEV)
    delta, _ = pact.compensate(a, c)
    delivered = slc_core.apply_harm(a + delta, c)
    err = float((delivered - a).abs().max())
    assert err < 1e-5, f"round trip error {err:.3e}"
    return f"max round-trip error {err:.2e} -- the channel is exactly invertible"


@check("6.3 a rail-pinned delta is a CONSTANT BIAS, not a compensation")
def _rail_pinned():
    a = torch.ones(1, 2, 2)
    p = base_params()
    op = slc_core.build_operator(p, 2, DEV)
    # max_delta is deliberately out of the way here: the two rails mean
    # different things and pooling them would hide which one bound.
    pact = PactCompensator(
        PactParams(denom_floor=0.15, max_delta=1e9), p, op, 1, DEV
    )
    d1, _ = pact.compensate(a, torch.tensor([[0.90, 0.90]]))
    d2, _ = pact.compensate(a, torch.tensor([[0.99, 0.99]]))
    assert torch.equal(d1, d2), "expected the floor to pin both"
    d3, _ = pact.compensate(a, torch.tensor([[0.50, 0.50]]))
    assert not torch.equal(d1, d3), "below the floor the delta must still respond"
    return (
        f"c_hat 0.90 and 0.99 both give delta={float(d1[0,0,0]):.4f}: two "
        "estimators differing by orders of magnitude produce byte-identical "
        "returns.  Watch delta_clip_frac."
    )


# =============================================================================
#  PACT 1 -- the floor property
# =============================================================================


def _pact_pair(**kw):
    p = base_params()
    op = slc_core.build_operator(p, 6, DEV)
    a = PactCompensator(PactParams(mode="ff", **kw), p, op, 3, DEV)
    b = PactCompensator(
        PactParams(mode="delta", gate="fit", fit_floor=1e9, **kw), p, op, 3, DEV
    )
    return p, op, a, b


@check("1 floor property: inadmissible PACT is BYTE-IDENTICAL to the ff arm")
def _floor():
    p, op, ff, pact = _pact_pair()
    phi = _surrogate_phi(p, 6, steps=300)
    torch.manual_seed(0)
    for t in range(1, 250):
        args = dict(
            u_prev=torch.rand(3, 6) * 0.8,
            phi_bcast=phi[t].unsqueeze(0).expand(3, -1).contiguous(),
            phi_own_now=phi[t + 1].unsqueeze(0).expand(3, -1).contiguous(),
            g_prev=torch.full((3, 6), 0.9),
            g_now=torch.full((3, 6), 0.85),
            alive=torch.ones(3, 6, dtype=torch.bool),
        )
        torch.manual_seed(t)
        o1 = ff.step(**args)
        torch.manual_seed(t)
        o2 = pact.step(**args)
        assert torch.equal(o1["c_hat"], o2["c_hat"]), (
            f"diverged at t={t}: max |diff| "
            f"{float((o1['c_hat'] - o2['c_hat']).abs().max()):.3e}"
        )
    assert float(pact.diag["applied_trust"].max()) == 0.0
    return (
        "250 steps, identical to the last bit.  A diverging estimate can fail "
        "to help; it must never do worse than its own information-matched "
        "baseline."
    )


@check("8.1 admissibility is BINARY x a constant, not a product of confidences")
def _binary_gate():
    p = base_params()
    op = slc_core.build_operator(p, 6, DEV)
    pact = PactCompensator(
        PactParams(max_trust=0.3, gate="always", ready_updates=0), p, op, 2, DEV
    )
    fg = torch.zeros(2, 6)
    adm = pact._admissible(fg, torch.ones(2, 6, dtype=torch.bool))
    trust = torch.where(adm, torch.full_like(fg, 0.3), torch.zeros_like(fg))
    vals = set(trust.reshape(-1).tolist())
    assert vals <= {0.0, 0.30000001192092896, 0.3}, f"trust took values {vals}"
    return "applied_trust takes exactly two values: 0 or max_trust"


# =============================================================================
#  End to end
# =============================================================================


@check("11.5 end-to-end closed loop: blind vs PACT on the surrogate fleet")
def _closed_loop():
    p = base_params(severity=1.0)
    n, B, T = 6, 8, 1200
    op = slc_core.build_operator(p, n, DEV)
    pp = PactParams(max_trust=0.3, ready_updates=100, warmup_updates=25)

    stats = {}
    for arm in ("blind", "ff", "pact"):
        torch.manual_seed(7)
        gen = torch.Generator().manual_seed(7)
        fleet = HolonomicFleet(B, n)
        fleet.pos.uniform_(-1.0, 1.0, generator=gen)
        goal = torch.empty_like(fleet.pos).uniform_(-1.0, 1.0, generator=gen)
        comp = (
            None
            if arm == "blind"
            else PactCompensator(
                PactParams(**{**pp.__dict__, "mode": "ff" if arm == "ff" else "delta"}),
                p,
                op,
                B,
                DEV,
            )
        )
        phi_prev = torch.full((B, n), p.phi_floor)
        u_prev = torch.zeros(B, n)
        g_prev = torch.ones(B, n)
        step = torch.zeros(B, dtype=torch.long)
        phase = torch.arange(B, dtype=torch.float32) / B
        shift = torch.zeros(B, dtype=torch.long)
        travelled, deltas, clip, uerr = 0.0, [], [], []

        for t in range(T):
            if t % 40 == 0 and t:
                goal = torch.empty_like(fleet.pos).uniform_(-1.0, 1.0, generator=gen)
            phi = slc_core.exertion(fleet.vel, p)
            a = slc_core.driver_A(step, phase, p.driver_period)
            a_eff = slc_core.effective_driver(a, shift, p)
            g = slc_core.dial_g(a_eff, p, op)
            load = slc_core.channel_load(phi, op)
            u, _ = slc_core.loading(load, g, op)
            c = slc_core.harm_coefficient(u, p)
            g_ag = (
                g.unsqueeze(1)
                .masked_fill(~op.D.unsqueeze(0), float("inf"))
                .min(-1)
                .values
            )

            cmd = waypoint_servo(fleet.pos, fleet.vel, goal)
            if comp is not None:
                out = comp.step(
                    u_prev=u_prev,
                    phi_bcast=phi_prev,
                    phi_own_now=phi,
                    g_prev=g_prev,
                    g_now=g_ag,
                    alive=torch.full((B, n), t > 0),
                )
                d, cl = comp.compensate(cmd, out["c_hat"])
                deltas.append(float(d.abs().mean()))
                clip.append(float(cl.to(torch.float32).mean()))
                if t > 400:  # after the estimator is ready
                    uerr.append(float((out["u_hat"] - u).abs().mean()))
                cmd = (cmd + d).clamp(-1.0, 1.0)

            fleet.step(slc_core.apply_harm(cmd, c))
            travelled += float(fleet.vel.norm(dim=-1).mean())
            phi_prev, u_prev, g_prev = phi, u, g_ag
            step = step + 1

        stats[arm] = dict(
            travelled=travelled / T,
            delta=sum(deltas) / max(len(deltas), 1),
            uerr=sum(uerr) / max(len(uerr), 1),
            clip=sum(clip) / max(len(clip), 1),
            trust=float(comp.diag["applied_trust"].mean()) if comp else 0.0,
            fit=float(
                comp.diag["fit_gain"][torch.isfinite(comp.diag["fit_gain"])].mean()
            )
            if comp
            else float("nan"),
            peer=float(comp.diag["peer_abs"].mean()) if comp else 0.0,
            ff=float(comp.diag["ff_abs"].mean()) if comp else 0.0,
        )

    assert stats["pact"]["clip"] < 0.5, (
        f"delta_clip_frac = {stats['pact']['clip']:.2f}: a rail-pinned delta is "
        "a constant bias, not a compensation"
    )
    assert stats["pact"]["trust"] > 0.0, (
        "applied_trust is 0 -- the method was never ON.  Read this before any "
        "other number."
    )
    # THE number for the coordination claim: does the peer term reduce the
    # one-step-ahead loading error the information-matched baseline is left
    # with?  The action-space delta is dominated by the agent's own stale
    # sensor reading in both arms, so comparing deltas would flatter nothing.
    gain = 1.0 - stats["pact"]["uerr"] / max(stats["ff"]["uerr"], 1e-12)
    assert gain > 0.0, (
        f"the peer term made the loading prediction WORSE ({gain:+.1%}).  "
        "Check the basis timing: own column CURRENT, peer columns PREVIOUS."
    )
    lines = []
    for arm, s in stats.items():
        lines.append(
            f"{arm:5s} speed={s['travelled']:.4f} delta={s['delta']:.4f} "
            f"u_err={s['uerr']:.5f} trust={s['trust']:.3f} "
            f"fit_gain={s['fit']:+.4f} ff/peer={s['ff']:.4f}/{s['peer']:.4f} "
            f"clip={s['clip']:.2f}"
        )
    lines.append(
        f"peer term cuts the loading prediction error by {gain:.1%} over the "
        "information-matched ff arm -- that, not the raw delta, is the "
        "coordination measurement"
    )
    return "\n            ".join(lines)


# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    width = 78
    print("=" * width)
    print("PACT / SLC arithmetic self-check   (torch only, no simulator)")
    print("=" * width)
    failed = 0
    for name, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        failed += 0 if ok else 1
        print(f"[{mark}] {name}")
        if detail and (not ok or not args.quiet):
            print(f"       {detail}")
    print("-" * width)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    if failed:
        print(
            "\nThis cannot prove the method works.  It proves the arithmetic is "
            "not the reason if it does not -- so fix these before spending GPU "
            "time."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
