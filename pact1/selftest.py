#!/usr/bin/env python
#  PACT-1 offline self-test.  Build-order step 5.
#
#      python pact1/selftest.py
#
#  Seconds, torch only, no simulator.  If beta is not recovered here the
#  arithmetic is wrong, not the domain -- and that is worth knowing before any
#  compute is spent.  Test names are the spec's own.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact1.core import (  # noqa: E402
    Basis,
    PactParams,
    RLS,
    confidence,
    herd_index,
    steer,
    trust_from_logit,
)
from road_ns.structure import load_structure  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []
STRUCT = load_structure()
ROUTES = STRUCT.routes()
NAMES, CLS = STRUCT.element_classes()
P = PactParams()
N = 20


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


def make_basis() -> Basis:
    return Basis(STRUCT.capacity, CLS, len(NAMES), ROUTES, P)


def fleet(n=N, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, len(ROUTES), (n,), generator=gen).tolist()


# ===========================================================================


@check("test_basis_zero_diagonal_and_N1")
def _zero_diag():
    b = make_basis()
    x1 = b.channels(fleet(1))
    assert float(x1.abs().max()) == 0.0, "a lone agent read non-zero peer load"
    x = b.channels(fleet(N))
    assert float(x.abs().max()) > 0.0, "the channels are inert with 20 agents"
    return f"N=1 reads exactly 0 on all {len(NAMES)} channels; N=20 is live"


@check("test_basis_waveform_arithmetic")
def _bruteforce():
    b = make_basis()
    for seed in (0, 1, 2):
        b.verify(fleet(N, seed))
    fast, slow = b.channels(fleet(N)), b.channels_bruteforce(fleet(N))
    return (
        f"vectorised == brute-force definition to "
        f"{float((fast - slow).abs().max()):.2e} over 3 fleets (gate 1)"
    )


@check("test_channel_pruning_keeps_everything_aligned")
def _prune():
    b = make_basis()
    live = b.prune(N)
    assert live == sorted(set(live)), "pruning returned an unsorted/duplicated index"
    assert all(0 <= m < len(NAMES) for m in live)
    ref = b.geometric_reference(N)
    scale = b.scale_reference(N)
    psi = b.design(fleet(N), ref, scale)
    assert psi.shape == (N, 1 + len(live)), psi.shape
    return f"kept {len(live)}/{len(NAMES)} channels {[NAMES[m] for m in live]}, design {tuple(psi.shape)}"


@check("test_channel_variation_flags_a_constant_channel")
def _constant_channel():
    b = make_basis()
    std = b.scale_reference(N)
    assert torch.isfinite(std).all() and float(std.min()) > 0, std
    return f"channel std over random fleets: {[round(float(s), 4) for s in std]}"


@check("test_centring_fixes_the_condition_number")
def _centring():
    """P-3.3.  Uncentred, the raw channels carry a large common mean against an
    intercept column of 1 and the split goes unidentifiable while prediction
    stays fine.  The source implementation measured 1.3e5 -> 24."""
    b = make_basis()
    b.prune(N)
    ref, scale = b.geometric_reference(N), b.scale_reference(N)
    raw, cen = [], []
    for seed in range(60):
        f = fleet(N, seed)
        x = b.channels(f)[:, b.live]
        raw.append(torch.cat([torch.ones(N, 1), x], dim=-1))
        cen.append(b.design(f, ref, scale))
    R = torch.cat(raw), torch.cat(cen)
    c_raw = float(torch.linalg.cond(R[0].T @ R[0]))
    c_cen = float(torch.linalg.cond(R[1].T @ R[1]))
    assert c_cen < c_raw, f"centring made conditioning worse: {c_raw:.3g} -> {c_cen:.3g}"
    return f"cond(design'design): uncentred {c_raw:.3g} -> centred {c_cen:.3g}"


@check("test_rls_recovers_known_beta")
def _recover():
    b = make_basis()
    b.prune(N)
    ref, scale = b.geometric_reference(N), b.scale_reference(N)
    dim = 1 + len(b.live)
    true = torch.tensor([0.30] + [0.7 - 0.2 * m for m in range(dim - 1)])
    r = RLS(N, dim, P)
    for t in range(3000):
        psi = b.design(fleet(N, seed=t), ref, scale)
        y = (psi * true.unsqueeze(0)).sum(-1)
        r.update(psi, y)
    err = float((r.beta - true.unsqueeze(0)).abs().max())
    assert err < 1e-2, f"max |beta - true| = {err:.4g}; true={true.tolist()}"
    return f"max |beta - true| = {err:.2e} over 3000 rows, mu={P.mu}"


@check("test_end_to_end_recovers_a_known_congestion_law")
def _known_law():
    """A known linear congestion law, plus noise, over the REAL basis."""
    b = make_basis()
    b.prune(N)
    ref, scale = b.geometric_reference(N), b.scale_reference(N)
    dim = 1 + len(b.live)
    true = torch.tensor([0.25] + [0.9, 0.45, 0.15][: dim - 1])
    r = RLS(N, dim, P)
    gen = torch.Generator().manual_seed(3)
    sse_full = sse_null = sst = 0.0
    ybar = 0.0
    for t in range(4000):
        psi = b.design(fleet(N, seed=10_000 + t), ref, scale)
        y = (psi * true.unsqueeze(0)).sum(-1) + 0.02 * torch.randn(
            N, generator=gen
        )
        resid = r.update(psi, y)
        if t > 500:
            sse_full += float(resid.pow(2).sum())
            ybar = 0.99 * ybar + 0.01 * float(y.mean())
            sse_null += float((y - ybar).pow(2).sum())
            sst += float((y - ybar).pow(2).sum())
    fit_gain = (sse_null - sse_full) / max(sst, 1e-12)
    err = float((r.beta - true.unsqueeze(0)).abs().max())
    assert fit_gain > 0.5, f"fit gain over an intercept-only null is only {fit_gain:.3f}"
    assert err < 5e-2, f"max |beta - true| = {err:.3g}"
    return f"fit_gain over the null = {fit_gain:.3f}, max |beta - true| = {err:.2e}"


@check("test_rls_dead_row_does_not_inflate_covariance")
def _dead_row():
    """P-4.2.  A dead row carries no information but would still divide P by mu
    every step, silently tightening the effective forgetting factor."""
    r_live = RLS(4, 3, P)
    r_dead = RLS(4, 3, P)
    zero = torch.zeros(4, 3)
    for _ in range(2000):
        r_dead.update(zero, torch.zeros(4))
    tr_dead = float(r_dead.P.diagonal(dim1=-2, dim2=-1).sum(-1).max())
    tr_init = float(r_live.P.diagonal(dim1=-2, dim2=-1).sum(-1).max())
    assert abs(tr_dead - tr_init) < 1e-6, (
        f"2000 dead rows changed tr(P) {tr_init:.4f} -> {tr_dead:.4f}; they must "
        "be skipped entirely"
    )
    naive = tr_init * (1.0 / P.mu) ** 2000
    assert float(r_dead.n_skipped.max()) == 2000
    return (
        f"tr(P) held at {tr_dead:.3f} over 2000 dead rows; feeding them would "
        f"have taken it to {naive:.3g}"
    )


@check("test_rls_tracks_drift")
def _drift():
    b = make_basis()
    b.prune(N)
    ref, scale = b.geometric_reference(N), b.scale_reference(N)
    dim = 1 + len(b.live)
    r = RLS(N, dim, P)
    for t in range(6000):
        drift = 0.3 + 0.5 * (t / 6000.0)  # beta* moves along the performance curve
        true = torch.tensor([0.2] + [drift] * (dim - 1))
        psi = b.design(fleet(N, seed=20_000 + t), ref, scale)
        r.update(psi, (psi * true.unsqueeze(0)).sum(-1))
    final = torch.tensor([0.2] + [0.8] * (dim - 1))
    err = float((r.beta - final.unsqueeze(0)).abs().max())
    assert err < 0.1, f"failed to track drift: max err {err:.3g}"
    return f"beta* drifted 0.3 -> 0.8; tracked to within {err:.3f}"


@check("test_trust_prior_is_inverted")
def _prior():
    g0 = float(trust_from_logit(torch.zeros(1), P))
    assert g0 > 0.85, f"w=0 gives trust {g0:.3f}; P-5.1 requires near-full reliance"
    return (
        f"w=0 -> trust {g0:.3f} of g_max (bias {P.trust_bias}). Starting at 0.5 "
        "measured return 3642 against 5444 with a 0.99-accurate estimate."
    )


@check("test_confidence_cold_to_warm")
def _conf():
    b = make_basis()
    b.prune(N)
    ref, scale = b.geometric_reference(N), b.scale_reference(N)
    dim = 1 + len(b.live)
    r = RLS(N, dim, P)
    psi0 = b.design(fleet(N), ref, scale)
    cold = float(confidence(psi0, r.P, P, dim).mean())
    true = torch.tensor([0.3] + [0.6] * (dim - 1))
    for t in range(1500):
        psi = b.design(fleet(N, seed=30_000 + t), ref, scale)
        r.update(psi, (psi * true.unsqueeze(0)).sum(-1))
    warm = float(confidence(psi0, r.P, P, dim).mean())
    assert warm > cold, f"confidence did not rise: {cold:.4f} -> {warm:.4f}"
    return f"prediction confidence {cold:.4f} (cold) -> {warm:.4f} (warm)"


@check("test_pred_confidence_survives_dead_excitation")
def _conf_survives():
    """P-5.2.  The trace gate would disarm here; the prediction gate must not."""
    b = make_basis()
    b.prune(N)
    ref, scale = b.geometric_reference(N), b.scale_reference(N)
    dim = 1 + len(b.live)
    r = RLS(N, dim, P)
    frozen = b.design(fleet(N, seed=7), ref, scale)  # excitation dies: one row forever
    true = torch.tensor([0.3] + [0.6] * (dim - 1))
    for _ in range(4000):
        r.update(frozen, (frozen * true.unsqueeze(0)).sum(-1))
    trP = float(r.P.diagonal(dim1=-2, dim2=-1).sum(-1).mean())
    trace_gate = 1.0 / (1.0 + trP / P.p0)
    pred_gate = float(confidence(frozen, r.P, P, dim).mean())
    assert pred_gate > 0.9, f"prediction gate fell to {pred_gate:.4f}"
    assert trace_gate < pred_gate, "the trace gate did not reproduce its failure"
    return (
        f"excitation frozen 4000 steps: prediction gate {pred_gate:.4f} (still "
        f"armed) vs trace gate {trace_gate:.4f} (disarmed). tr(P)={trP:.3g}"
    )


@check("test_floor_property_is_exact")
def _floor():
    """P-7.1.  Bit for bit, for any beta-hat, however wrong."""
    gen = torch.Generator().manual_seed(11)
    for _ in range(200):
        v = torch.randn(N, generator=gen)
        pred = torch.randn(N, generator=gen) * 1e6  # a diverged estimate
        out = steer(v, pred, torch.zeros(N), P)
        assert torch.equal(out, v), "g=0 was not a bit-for-bit no-op"
    same = steer(torch.randn(N, generator=gen), torch.full((N,), 3.0), torch.ones(N), P)
    assert torch.isfinite(same).all(), "identical predictions produced NaN"
    v = torch.randn(N, generator=gen)
    assert torch.equal(steer(v, torch.full((N,), 3.0), torch.ones(N), P), v)
    return (
        "g=0 is bit-identical to the untouched command over 200 draws with a "
        "1e6-scale estimate; identical predictions give exactly zero shift, not NaN"
    )


@check("test_steer_direction")
def _direction():
    v = torch.ones(4)
    pred = torch.tensor([0.0, 1.0, 2.0, 3.0])
    out = steer(v, pred, torch.full((4,), 0.5), P)
    assert out[0] > out[1] > out[2] > out[3], out
    assert abs(float(out.mean()) - 1.0) < 1e-6, "a uniform shift should net to zero"
    return (
        "the agent on the most congested route eases off, the one on the "
        "clearest presses on, and the fleet mean is unchanged -- only the "
        "differential shift does anything"
    )


@check("test_herd_index_bounds")
def _herd():
    assert abs(herd_index([3] * 10, len(ROUTES)) - 1.0) < 1e-6, "all-same is not 1"
    spread = herd_index(list(range(10)), len(ROUTES))
    assert abs(spread) < 1e-6, f"perfectly spread is {spread}"
    mid = herd_index(fleet(20), len(ROUTES))
    assert 0.0 <= mid <= 1.0
    return f"all-on-one=1.000, perfectly-spread=0.000, random fleet={mid:.3f}"


# ===========================================================================


def main() -> int:
    width = 78
    print("=" * width)
    print("PACT-1 self-test   (build-order step 5 -- offline, no simulator)")
    print(f"basis: {len(NAMES)} element classes {NAMES}")
    print(f"mu={P.mu} p0={P.p0} trust_bias={P.trust_bias} kappa={P.kappa}")
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
