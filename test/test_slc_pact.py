#  Unit tests for the SLC non-stationarity and PACT.
#
#      python test/test_slc_pact.py          # torch only, no simulator
#      pytest test/test_slc_pact.py
#
#  The substantive battery lives in ``pact2/selfcheck.py``, which is written to
#  be read as a REPORT -- each check prints the measured number next to the
#  POWER number it is calibrated against, because a bare green tick hides
#  whether the fixture was strong enough to catch anything.  This file runs that
#  battery under pytest and adds the property tests that are pure invariants.

from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    import pytest
except ModuleNotFoundError:  # runs standalone on a machine without pytest
    pytest = None

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pact2._bootstrap import load_cores  # noqa: E402

slc_core, pact_core = load_cores()
SlcParams = slc_core.SlcParams
PactParams = pact_core.PactParams
PactCompensator = pact_core.PactCompensator
DEV = torch.device("cpu")


def P(**kw):
    d = dict(severity=1.0, n_chan=3)
    d.update(kw)
    return SlcParams(**d)


# ---------------------------------------------------------------------------
#  the full self-check battery, as one test per check
# ---------------------------------------------------------------------------


def _selfcheck_results():
    import pact2.selfcheck as sc  # imported for its side effect: it RUNS

    return sc.RESULTS


def test_selfcheck():
    failures = [f"{n}: {d}" for n, ok, d in _selfcheck_results() if not ok]
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
#  invariants
# ---------------------------------------------------------------------------


def test_w_is_zero_diagonal():
    for n in (1, 2, 3, 6, 9, 12):
        op = slc_core.build_operator(P(), n, DEV)
        assert torch.all(op.W.diagonal() == 0), n


def test_dial_never_exceeds_one():
    for sigma in (0.0, 0.1, 1.0, 3.0):
        op = slc_core.build_operator(P(severity=sigma), 6, DEV)
        g = slc_core.dial_g(torch.linspace(0, 1, 129), P(severity=sigma), op)
        assert float(g.max()) <= 1.0, sigma


def test_dial_identity_is_bit_exact_not_approximate():
    p = P(severity=0.0)
    op = slc_core.build_operator(p, 6, DEV)
    g = slc_core.dial_g(torch.linspace(0, 1, 1025), p, op)
    assert torch.all(g == 1.0)


def test_quiet_shift_is_inert_for_every_severity():
    a = torch.linspace(0, 1, 65)
    shift = torch.full((65,), slc_core.SHIFT_QUIET, dtype=torch.long)
    ref = None
    for sigma in (0.0, 0.7, 1.0, 4.0):
        p = P(severity=sigma)
        op = slc_core.build_operator(p, 6, DEV)
        g = slc_core.dial_g(slc_core.effective_driver(a, shift, p), p, op)
        ref = g if ref is None else ref
        assert torch.equal(g, ref)
    assert torch.all(ref == 1.0)


def test_harm_is_disabled_exactly_when_asked():
    u = torch.rand(32, 6)
    assert torch.all(slc_core.harm_coefficient(u, P(harm_enabled=False)) == 0.0)


def test_harm_inverse_round_trips():
    p = P()
    op = slc_core.build_operator(p, 4, DEV)
    pact = PactCompensator(PactParams(denom_floor=0.01, max_delta=1e9), p, op, 16, DEV)
    a = torch.randn(16, 4, 2)
    c = torch.rand(16, 4) * 0.7
    delta, _ = pact.compensate(a, c)
    assert torch.allclose(slc_core.apply_harm(a + delta, c), a, atol=1e-5)


def test_psi_lands_in_the_declared_box():
    """Per-agent, per-channel scaling by each channel's OWN declared range is
    what keeps the Gram well conditioned.  If psi leaves [-0.5, 0.5] the scaling
    is wrong, not the data."""
    p = P()
    op = slc_core.build_operator(p, 6, DEV)
    pact = PactCompensator(PactParams(), p, op, 4, DEV)
    for phi in (
        torch.full((4, 6), p.phi_floor),
        torch.full((4, 6), p.phi_max),
        torch.rand(4, 6) * p.phi_span + p.phi_floor,
    ):
        psi = pact.peer_basis(phi)
        assert float(psi.abs().max()) <= 0.5 + 1e-5


def _uncoupled_operator(n: int = 6):
    """A fleet with no live coupling.

    The shipped frequency plan cannot produce one -- each robot's ranging band
    sits on a neighbour's data channel by construction, which is exactly why
    ``agents_without_coupling`` reads 0 in the real config.  So the guard is
    exercised by zeroing ``W`` directly rather than by contriving parameters
    that never occur.
    """
    p = P()
    op = slc_core.build_operator(p, n, DEV)
    op.W = torch.zeros_like(op.W)
    return p, op


def test_dead_channel_gives_a_zero_column_not_a_nan():
    p, op = _uncoupled_operator()
    pact = PactCompensator(PactParams(), p, op, 2, DEV)
    psi = pact.peer_basis(torch.rand(2, 6))
    assert torch.isfinite(psi).all()
    assert float(psi.abs().max()) == 0.0


def test_agents_with_no_coupling_never_become_admissible():
    p, op = _uncoupled_operator()
    pact = PactCompensator(PactParams(gate="always", ready_updates=0), p, op, 2, DEV)
    adm = pact._admissible(torch.ones(2, 6), torch.ones(2, 6, dtype=torch.bool))
    assert not bool(adm.any())


def test_shipped_plan_leaves_no_agent_uncoupled():
    for n in (2, 3, 6, 9, 12):
        op = slc_core.build_operator(P(), n if n > 1 else 2, DEV)
        assert float(op.live_peers().min()) > 0.0, n


def test_covariance_stays_symmetric():
    p = P()
    op = slc_core.build_operator(p, 4, DEV)
    pact = PactCompensator(PactParams(), p, op, 2, DEV)
    active = torch.ones(2, 4, dtype=torch.bool)
    for _ in range(200):
        reg = torch.randn(2, 4, pact.dim)
        pact.beta, pact.P, _ = pact._rls(
            pact.beta, pact.P, reg, torch.randn(2, 4), active, pact._eye
        )
    assert torch.allclose(pact.P, pact.P.transpose(-1, -2), atol=1e-6)


def test_reset_keeps_beta_and_clears_history():
    p = P()
    op = slc_core.build_operator(p, 6, DEV)
    pact = PactCompensator(PactParams(), p, op, 3, DEV)
    pact.beta += 1.0
    pact.have_lag[:] = True
    pact.psi_lag += 2.0
    beta = pact.beta.clone()
    pact.reset(1)
    assert torch.equal(pact.beta, beta)
    assert not bool(pact.have_lag[1].any())
    assert bool(pact.have_lag[0].all()) and bool(pact.have_lag[2].all())
    assert float(pact.psi_lag[1].abs().max()) == 0.0
    assert float(pact.psi_lag[0].abs().max()) > 0.0


def test_ceiling_shares_sum_to_one():
    p = P()
    op = slc_core.build_operator(p, 6, DEV)
    phi = torch.rand(128, 6) * p.phi_span + p.phi_floor
    g = slc_core.dial_g(torch.rand(128), p, op)
    d = slc_core.decompose_excess(phi, g, op)
    total = (
        float(d["irreducible"]) + float(d["own_free"]) + float(d["coordination_gap"])
    )
    assert abs(total - 1.0) < 1e-5


def test_capacity_mode_fixed_really_congests_with_n():
    """``scaled`` provisions for the fleet so the N-sweep isolates the
    coordination gap; ``fixed`` does not, and loading must then rise with N."""
    prev = -1.0
    for n in (3, 6, 12):
        p = P(capacity_mode="fixed", capacity_ref_agents=6)
        op = slc_core.build_operator(p, n, DEV)
        phi = torch.full((16, n), p.phi_nominal)
        g = torch.ones(16, op.n_chan)
        u, _ = slc_core.loading(slc_core.channel_load(phi, op), g, op)
        assert float(u.mean()) > prev
        prev = float(u.mean())


if __name__ == "__main__":
    if pytest is not None:
        raise SystemExit(pytest.main([__file__, "-q"]))
    failed = 0
    for _name, _fn in sorted(dict(globals()).items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn()
            print(f"[PASS] {_name}")
        except AssertionError as _exc:
            failed += 1
            print(f"[FAIL] {_name}\n       {_exc}")
    print(f"\n{'FAILED' if failed else 'OK'}: {failed} failure(s)")
    raise SystemExit(1 if failed else 0)
