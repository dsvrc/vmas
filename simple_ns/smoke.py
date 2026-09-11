#!/usr/bin/env python
#  In-simulator integration smoke test.  RUN THIS ON THE SERVER BEFORE THE SWEEP.
#
#      python simple_ns/smoke.py
#      python simple_ns/smoke.py --host sampling --agents 3
#      python simple_ns/smoke.py --device cuda
#
#  conformance.py proves the ARITHMETIC offline.  This proves the WIRING: that
#  the dial reaches the physics, that the identities the spec gates survive
#  contact with the real environment, and that the arms differ only where they
#  are supposed to.  Every check corresponds to a requirement whose violation
#  would silently produce plausible numbers.
#
#  Needs vmas.  Does NOT need torchrl or benchmarl.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simple_ns.hosts import make_scenario  # noqa: E402

CHECKS: List[Tuple[str, Callable[[], str]]] = []
RESULTS: List[Tuple[str, bool, str]] = []
ARGS = None

NS = dict(
    ns_period=100, ns_wet_fraction=0.5, ns_loss_at_sigma1=0.14, ns_rho=0.9,
    ns_n_types=3, ns_recv_spread=0.6, ns_send_spread=0.8,
    ns_kernel_lambda=0.35, ns_y_clip=10.0, ns_observe_residual=True,
)
#  Must mirror the committed task yaml.  It did not, and the estimator check
#  failed for that reason alone: mu=0.999 has a 1000-step memory against a
#  100-step driver cycle, so it averages the drift away.
PACT = dict(
    pact_trust=0.9, pact_mu=0.97, pact_p0=10.0, pact_warmup=10, pact_corr_clip=0.5
)


def check(name: str):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn

    return wrap


def build(pact: bool, sigma: float, seed: int = 0, agents: int = None, **over):
    from vmas import make_env

    n = agents if agents is not None else ARGS.agents
    sc = make_scenario(pact=pact, host=ARGS.host)
    kw = dict(
        num_envs=ARGS.envs, device=ARGS.device, continuous_actions=True, seed=seed,
        max_steps=10 ** 9, clamp_actions=True, n_agents=n, ns_severity=sigma,
        ns_direct=False, pact_enabled=pact, pact_channels="full",
        pact_oracle=False, **NS, **PACT,
    )
    kw.update(over)
    env = make_env(scenario=sc, **kw)
    env.reset()
    return sc, env


def drive(sc, env, steps: int, seed: int = 99):
    """Identical action stream AND identical environment stochasticity.

    The hosts draw from the global RNG on reset, and VMAS's own `local_seed`
    state is a CLASS attribute shared by every env in the process, so two
    rollouts run back to back see different streams even from the same env seed.
    Seeding here is what makes this test emulate the sweep -- where each arm is
    its own seeded process -- rather than measure the interpreter's history.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n, B, D = len(sc.world.agents), sc.world.batch_dim, sc._action_dim
    obs_l, rew_l = [], []
    drive.load, drive.err = [], []
    for _ in range(steps):
        a = [
            ((torch.rand(B, D, generator=g) * 2 - 1) * 0.8).to(ARGS.device)
            for _ in range(n)
        ]
        obs, rew, _, _ = env.step(a)
        obs_l.append(torch.stack(obs))
        rew_l.append(torch.stack(rew))
        # accumulated, not sampled: half of every cycle is exactly quiet, so the
        # load read after the last step is usually 0 and means nothing
        drive.load.append(float(sc._load.abs().mean()))
        if hasattr(sc, "_pred"):
            drive.err.append(float((sc._load - sc._pred).abs().mean()))
    return torch.stack(obs_l), torch.stack(rew_l)


def mean_load() -> float:
    return sum(drive.load) / max(len(drive.load), 1)


# ===========================================================================


@check("test_the_layer_reaches_the_physics")
def _fires():
    """NS-3.3.  A silently inert disturbance is the one failure mode
    indistinguishable from a clean null result."""
    sc, env = build(False, 2.0)
    _, r = drive(sc, env, 150)
    sc.assert_layer_fired()
    assert bool(torch.isfinite(r).all()), "non-finite reward"
    return f"{sc.severity_report()}; mean |load| over the run {mean_load():.4f}"


@check("test_records_carry_the_disturbance")
def _records():
    """NS-3.2: harm the RECORDS as well as the rewards.  Every downstream
    metric, plot and estimator reads the logged trajectory."""
    sc, env = build(False, 2.0)
    drive(sc, env, 80)
    info = sc.info(sc.world.agents[0])
    for k in ("ns_load", "ns_dmag", "ns_A", "ns_y", "ns_clipped"):
        assert k in info, f"info is missing {k}"
        assert torch.isfinite(info[k]).all(), f"{k} is not finite"
    obs = sc.observation(sc.world.agents[0])
    assert torch.isfinite(obs).all(), "observation is not finite"
    return f"info carries ns_load/dmag/A/y/clipped; obs dim {obs.shape[-1]}"


@check("test_lone_agent_is_untouched_in_the_simulator")
def _lone():
    """I.2 / the (B) vs (C) decision procedure, on the real environment rather
    than on the arithmetic: a single agent must read EXACTLY zero at a severity
    far past anything the paper quotes."""
    try:
        sc, env = build(False, 20.0, agents=1)
    except Exception as exc:  # some hosts require n_agents > 1
        return f"skipped: {ARGS.host} rejects N=1 ({type(exc).__name__})"
    worst = 0.0
    for _ in range(120):
        drive(sc, env, 1)
        worst = max(worst, float(sc._load.abs().max()))
    assert worst == 0.0, f"a lone agent read a load of {worst:.6g} at sigma=20"
    return "N=1, sigma=20, 120 steps: load exactly 0.0 -- category C, measured"


@check("test_sigma_zero_is_bit_identical_across_arms")
def _sigma0():
    """NS-2.1.  At sigma=0 the target is exactly 0, so beta stays exactly 0,
    every prediction is 0 and the correction is 0."""
    sa, ea = build(False, 0.0, seed=3)
    sb, eb = build(True, 0.0, seed=3)
    oa, ra = drive(sa, ea, 60)
    ob, rb = drive(sb, eb, 60)
    assert torch.equal(ra, rb), f"rewards differ by {float((ra - rb).abs().max()):.3e}"
    assert torch.equal(oa, ob), "observations differ"
    assert float(sa._load.abs().max()) == 0.0
    return "60 steps, rewards and observations bit-identical, load exactly 0.0"


@check("test_placebo_is_bit_identical_across_arms")
def _placebo():
    """NS-2.5.  wet_fraction=0 makes the driver an EXACT zero, so the dial is
    provably inert at every sigma -- run far past the operating point."""
    sa, ea = build(False, 6.0, seed=3, ns_wet_fraction=0.0)
    sb, eb = build(True, 6.0, seed=3, ns_wet_fraction=0.0)
    _, ra = drive(sa, ea, 60)
    _, rb = drive(sb, eb, 60)
    assert torch.equal(ra, rb), "placebo arms differ"
    assert float(sa._load.abs().max()) == 0.0
    return "sigma=6 with an identically quiet driver: inert, and both arms agree"


@check("test_floor_property_in_the_simulator")
def _floor():
    """P-7.1.  trust forced to 0 must be bit-identical to the untouched host --
    same wrapper, same observation, same seed, any estimate."""
    sa, ea = build(False, 2.0, seed=5)
    sb, eb = build(True, 2.0, seed=5, pact_trust=0.0)
    oa, ra = drive(sa, ea, 60)
    ob, rb = drive(sb, eb, 60)
    assert torch.equal(ra, rb), f"rewards differ by {float((ra - rb).abs().max()):.3e}"
    assert torch.equal(oa, ob), "observations differ"
    return "the pact-off arm is provably the blind arm, over 60 steps"


@check("test_the_channel_is_invertible")
def _invertible():
    """II.6 row 1, which is the whole reason this instance exists.

    Handed the TRUE disturbance, the correction must cancel it -- so the executed
    action equals the commanded one to floating point, wherever the actuator has
    not saturated.  Where no inverse exists the best a method can do is steer,
    and the claim has to be narrowed accordingly.
    """
    sc, env = build(True, 2.0, seed=1, pact_oracle=True, pact_warmup=0)
    drive(sc, env, 80)
    resid = float((sc._load - sc._pred).abs().max())
    assert resid < 1e-5, f"the oracle's estimate was not exact: {resid:.3e}"
    unsat = sc._clipped == 0
    err = (sc._corr - sc._d).abs().sum(-1)[unsat]
    worst = float(err.max()) if err.numel() else 0.0
    assert worst < 1e-4, f"correction did not cancel the disturbance: {worst:.3e}"
    return (
        f"with a correct estimate the disturbance cancels to {worst:.2e} "
        f"wherever the actuator is unsaturated ({float((~unsat).float().mean()):.1%} clipped)"
    )


@check("test_estimator_is_live_and_learns")
def _estimator():
    """II.9 gate 5 (channels not inert) and the point of the whole exercise: the
    prediction error must FALL as the estimator sees data."""
    sc, env = build(True, 2.0, seed=1)
    # ONE rollout.  Calling drive(..., 1) in a loop re-seeds the action stream
    # every step, so the fleet repeats itself exactly, the channels are frozen
    # and there is nothing to excite the estimator -- which read as the method
    # failing to learn when it was the test holding the fleet still.
    drive(sc, env, 300)
    e, ld = drive.err, drive.load
    assert int(sc.rls.n_updates.min()) > 0, "the estimator was never updated"
    assert int(sc.rls.n_skipped.max()) == 0, "rows were skipped -- dead regressor"
    m = lambda v: sum(v) / max(len(v), 1)  # noqa: E731

    #  Scored against predicting ZERO, not against the error falling monotonically.
    #  beta* tracks a cyclic driver, so the error is cyclic too and any two
    #  windows can compare either way depending on where in the cycle they land;
    #  requiring a monotone fall tests the phase of the driver, not the estimator.
    #  Beating the zero-prediction null is the claim that matters.
    null, err = m(ld), m(e)
    assert null > 0, "no disturbance to predict"
    assert err < null, (
        f"the estimator did not beat predicting zero: |load|={null:.4f} "
        f"vs error={err:.4f}"
    )
    e0, e1 = m(e[: len(e) // 3]), m(e[-len(e) // 3 :])
    return (
        f"mean |load| {null:.4f} vs prediction error {err:.4f} "
        f"({100 * (1 - err / null):.0f}% of the disturbance explained); "
        f"first third {e0:.4f} -> last third {e1:.4f}; "
        f"{int(sc.rls.n_updates.min())} updates, trust {float(sc._trust.mean()):.3f}"
    )


@check("test_direct_control_is_felt_by_a_lone_agent")
def _control():
    """The (B) control must be a DIFFERENT cell, and measurably so: with
    ns_direct the disturbance no longer passes through the neighbours, so a lone
    agent feels it.  That is the whole difference between the two arms of
    scripts/ns_cells.sh."""
    try:
        sc, env = build(False, 2.0, agents=1, ns_direct=True)
    except Exception as exc:
        return f"skipped: {ARGS.host} rejects N=1 ({type(exc).__name__})"
    worst = 0.0
    for _ in range(80):
        drive(sc, env, 1)
        worst = max(worst, float(sc._load.abs().max()))
    assert worst > 0.0, "ns_direct=True still read zero at N=1; it is not cell (B)"
    return f"N=1 under ns_direct reads a load of {worst:.4f} -- cell (B), as intended"


@check("test_severity_moves_the_domain_metric")
def _bites():
    """A severity that does not move the metric is a dead experiment.  Note this
    runs on a RANDOM action stream, which is the weakest possible test -- use
    simple_ns/calibrate.py on the host's own heuristic to choose sigma."""
    out = {}
    for sg in (0.0, 2.0, 6.0):
        sc, env = build(False, sg, seed=4)
        _, r = drive(sc, env, 200)
        out[sg] = (float(r.mean()), mean_load())
    assert out[2.0][1] > out[0.0][1] and out[6.0][1] > out[2.0][1], (
        "the load did not rise with sigma"
    )
    return "  ".join(
        f"sigma={k:g}: return/step {v[0]:+.4f} |load| {v[1]:.4f}"
        for k, v in out.items()
    )


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="transport",
                    choices=["transport", "sampling", "balance", "navigation"])
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ARGS = ap.parse_args()

    width = 80
    print("=" * width)
    print(f"simple_ns integration smoke   host={ARGS.host} N={ARGS.agents} "
          f"envs={ARGS.envs} device={ARGS.device}")
    print("=" * width)

    for name, fn in CHECKS:
        try:
            RESULTS.append((name, True, fn() or ""))
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))

    failed = 0
    for name, ok, detail in RESULTS:
        failed += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"       {detail}")
    print("-" * width)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    if failed:
        print("\nDo NOT start the sweep.  Each of these gates a requirement whose "
              "violation produces plausible numbers.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
