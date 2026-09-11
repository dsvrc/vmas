#!/usr/bin/env python
#  In-simulator integration smoke test.  RUN THIS ON THE SERVER BEFORE THE SWEEP.
#
#      python road_ns/smoke.py                      # lanelet_flow, cpu
#      python road_ns/smoke.py --host road_traffic --agents 6 --envs 8
#      python road_ns/smoke.py --device cuda
#
#  conformance.py and pact1/selftest.py prove the ARITHMETIC offline.  This
#  proves the WIRING: that the dial reaches the physics, that the identities the
#  spec gates survive contact with the real environment, and that the arms
#  differ only where they are supposed to.  Every check here corresponds to a
#  requirement whose violation would silently produce plausible numbers.
#
#  Needs vmas.  Does NOT need torchrl or benchmarl.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_ns.scenario import make_scenario  # noqa: E402

CHECKS: List[Tuple[str, Callable[[], str]]] = []
RESULTS: List[Tuple[str, bool, str]] = []
ARGS = None


def check(name: str):
    """Register, do not run.  These need ARGS, which main() parses."""

    def wrap(fn: Callable[[], str]):
        CHECKS.append((name, fn))
        return fn

    return wrap


def run_checks() -> None:
    for name, fn in CHECKS:
        try:
            RESULTS.append((name, True, fn() or ""))
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))


def build(pact: bool, sigma: float, seed: int = 0, agents: int = None, **over):
    from vmas import make_env

    a = agents if agents is not None else ARGS.agents
    sc = make_scenario(pact=pact, host=ARGS.host)
    kw = dict(
        num_envs=ARGS.envs,
        device=ARGS.device,
        continuous_actions=True,
        seed=seed,
        max_steps=10**9,
        clamp_actions=True,
        n_agents=a,
        n_nearing_agents_observed=min(2, max(0, a - 1)),
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
        pact_warmup=10,
        pact_shift_mode="centred",
        pact_shift_clip=0.5,
    )
    if ARGS.host == "road_traffic":
        kw.update(map_type="1", is_partial_observation=True,
                  is_observe_vertices=False, is_add_noise=False)
    else:
        kw.update(
            flow_n_samples=512, flow_n_lookahead=3, flow_lookahead_stride=6,
            flow_search_window=12, flow_respawn=True, flow_reroute_on_lap=True,
            flow_n_background=ARGS.background, flow_background_speed=0.6,
            flow_integration="rk4",
        )
    kw.update(over)
    env = make_env(scenario=sc, **kw)
    env.reset()
    return sc, env


def drive(sc, env, steps: int, seed: int = 99):
    """Identical action stream AND identical environment stochasticity.

    The host's respawn draws from the global RNG, so two rollouts run
    back-to-back in ONE process see different streams even from the same env
    seed -- the first rollout advances it.  In the sweep each arm is its own
    process seeded at construction, so they do match; seeding here is what makes
    this test emulate that rather than measure the interpreter's history.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = len(sc.world.agents)
    B = sc.world.batch_dim
    out = []
    for _ in range(steps):
        a = [
            torch.stack(
                [
                    torch.rand(B, generator=g) * sc_max_speed(sc),
                    (torch.rand(B, generator=g) * 2 - 1) * 0.3,
                ],
                -1,
            ).to(ARGS.device)
            for _ in range(N)
        ]
        obs, rew, done, info = env.step(a)
        out.append((torch.stack(obs), torch.stack(rew)))
    return torch.stack([o for o, _ in out]), torch.stack([r for _, r in out])


def sc_max_speed(sc) -> float:
    return float(getattr(sc, "max_speed", 1.0))


# ===========================================================================


@check("test_the_layer_reaches_the_physics")
def _fires():
    """NS-3.3.  A silently inert disturbance is the one failure mode
    indistinguishable from a clean null result."""
    sc, env = build(False, 1.0)
    _, r = drive(sc, env, 120)
    sc.assert_layer_fired()
    assert bool(torch.isfinite(r).all()), "non-finite reward"
    harmed = int(sc._n_harmed)
    seen = int(sc._n_seen)
    assert harmed > 0
    return f"{harmed}/{seen} agent-steps harmed; {sc.severity_report()}"


@check("test_records_carry_the_harm")
def _records():
    """NS-3.2: harm the RECORDS as well as the rewards.  Every downstream
    metric, plot and estimator reads the logged trajectory."""
    sc, env = build(False, 1.0)
    drive(sc, env, 80)
    info = sc.info(sc.world.agents[0])
    for k in ("ns_u", "ns_harm", "ns_g", "ns_A", "ns_excess"):
        assert k in info, f"info is missing {k}"
        assert torch.isfinite(info[k]).all(), f"{k} is not finite"
    obs = sc.observation(sc.world.agents[0])
    assert torch.isfinite(obs).all(), "observation is not finite"
    return f"info carries ns_u/harm/g/A/excess; obs dim {obs.shape[-1]} (sensor appended)"


@check("test_sigma_zero_is_bit_identical_across_arms")
def _sigma0():
    """NS-2.1 in the simulator.  At sigma=0 the target is exactly 0, so beta
    stays exactly 0, every prediction is identical and the shift is exactly
    1.0.  If the arms differ here the compensator is acting on a disturbance
    that does not exist."""
    sa, ea = build(False, 0.0, seed=3)
    sb, eb = build(True, 0.0, seed=3)
    oa, ra = drive(sa, ea, 60)
    ob, rb = drive(sb, eb, 60)
    assert torch.equal(ra, rb), f"rewards differ by {float((ra - rb).abs().max()):.3e}"
    assert torch.equal(oa, ob), "observations differ"
    assert float(sa._harm.max()) == 1.0, f"harm {float(sa._harm.max())} at sigma=0"
    return "60 steps, rewards and observations bit-identical, harm exactly 1.0"


@check("test_placebo_is_bit_identical_across_arms")
def _placebo():
    """NS-2.5.  wet_fraction=0 makes the driver an EXACT zero, so the dial is
    provably inert at every sigma -- run at 3x the physical anchor."""
    sa, ea = build(False, 3.0, seed=3, ns_wet_fraction=0.0)
    sb, eb = build(True, 3.0, seed=3, ns_wet_fraction=0.0)
    _, ra = drive(sa, ea, 60)
    _, rb = drive(sb, eb, 60)
    assert torch.equal(ra, rb), "placebo arms differ"
    assert float(sa._harm.max()) == 1.0
    return "sigma=3 with an identically dry driver: inert, and both arms agree"


@check("test_floor_property_in_the_simulator")
def _floor():
    """P-7.1.  trust forced to 0 must be bit-identical to the untouched host --
    same wrapper, same observation, same seed."""
    sa, ea = build(False, 1.0, seed=5)
    sb, eb = build(True, 1.0, seed=5, pact_trust=0.0)
    oa, ra = drive(sa, ea, 60)
    ob, rb = drive(sb, eb, 60)
    assert torch.equal(ra, rb), f"rewards differ by {float((ra - rb).abs().max()):.3e}"
    assert torch.equal(oa, ob), "observations differ"
    return "the pact-off arm is provably the blind arm, over 60 steps"


@check("test_lone_agent_is_untouched")
def _lone():
    """I.2's practical test, in the simulator: a single agent must read harm
    exactly 1.0 in the worst storm the dial reaches."""
    if ARGS.host != "lanelet_flow":
        return "skipped: road_traffic's observation builder requires N > 1"
    sc, env = build(False, 3.0, agents=1, flow_n_background=0)
    worst = 1.0
    for _ in range(120):  # more than a full wet half-cycle
        drive(sc, env, 1)
        worst = max(worst, float(sc._harm.max()))
    assert worst == 1.0, f"a lone agent read harm {worst:.6f} at sigma=3"
    return "N=1, sigma=3, 120 steps: harm exactly 1.0 -- the driver never adds to the loss"


@check("test_background_is_irreducible_not_invisible")
def _background():
    """I.5's Delta_fixed must actually raise the loading, or the controllable-
    share sweep has nothing to vary."""
    if ARGS.host != "lanelet_flow":
        return "skipped: road_traffic has no background demand"
    lo, env_lo = build(False, 1.0, seed=2, flow_n_background=0)
    drive(lo, env_lo, 80)
    hi, env_hi = build(False, 1.0, seed=2, flow_n_background=max(8, ARGS.background))
    drive(hi, env_hi, 80)
    u_lo, u_hi = float(lo._u.mean()), float(hi._u.mean())
    assert u_hi > u_lo, f"background did not raise the loading: {u_lo:.3f} -> {u_hi:.3f}"
    return f"u {u_lo:.3f} (no background) -> {u_hi:.3f} with background demand"


@check("test_shift_is_differential_and_bounded")
def _shift():
    """II.6 / P-8.1: a uniform shift accomplishes nothing, so the fleet mean
    must stay at 1.  And no estimate may reverse a vehicle."""
    sc, env = build(True, 1.0, seed=1)
    drive(sc, env, 150)
    s = sc._shift
    lo, hi = float(s.min()), float(s.max())
    mean = float(s.mean())
    assert lo > 0.0, f"the shift reversed a vehicle: min {lo}"
    assert lo >= 1.0 - sc.pact_params.shift_clip - 1e-6
    assert hi <= 1.0 + sc.pact_params.shift_clip + 1e-6
    assert abs(mean - 1.0) < 0.05, f"fleet mean shift {mean:.4f} -- not differential"
    return (
        f"shift in [{lo:.3f}, {hi:.3f}], fleet mean {mean:.4f}, "
        f"never reversed, |1-shift| mean {float((s - 1).abs().mean()):.4f}"
    )


@check("test_estimator_is_live")
def _estimator():
    """II.9 gate 5: the channels must not be inert.  With the fleet layout
    frozen there is nothing to identify, which is what route turnover fixes."""
    sc, env = build(True, 1.0, seed=1)
    drive(sc, env, 150)
    psi = sc.basis.design(sc._ns_route, sc._ref, sc._scale)
    x_std = float(psi[..., 1:].std())
    assert int(sc.rls.n_updates.min()) > 0, "the estimator was never updated"
    assert int(sc.rls.n_skipped.max()) == 0, "rows were skipped -- dead regressor"
    assert x_std > 1e-6, f"channels are inert (x_std={x_std:.2e}); gate 5 would abort"
    assert float(sc._trust.mean()) > 0.0, "trust never armed"
    return (
        f"x_std {x_std:.4f}, {int(sc.rls.n_updates.min())} updates, "
        f"trust applied {float(sc._trust.mean()):.3f}, conf {float(sc._conf.mean()):.3f}"
    )


@check("test_severity_moves_the_domain_metric")
def _bites():
    """Q7 / I.6.  A medium far below its limit cannot express a capacity loss
    however large sigma grows -- and a severity that moves the metric by less
    than seed noise is a dead experiment."""
    out = {}
    for sg in (0.0, 1.0, 3.0):
        sc, env = build(False, sg, seed=4)
        _, r = drive(sc, env, 200)
        out[sg] = (float(r.mean()), float(sc._u.mean()))
    b = out[0.0][0]
    assert out[1.0][0] != b or out[3.0][0] != b, "sigma changed nothing at all"
    return "  ".join(
        f"sigma={k:g}: return/step {v[0]:+.4f} u {v[1]:.3f}" for k, v in out.items()
    )


# ===========================================================================


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="lanelet_flow",
                    choices=["lanelet_flow", "road_traffic"])
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--background", type=int, default=18)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ARGS = ap.parse_args()

    width = 80
    print("=" * width)
    print(f"road_ns integration smoke test   host={ARGS.host} N={ARGS.agents} "
          f"bg={ARGS.background} envs={ARGS.envs} device={ARGS.device}")
    print("=" * width)

    run_checks()
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
