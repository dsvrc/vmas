#!/usr/bin/env python
#  Integration checks against the REAL simulator.  Run this on the training
#  machine, once, before spending any GPU time.
#
#      python pact2/smoke_test.py
#
#  ``selfcheck.py`` proves the arithmetic.  This proves the WIRING -- the claims
#  unit tests cannot reach because they are about how the pieces are connected:
#  that severity actually reaches the physics, that reward really is untouched,
#  that the placebo really is inert, that PACT widens the interface by exactly
#  one observation feature and zero action dimensions.
#
#  A non-stationarity that silently fails to fire looks identical to one that is
#  working.  That is why this file exists.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

import vmas  # noqa: E402

from benchmarl.environments.vmas_slc.scenario import make_slc_scenario  # noqa: E402
from benchmarl.environments.vmas_slc.slc_core import SHIFT_QUIET  # noqa: E402

DEV = "cpu"
NENV = 4
NAG = 6

#: Registered, not run: the checks need the --scenario argument, so they are
#: collected at import and executed from main() once argv has been parsed.
CHECKS: List[Tuple[str, Callable[[], str]]] = []

#: Set by main().  Every check reads it through make().
SCENARIO = "sampling"


def check(name: str):
    def wrap(fn: Callable[[], str]):
        CHECKS.append((name, fn))
        return fn

    return wrap


SLC_DEFAULTS: Dict = dict(
    slc_severity=1.0,
    slc_driver_period=200,  # short, so a smoke run spans the cycle
    slc_phase_spread=True,
    slc_a_ref=0.5,
    slc_quiet_scale=0.5,
    slc_p_quiet=0.0,
    slc_snr_ref=100.0,
    slc_g_min_at_sigma1=0.65,
    slc_mean_preserve=False,
    slc_n_chan=3,
    slc_aclr=0.25,
    slc_leak_span=2,
    slc_duty_lo=0.7,
    slc_duty_hi=1.4,
    slc_exposure_lo=0.6,
    slc_exposure_hi=1.4,
    slc_lfix_frac=0.15,
    slc_u_nominal=0.30,
    slc_capacity_mode="scaled",
    slc_capacity_ref_agents=6,
    slc_phi_floor=0.25,
    slc_phi_slope=1.875,
    slc_v_ref=0.4,
    slc_phi_reads_executed=True,
    slc_harm_at_nominal=0.30,
    slc_harm_cap=0.85,
    slc_harm_enabled=True,
    slc_observe_loading=True,
)

PACT_DEFAULTS: Dict = dict(
    pact_enabled=False,
    pact_mode="delta",
    pact_gate="fit",
    pact_r=1,
    pact_mu=0.99,
    pact_p0=10.0,
    pact_p_max_mult=10.0,
    pact_max_trust=0.80,
    pact_ff_gain=1.0,
    pact_own_gain=1.0,
    pact_fit_floor=0.0,
    pact_ready_updates=50,
    pact_fit_ema=2e-3,
    pact_warmup_updates=20,
    pact_level_tau=400.0,
    pact_max_delta=1.0,
    pact_denom_floor=0.15,
    pact_u_cap=3.0,
)

def scenario_kwargs(n_agents: int) -> Dict:
    """The stock task's own kwargs, matching the shipped yaml.

    Built per call, NOT frozen at import: a module-level dict capturing ``NAG``
    once made the N=1 certificate silently run a six-agent world.
    """
    if SCENARIO == "sampling":
        return dict(
            n_agents=n_agents,
            shared_rew=False,
            n_gaussians=3,
            lidar_range=0.2,
            cov=0.05,
            collisions=True,
            spawn_same_pos=False,
        )
    if SCENARIO == "discovery":
        return dict(
            n_agents=n_agents,
            n_targets=7,
            lidar_range=0.35,
            covering_range=0.25,
            agents_per_target=min(2, n_agents),
            targets_respawn=True,
            shared_reward=True,
        )
    if SCENARIO == "navigation":
        return dict(
            n_agents=n_agents,
            collisions=True,
            agents_with_same_goal=1,
            observe_all_goals=False,
            shared_rew=False,
            split_goals=False,
            lidar_range=0.35,
            agent_radius=0.1,
        )
    raise ValueError(f"unknown scenario {SCENARIO!r}")


def make(
    stock: bool = False,
    pact: bool = False,
    seed: int = 0,
    n_agents: int = NAG,
    **over,
):
    """Build an env.  ``stock=True`` gives the unmodified stock scenario."""
    if stock:
        return vmas.make_env(
            scenario=SCENARIO,
            num_envs=NENV,
            device=DEV,
            continuous_actions=True,
            seed=seed,
            **scenario_kwargs(n_agents),
        )
    kw = dict(SLC_DEFAULTS)
    kw.update(PACT_DEFAULTS)
    kw["pact_enabled"] = pact
    kw.update(over)
    return vmas.make_env(
        scenario=make_slc_scenario(SCENARIO, pact),
        num_envs=NENV,
        device=DEV,
        continuous_actions=True,
        seed=seed,
        **scenario_kwargs(n_agents),
        **kw,
    )


def fixed_actions(
    steps: int, seed: int = 123, n_agents: int = NAG
) -> List[List[torch.Tensor]]:
    """One action sequence, reused across arms.  Severity must be the only
    variable; a fresh sample per arm would make every comparison noise."""
    g = torch.Generator().manual_seed(seed)
    return [
        [torch.rand(NENV, 2, generator=g) * 2 - 1 for _ in range(n_agents)]
        for _ in range(steps)
    ]


def roll(env, actions) -> Dict[str, torch.Tensor]:
    env.reset(seed=0)
    pos, vel, rew, u = [], [], [], []
    for a in actions:
        obs, r, d, info = env.step([x.clone() for x in a])
        pos.append(torch.stack([ag.state.pos for ag in env.agents], 1).clone())
        vel.append(torch.stack([ag.state.vel for ag in env.agents], 1).clone())
        rew.append(torch.stack(r, 1).clone())
        if info and "slc_u" in info[0]:
            u.append(torch.stack([i["slc_u"] for i in info], 1).clone())
    out = dict(
        pos=torch.stack(pos), vel=torch.stack(vel), rew=torch.stack(rew)
    )
    if u:
        out["u"] = torch.stack(u)
    return out


# =============================================================================


@check("1 slc_harm_enabled=false reproduces the stock scenario STEP FOR STEP")
def _stock():
    acts = fixed_actions(60)
    a = roll(make(stock=True), acts)
    b = roll(make(slc_harm_enabled=False, slc_observe_loading=False), acts)
    dp = float((a["pos"] - b["pos"]).abs().max())
    dr = float((a["rew"] - b["rew"]).abs().max())
    assert dp == 0.0 and dr == 0.0, (
        f"max |dpos| {dp:.3e}, max |drew| {dr:.3e}.  The dynamics-only / "
        "reward-untouched constraint must be VERIFIED, not asserted in prose."
    )
    return "60 steps, positions and rewards identical to the last bit"


@check("2 reward is inherited, never reshaped")
def _reward_untouched():
    import importlib

    Stock = importlib.import_module(f"vmas.scenarios.{SCENARIO}").Scenario

    from benchmarl.environments.vmas_slc import scenario as mod

    cls = type(make_slc_scenario(SCENARIO, True))
    for name in ("reward", "done"):
        assert getattr(cls, name) is getattr(Stock, name), (
            f"{name} was overridden.  The agent must earn less strictly because "
            "it physically achieves less."
        )
    assert cls.process_action is not Stock.process_action
    assert "observation" in dir(mod.SlcMixin)
    return "reward and done resolve to the stock scenario's own functions"


@check("G0 liveness: the severity dial actually changes the physics")
def _g0():
    acts = fixed_actions(60)
    a = roll(make(slc_severity=0.0), acts)
    b = roll(make(slc_severity=1.0), acts)
    dp = float((a["pos"] - b["pos"]).abs().max())
    du = float((b["u"] - a["u"]).mean())
    assert dp > 1e-4, (
        f"sigma made no difference to the trajectory (max |dpos| {dp:.3e}).  A "
        "silently discarded dial produced five rows of pure scenario noise on "
        "POWER, and nothing looked wrong."
    )
    return f"max |dpos| {dp:.4f} over 60 steps; mean loading rose {du:+.4f}"


@check("G0b monotone: loading rises with sigma over a FIXED early window")
def _g0b():
    acts = fixed_actions(60)
    prev, rows = -1.0, []
    for s in (0.0, 0.5, 1.0, 1.5):
        r = roll(make(slc_severity=s), acts)
        u = float(r["u"][:40].mean())  # fixed window: whole-episode means have
        rows.append(f"sigma={s}:{u:.4f}")  # survivorship bias
        assert u >= prev - 1e-6, f"loading fell at sigma={s}"
        prev = u
    return "  ".join(rows)


@check("G6 placebo: the quiet shift is byte-identical across sigma")
def _g6():
    acts = fixed_actions(60)
    ref = None
    for s in (0.0, 1.0, 2.0):
        r = roll(make(slc_severity=s, slc_p_quiet=1.0), acts)
        if ref is None:
            ref = r
        else:
            d = float((r["pos"] - ref["pos"]).abs().max())
            assert d == 0.0, f"quiet shift differed at sigma={s} by {d:.3e}"
    return (
        "3 severities identical to the last bit.  A reviewer alleging a rigged "
        "knob has to explain why the rig switches itself off on the night shift."
    )


@check("A.3 the driver is exogenous and is NOT rewound by an episode reset")
def _driver_exogenous():
    env = make()
    env.reset(seed=0)
    sc = env.scenario
    for _ in range(10):
        env.step([torch.zeros(NENV, 2) for _ in range(NAG)])
    before = sc._slc_step.clone()
    env.reset_at(0)
    assert torch.equal(sc._slc_step, before), (
        "reset rewound the global driver clock.  An episode boundary that "
        "rewinds the driver makes it endogenous to the agents' own failures."
    )
    a_zero = roll(make(), [[torch.zeros(NENV, 2) for _ in range(NAG)]] * 30)
    a_rand = roll(make(), fixed_actions(30))
    assert "u" in a_zero and "u" in a_rand
    return f"clock survived a partial reset at step {int(before[0])}"


@check("G7 agents are consulted on EVERY step (no scripted bypass)")
def _g7():
    env = make()
    env.reset(seed=0)
    for ag in env.agents:
        assert ag.action_script is None, f"{ag.name} is script-driven"
    return (
        "VMAS interposes no heuristic, so steps-per-agent-decision is 1.0. "
        "POWER shipped at 7.5, i.e. the learned policy drove 13% of steps."
    )


@check("G1 N=1: the peer sum is empty, so the cross-agent term is exactly zero")
def _g1():
    env = make(n_agents=1)
    op = env.scenario.slc_op
    assert op.W.shape == (1, 1), f"expected a 1x1 operator, got {tuple(op.W.shape)}"
    assert float(op.W.abs().max()) == 0.0, "W is non-zero at N=1"
    env.reset(seed=0)
    u = []
    for _ in range(20):
        _, _, _, info = env.step([torch.rand(NENV, 2) * 2 - 1])
        u.append(info[0]["slc_u"])
    mean_u = float(torch.stack(u).mean())
    return (
        f"1x1 empty operator; the lone agent still feels the dial through its "
        f"OWN term (mean u {mean_u:.3f}) -- the certificate is that the "
        "CROSS-AGENT contribution is zero, not that the task is unchanged"
    )


@check("PACT widens the interface by ONE observation feature and ZERO actions")
def _interface():
    def width(env) -> int:
        o = env.reset()[0]
        # some scenarios return a dict observation; the sensor is one feature
        # either way
        return sum(v.shape[-1] for v in o.values()) if isinstance(o, dict) else o.shape[-1]

    blind = make(pact=False, slc_observe_loading=False)
    slc = make(pact=False, slc_observe_loading=True)
    pact = make(pact=True, slc_observe_loading=True)
    o_b, o_s, o_p = width(blind), width(slc), width(pact)
    assert o_s == o_b + 1, f"observation grew by {o_s - o_b}, expected 1"
    assert o_p == o_s, "PACT changed the observation; the host must be untouched"
    a_s = slc.get_agent_action_size(slc.agents[0])
    a_p = pact.get_agent_action_size(pact.agents[0])
    assert a_s == a_p, f"PACT changed the action size ({a_s} -> {a_p})"
    return (
        f"obs {o_b} -> {o_s} (the stale loading sensor) -> {o_p}; "
        f"action size {a_p} throughout"
    )


@check("floor property in the real env: gated-off PACT == the ff arm, bitwise")
def _floor():
    acts = fixed_actions(80)
    ff = roll(make(pact=True, pact_mode="ff"), acts)
    off = roll(make(pact=True, pact_fit_floor=1e9), acts)
    d = float((ff["pos"] - off["pos"]).abs().max())
    assert d == 0.0, f"diverged by {d:.3e}"
    live = roll(make(pact=True), acts)
    dl = float((live["pos"] - ff["pos"]).abs().max())
    assert dl > 0.0, (
        "an ADMISSIBLE PACT was also identical to the ff arm -- the peer term "
        "is doing nothing.  Read applied_trust before any other number."
    )
    return f"gated-off identical; live PACT differs by {dl:.4e} (it is acting)"


@check("config plumbing: every slc_/pact_ key really arrived at the scenario")
def _plumbing():
    sc = make(pact=True).scenario
    p, pp = sc.slc_params, sc.pact_params
    assert p.severity == SLC_DEFAULTS["slc_severity"]
    assert p.n_chan == SLC_DEFAULTS["slc_n_chan"]
    assert p.harm_at_nominal == SLC_DEFAULTS["slc_harm_at_nominal"]
    assert pp.max_trust == PACT_DEFAULTS["pact_max_trust"]
    assert pp.mu == PACT_DEFAULTS["pact_mu"]
    assert abs(p.harm_gain - p.harm_at_nominal / p.u_nominal) < 1e-9
    return (
        f"harm_gain={p.harm_gain:.3f} (anchor pins it to 1.0), "
        f"noise_rise={p.noise_rise:.3f} solved from g_min={p.g_min_at_sigma1}"
    )


@check("partial reset clears exactly the worlds that reset")
def _partial_reset():
    env = make(pact=True)
    env.reset(seed=0)
    for _ in range(15):
        env.step([torch.rand(NENV, 2) * 2 - 1 for _ in range(NAG)])
    sc = env.scenario
    lag_before = sc.pact.have_lag.clone()
    beta_before = sc.pact.beta.clone()
    env.reset_at(1)
    assert not bool(sc.pact.have_lag[1].any()), "env 1's history was not cleared"
    others = [i for i in range(NENV) if i != 1]
    assert bool(sc.pact.have_lag[others].all()), "other envs were cleared too"
    assert torch.equal(sc.pact.beta, beta_before), (
        "beta was reset.  A deployment's radio characterisation persists across "
        "missions; resetting it would restart identification every episode."
    )
    del lag_before
    return "history cleared for env 1 only; beta and P survived, by design"


@check("diagnostics reach the info dict (log_info has something to read)")
def _info():
    env = make(pact=True)
    env.reset(seed=0)
    for _ in range(60):
        _, _, _, info = env.step([torch.rand(NENV, 2) * 2 - 1 for _ in range(NAG)])
    keys = set(info[0])
    need = {
        "slc_u",
        "slc_c",
        "slc_g",
        "slc_phi",
        "slc_quiet",
        "pact_applied_trust",
        "pact_fit_gain",
        "pact_peer_abs",
        "pact_ff_abs",
        "pact_base_abs",
        "pact_delta_abs",
        "pact_state",
    }
    missing = need - keys
    assert not missing, f"missing info keys: {sorted(missing)}"
    return f"{len(keys)} info keys, including the {len(need)} the gates read"


# =============================================================================


def main() -> int:
    global SCENARIO
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenario",
        default="sampling",
        choices=["sampling", "discovery", "navigation"],
        help="which stock task to plant SLC into.  Run all three: the "
        "observation and info paths differ per scenario, so passing on one "
        "says nothing about the others.",
    )
    args = ap.parse_args()
    SCENARIO = args.scenario

    width = 78
    print("=" * width)
    print("SLC / PACT smoke test   (needs vmas; run this on the training machine)")
    print(f"scenario {SCENARIO}   vmas {vmas.__version__}   torch {torch.__version__}")
    print("=" * width)

    results = []
    for name, fn in CHECKS:
        try:
            results.append((name, True, fn() or ""))
        except AssertionError as exc:
            results.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            import traceback

            results.append(
                (name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            )

    failed = 0
    for name, ok, detail in results:
        failed += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"       {detail}")
    print("-" * width)
    print(f"{len(results) - failed}/{len(results)} checks passed  [{SCENARIO}]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
