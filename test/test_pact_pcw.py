#  Unit tests for the PCW non-stationarity and the PACT arithmetic.
#
#  These run on a plain torch install -- no torchrl, no vmas, no simulator --
#  which is the point of keeping `pcw_core` dependency-free.  The reference
#  implementations below are written independently, in plain Python loops, so
#  the tests check the *semantics* rather than re-executing the same code.
#
#  Run:  pytest test/test_pact_pcw.py      or      python test/test_pact_pcw.py

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


def _load(name: str, relative: str):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


core = _load("pcw_core", "benchmarl/environments/vmas_ns/pcw_core.py")

RHO = 0.8
GAIN = 20.0
SEVERITY = 0.8


# =============================================================================
# Independent reference implementations (plain Python, no vectorisation)
# =============================================================================


def ref_message(pos, u):
    """(p x u)_z for one agent."""
    return pos[0] * u[1] - pos[1] * u[0]


def ref_phi(messages, i):
    """Mean over the OTHER agents.  Empty sum -> 0."""
    others = [m for j, m in enumerate(messages) if j != i]
    if not others:
        return 0.0
    return sum(others) / len(others)


def ref_rotate(v, theta):
    return (
        math.cos(theta) * v[0] - math.sin(theta) * v[1],
        math.sin(theta) * v[0] + math.cos(theta) * v[1],
    )


def ref_driver(step, index, batch, period, spread, freeze=None):
    if freeze is not None:
        return freeze
    offset = (index / batch) if (spread and batch > 1) else 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * (step / period + offset)))


def ref_env_and_pact(
    positions,
    actions,
    *,
    n_agents,
    severity,
    gain=GAIN,
    rho=RHO,
    betas=None,
    driver=1.0,
    resets=(),
):
    """The full loop, written the way the two implementations must interleave.

    Mirrors, step by step:
      * PACT (env-side wrapper): compensate with the *cached* x2, then advance
        the cache from the executed command and the position it was issued from;
      * the scenario: apply theta = c * x2 to the executed command, then advance
        its own accumulator from the same two quantities.

    Returns per-step records so a test can assert on any of them.
    """
    env_x2 = [0.0] * n_agents
    pact_x2 = [0.0] * n_agents
    records = []
    c = driver * severity

    for t, (pos_t, act_t) in enumerate(zip(positions, actions)):
        if t in resets:
            env_x2 = [0.0] * n_agents
            pact_x2 = [0.0] * n_agents

        # --- agent side: compensate using the waveform cached at t-1 ---------
        executed = []
        for i in range(n_agents):
            beta = 0.0 if betas is None else betas[i]
            executed.append(ref_rotate(act_t[i], -beta * pact_x2[i]))

        # --- env side: harm for this step, from ITS accumulator at t ---------
        thetas = [c * env_x2[i] for i in range(n_agents)]
        delivered = [ref_rotate(executed[i], thetas[i]) for i in range(n_agents)]

        records.append(
            {
                "executed": list(executed),
                "delivered": list(delivered),
                "theta": list(thetas),
                "env_x2": list(env_x2),
                "pact_x2": list(pact_x2),
            }
        )

        # --- both sides advance from the same (pos, executed command) --------
        messages = [ref_message(pos_t[i], executed[i]) for i in range(n_agents)]
        env_x2 = [
            rho * env_x2[i] + (1 - rho) * gain * ref_phi(messages, i)
            for i in range(n_agents)
        ]
        pact_x2 = [
            rho * pact_x2[i] + (1 - rho) * gain * ref_phi(messages, i)
            for i in range(n_agents)
        ]
    return records


def random_traj(n_steps, n_agents, seed=0):
    generator = torch.Generator().manual_seed(seed)
    pos = (torch.rand(n_steps, n_agents, 2, generator=generator) * 2 - 1).tolist()
    act = (torch.rand(n_steps, n_agents, 2, generator=generator) * 2 - 1).tolist()
    return pos, act


# =============================================================================
# 1. The exertion functional and the leak
# =============================================================================


def test_angular_impulse_matches_reference():
    pos, act = random_traj(5, 4, seed=1)
    got = core.angular_impulse(torch.tensor(pos), torch.tensor(act))
    for t in range(5):
        for i in range(4):
            assert abs(float(got[t, i]) - ref_message(pos[t][i], act[t][i])) < 1e-6


def test_peer_mean_excludes_self():
    m = torch.tensor([[1.0, 2.0, 6.0]])
    phi = core.peer_mean(m)
    assert abs(float(phi[0, 0]) - 4.0) < 1e-6  # (2+6)/2
    assert abs(float(phi[0, 1]) - 3.5) < 1e-6  # (1+6)/2
    assert abs(float(phi[0, 2]) - 1.5) < 1e-6  # (1+2)/2


def test_peer_mean_is_identically_zero_at_n1():
    """The irreducibility certificate: at N=1 the effect vanishes, not shrinks."""
    for value in (0.0, 1e-3, 1.0, 1e6):
        m = torch.full((7, 1), value)
        phi = core.peer_mean(m)
        assert torch.equal(phi, torch.zeros_like(phi))


def test_n1_reduces_the_environment_exactly():
    """With one agent the delivered command equals the command, bit for bit,
    at any severity -- including severities far past where N>=2 collapses."""
    pos, act = random_traj(40, 1, seed=2)
    for severity in (0.0, 0.8, 50.0):
        records = ref_env_and_pact(pos, act, n_agents=1, severity=severity)
        for t, record in enumerate(records):
            assert record["theta"] == [0.0]
            got = core.rotate(
                torch.tensor(record["executed"][0]), torch.tensor(0.0)
            )
            assert torch.equal(got, torch.tensor(act[t][0]))


def test_leak_step_matches_reference():
    x2 = torch.tensor([0.3, -1.2])
    phi = torch.tensor([0.5, 0.25])
    got = core.leak_step(x2, phi, rho=RHO, gain=GAIN)
    for i in range(2):
        want = RHO * float(x2[i]) + (1 - RHO) * GAIN * float(phi[i])
        assert abs(float(got[i]) - want) < 1e-6


# =============================================================================
# 2. The harm channel and its inverse
# =============================================================================


def test_rotate_by_zero_is_bit_identical():
    """severity=0 must be an *exact* reduction, not an approximate one."""
    v = torch.randn(64, 3, 2)
    assert torch.equal(core.rotate(v, torch.zeros(64, 3)), v)


def test_channel_inverse_cancels_exactly():
    v = torch.randn(32, 4, 2)
    theta = torch.randn(32, 4) * 3.0
    delivered = core.rotate(core.channel_inverse(v, theta), theta)
    assert torch.allclose(delivered, v, atol=1e-5)


def test_beta_zero_is_the_blind_policy():
    """The floor property: with beta = 0 the executed action IS the raw action."""
    a = torch.randn(16, 3, 2)
    x2 = torch.randn(16, 3) * 5.0
    beta = torch.zeros(16, 3)
    assert torch.equal(core.channel_inverse(a, beta * x2), a)


def test_beta_is_per_agent():
    """Agent i's gain must not touch agent j's command."""
    a = torch.randn(8, 3, 2)
    x2 = torch.ones(8, 3)
    beta = torch.tensor([0.0, 0.7, 0.0]).expand(8, 3)
    out = core.channel_inverse(a, beta * x2)
    assert torch.equal(out[:, 0], a[:, 0])
    assert torch.equal(out[:, 2], a[:, 2])
    assert not torch.allclose(out[:, 1], a[:, 1])


def test_beta_from_w_range_and_init():
    beta_max = 1.04
    w = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    beta = core.beta_from_w(w, beta_max, "affine")
    assert float(beta.min()) >= 0.0
    assert float(beta.max()) <= beta_max + 1e-6
    assert abs(float(beta[2]) - beta_max / 2) < 1e-6  # direct-mode init
    assert abs(core.beta_init_value(beta_max, "affine") - beta_max / 2) < 1e-6
    sig = core.beta_from_w(w, beta_max, "sigmoid")
    assert float(sig.min()) >= 0.0 and float(sig.max()) <= beta_max


# =============================================================================
# 3. The exogenous driver
# =============================================================================


def test_driver_range_and_reference():
    for step in (0, 37, 999, 2000):
        got = core.driver_A(step, 8, period=2000, phase_spread=True)
        assert float(got.min()) >= -1e-7 and float(got.max()) <= 1 + 1e-7
        for b in range(8):
            want = ref_driver(step, b, 8, 2000, True)
            assert abs(float(got[b]) - want) < 1e-6


def test_driver_is_periodic_and_slow():
    period = 2000
    a0 = core.driver_A(0, 4, period=period, phase_spread=True)
    a1 = core.driver_A(period, 4, period=period, phase_spread=True)
    assert torch.allclose(a0, a1, atol=1e-5)
    # within one 60-step episode the driver must barely move, so that `c` is
    # approximately constant inside an episode
    worst = max(
        float((core.driver_A(t + 60, 1, period=period, phase_spread=False)
               - core.driver_A(t, 1, period=period, phase_spread=False)).abs())
        for t in range(0, period, 50)
    )
    assert worst < 0.11, worst


def test_phase_spread_covers_the_cycle():
    batch = 16
    values = core.driver_A(0, batch, period=2000, phase_spread=True)
    assert float(values.min()) < 0.05
    assert float(values.max()) > 0.95
    flat = core.driver_A(0, batch, period=2000, phase_spread=False)
    assert float(flat.std()) == 0.0


def test_freeze_knob_overrides_everything():
    for step in (0, 123, 4567):
        got = core.driver_A(step, 5, period=2000, phase_spread=True, freeze=0.75)
        assert torch.allclose(got, torch.full((5,), 0.75))


def test_driver_is_agent_independent_and_exogenous():
    """The driver depends only on the clock: same value regardless of anything
    the agents did.  This is what survives freezing the teammates."""
    a = core.driver_A(500, 4, period=2000, phase_spread=True)
    b = core.driver_A(500, 4, period=2000, phase_spread=True)
    assert torch.equal(a, b)


# =============================================================================
# 4. The one-step timing contract and the gate
# =============================================================================


def test_pact_waveform_equals_env_accumulator_every_step():
    """The Phase-2 gate, at the semantic level.

    Two independently-advanced accumulators -- the environment's and the one
    PACT rebuilds from shared messages -- must agree at every step, including
    while the compensation is actively changing the executed commands (the
    category-C loop gain).
    """
    pos, act = random_traj(120, 3, seed=3)
    records = ref_env_and_pact(
        pos, act, n_agents=3, severity=SEVERITY, betas=[0.4, 0.8, 0.1]
    )
    for t, record in enumerate(records):
        env_v = torch.tensor(record["env_x2"])
        pact_v = torch.tensor(record["pact_x2"])
        assert torch.allclose(env_v, pact_v, atol=1e-9), t
        if env_v.norm() > 1e-8:
            cos, valid = core.per_step_cosine(pact_v, env_v)
            assert bool(valid) and float(cos) > 0.999999


def test_waveform_uses_the_previous_step_only():
    """x2 used at step t must not depend on anything from step t.

    Perturbing the action at step t may change x2 from t+1 onwards, never at t.
    """
    pos, act = random_traj(20, 3, seed=4)
    base = ref_env_and_pact(pos, act, n_agents=3, severity=SEVERITY)
    perturbed_act = [list(map(list, row)) for row in act]
    perturbed_act[10][0] = [0.9, -0.9]
    other = ref_env_and_pact(perturbed_act and pos, perturbed_act, n_agents=3,
                             severity=SEVERITY)
    for t in range(11):
        assert base[t]["env_x2"] == other[t]["env_x2"], t
    assert base[11]["env_x2"] != other[11]["env_x2"]


def test_reset_clears_the_accumulator():
    """Probe/wrapper state must die exactly when the episode does."""
    pos, act = random_traj(30, 3, seed=5)
    records = ref_env_and_pact(pos, act, n_agents=3, severity=SEVERITY, resets={15})
    assert records[15]["env_x2"] == [0.0, 0.0, 0.0]
    assert records[15]["theta"] == [0.0, 0.0, 0.0]
    assert records[14]["env_x2"] != [0.0, 0.0, 0.0]


def test_per_step_cosine_beats_a_pooled_correlation():
    """Why the gate is a per-step cosine and not a pooled correlation.

    Construct data where every point satisfies theta = c * x2 exactly but c
    varies across the batch.  The per-step cosine sees 1.0; a correlation pooled
    over the varying-c fan does not, which is how a correct pipeline gets
    misdiagnosed as broken.
    """
    generator = torch.Generator().manual_seed(6)
    steps, n_agents = 400, 4
    x2 = torch.randn(steps, n_agents, generator=generator)
    c = torch.rand(steps, 1, generator=generator) * 0.8
    theta = c * x2  # exact by construction

    cos, valid = core.per_step_cosine(x2, theta)
    assert float(cos[valid].mean()) > 0.9999

    flat_x2 = x2.reshape(-1)
    flat_theta = theta.reshape(-1)
    pooled = torch.corrcoef(torch.stack((flat_x2, flat_theta)))[0, 1]
    assert float(pooled) < 0.999, float(pooled)


def test_per_step_cosine_flags_a_shuffled_agent_index():
    """The gate's job: catch index-order bugs."""
    generator = torch.Generator().manual_seed(7)
    x2 = torch.randn(200, 4, generator=generator)
    shuffled = x2[:, [1, 0, 3, 2]]
    cos, valid = core.per_step_cosine(shuffled, x2)
    assert float(cos[valid].mean()) < 0.999


def test_per_step_cosine_flags_a_one_step_offset():
    """The gate's job: catch timing bugs."""
    generator = torch.Generator().manual_seed(8)
    x2 = torch.randn(200, 4, generator=generator).cumsum(0) * 0.1
    cos, valid = core.per_step_cosine(x2[1:], x2[:-1])
    assert float(cos[valid].mean()) < 0.999


# =============================================================================
# 5. Compensation semantics end to end
# =============================================================================


def test_perfect_gain_delivers_the_raw_action():
    """Theorem T2: with beta = c the medium delivers the stationary command."""
    pos, act = random_traj(60, 3, seed=9)
    records = ref_env_and_pact(
        pos, act, n_agents=3, severity=SEVERITY, betas=[SEVERITY] * 3, driver=1.0
    )
    for t, record in enumerate(records):
        for i in range(3):
            got = torch.tensor(record["delivered"][i])
            want = torch.tensor(act[t][i])
            assert torch.allclose(got, want, atol=1e-6), (t, i)


def test_blind_is_actually_disturbed():
    """Sanity: the same trajectory with beta = 0 must NOT be delivered intact,
    otherwise the previous test proves nothing."""
    pos, act = random_traj(60, 3, seed=9)
    records = ref_env_and_pact(pos, act, n_agents=3, severity=SEVERITY)
    worst = max(
        float((torch.tensor(r["delivered"][i]) - torch.tensor(act[t][i])).abs().max())
        for t, r in enumerate(records)
        for i in range(3)
    )
    assert worst > 0.1, worst


def test_wrong_gain_residual_is_linear_in_the_error():
    """Theorem T3: the residual felt is (c - beta) * x2 -- linear in the error
    of the single scalar, which is what makes 1-D tracking sufficient."""
    pos, act = random_traj(50, 3, seed=10)
    for error in (0.1, 0.2, 0.4):
        records = ref_env_and_pact(
            pos,
            act,
            n_agents=3,
            severity=SEVERITY,
            betas=[SEVERITY - error] * 3,
        )
        for record in records[5:]:
            for i in range(3):
                residual = record["theta"][i] - (SEVERITY - error) * record["pact_x2"][i]
                assert abs(residual - error * record["env_x2"][i]) < 1e-9


# =============================================================================
# 6. Torchrl-level wiring (skipped when torchrl is unavailable)
# =============================================================================


def test_transform_specs_widen_action_and_observation():
    try:
        import torchrl  # noqa: F401
        from torchrl.data import Bounded, Composite, Unbounded
    except ImportError:
        print("  (skipped: torchrl not installed)")
        return

    pact = _load("pact_mod", "benchmarl/environments/vmas_ns/pact.py")
    params = core.PcwParams(severity=SEVERITY, gain=GAIN, rho=RHO)
    n_agents, obs_dim = 3, 18
    transform = pact.PactTransform(
        group="agents",
        n_agents=n_agents,
        params=params,
        beta_max=1.04,
        action_low=-torch.ones(n_agents, 2),
        action_high=torch.ones(n_agents, 2),
    )

    action_spec = Composite(
        {
            "agents": Composite(
                {
                    "action": Bounded(
                        low=-torch.ones(n_agents, 2),
                        high=torch.ones(n_agents, 2),
                        shape=torch.Size((n_agents, 2)),
                    )
                },
                shape=(n_agents,),
            )
        }
    )
    widened = transform.transform_action_spec(action_spec)
    assert widened[("agents", "action")].shape[-1] == 3

    obs_spec = Composite(
        {
            "agents": Composite(
                {"observation": Unbounded(shape=torch.Size((n_agents, obs_dim)))},
                shape=(n_agents,),
            )
        }
    )
    widened_obs = transform.transform_observation_spec(obs_spec)
    assert widened_obs[("agents", "observation")].shape[-1] == obs_dim + 3


# =============================================================================


def _main():
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
