#  Integration smoke test -- run this ONCE on the machine that has vmas+torchrl,
#  before spending any training compute.
#
#      python pact/smoke_test.py
#
#  It checks the claims that the pure-torch unit tests cannot reach, because they
#  are about wiring rather than arithmetic:
#
#    1. severity 0 reproduces stock VMAS `navigation` step for step  (constraint 3:
#       dynamics-only, reward untouched)
#    2. n_agents=1 reproduces the stationary task at full severity   (constraint 2:
#       the irreducibility certificate)
#    3. the driver is exogenous: unaffected by what the agents do, and NOT reset
#       at episode boundaries
#    4. PACT widens the specs by exactly (1 action dim, 3 obs features)
#    5. the per-step cosine gate reads 1.0 while compensation is active
#    6. beta = 0 reproduces the blind policy bit for bit                (the floor property)
#    7. a partial reset clears exactly the worlds that reset            (pitfall P2)
#    8. mappo_ctde's ACTOR spec does not contain the privileged payload
#
#  Anything that fails here will otherwise show up as a quietly wrong number.

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchrl.envs import Compose, TransformedEnv  # noqa: E402
from torchrl.envs.utils import step_mdp  # noqa: E402

from benchmarl.environments import task_config_registry  # noqa: E402
from benchmarl.environments.vmas_ns.pcw_core import per_step_cosine  # noqa: E402

GROUP = "agents"
NUM_ENVS = 6
STEPS = 25
SEED = 7


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_task(name="vmas_ns/navigation_pcw", **overrides):
    task = task_config_registry[name].get_from_yaml()
    config = dict(task.config)
    config.update(overrides)
    return task.__class__(name=task.name, config=config)


def make_env(task, num_envs=NUM_ENVS, with_transforms=True):
    base = task.get_env_fun(
        num_envs=num_envs, continuous_actions=True, seed=SEED, device="cpu"
    )()
    if not with_transforms:
        return base
    transforms = task.get_env_transforms(base)
    return TransformedEnv(base, Compose(*transforms)) if transforms else base


def drive(env, actions, seed=SEED):
    """Step the env through a fixed action sequence; return per-step records."""
    env.set_seed(seed)
    td = env.reset()
    records = []
    for action in actions:
        td.set((GROUP, "action"), action)
        td = env.step(td)
        records.append(
            {
                "obs": td.get(("next", GROUP, "observation")).clone(),
                "reward": td.get(("next", GROUP, "reward")).clone(),
                "done": td.get(("next", "done")).clone(),
                "info": td.get(("next", GROUP, "info")).clone()
                if (("next", GROUP, "info") in td.keys(include_nested=True))
                else None,
            }
        )
        td = step_mdp(td)
    return records


def fixed_actions(n_steps, num_envs, n_agents, width, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        (torch.rand(num_envs, n_agents, width, generator=generator) * 2 - 1)
        for _ in range(n_steps)
    ]


def info_of(record, key):
    return record["info"].get(key).squeeze(-1)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_severity_zero_matches_stock_navigation():
    """Constraint 3: the reward function and the dynamics are the originals."""
    ns_task = make_task(ns_severity=0.0, pact_enabled=False)
    stock = task_config_registry["vmas/navigation"].get_task(
        {
            key: ns_task.config[key]
            for key in (
                "max_steps",
                "n_agents",
                "collisions",
                "agents_with_same_goal",
                "observe_all_goals",
                "shared_rew",
                "split_goals",
                "lidar_range",
                "agent_radius",
            )
        }
    )
    n_agents = ns_task.config["n_agents"]
    actions = fixed_actions(STEPS, NUM_ENVS, n_agents, 2)

    ns_records = drive(make_env(ns_task), actions)
    stock_records = drive(make_env(stock), actions)

    for t, (a, b) in enumerate(zip(ns_records, stock_records)):
        assert torch.equal(a["obs"], b["obs"]), f"observation diverges at step {t}"
        assert torch.equal(a["reward"], b["reward"]), f"reward diverges at step {t}"
        assert torch.equal(a["done"], b["done"]), f"done diverges at step {t}"
    return f"{STEPS} steps identical to vmas/navigation"


def check_n1_is_the_stationary_task():
    """Constraint 2: at N=1 the effect vanishes, at ANY severity."""
    on = make_task(n_agents=1, ns_severity=5.0, pact_enabled=False)
    off = make_task(n_agents=1, ns_severity=0.0, pact_enabled=False)
    actions = fixed_actions(STEPS, NUM_ENVS, 1, 2)

    on_records = drive(make_env(on), actions)
    off_records = drive(make_env(off), actions)

    worst_theta = 0.0
    for t, (a, b) in enumerate(zip(on_records, off_records)):
        assert torch.equal(a["obs"], b["obs"]), f"observation diverges at step {t}"
        assert torch.equal(a["reward"], b["reward"]), f"reward diverges at step {t}"
        worst_theta = max(worst_theta, float(info_of(a, "ns_theta_applied").abs().max()))
    assert worst_theta == 0.0, f"theta must be exactly 0 at N=1, got {worst_theta}"
    return "severity 5.0 at N=1 is byte-identical to severity 0"


def check_driver_is_exogenous_and_persistent():
    """Constraint 2 (not category A): the driver ignores the agents and the clock
    is not reset by an episode boundary."""
    task = make_task(ns_severity=0.8, pact_enabled=False, ns_phase_spread=True)
    n_agents = task.config["n_agents"]

    env = make_env(task)
    quiet = drive(env, [torch.zeros(NUM_ENVS, n_agents, 2) for _ in range(STEPS)])
    env = make_env(task)
    loud = drive(env, fixed_actions(STEPS, NUM_ENVS, n_agents, 2))

    for t, (a, b) in enumerate(zip(quiet, loud)):
        assert torch.allclose(
            info_of(a, "ns_A"), info_of(b, "ns_A")
        ), f"driver depends on the agents at step {t}"

    # a mid-rollout reset must not rewind the clock
    env = make_env(task)
    env.set_seed(SEED)
    td = env.reset()
    for _ in range(10):
        td.set((GROUP, "action"), torch.zeros(NUM_ENVS, n_agents, 2))
        td = env.step(td)
        before = td.get(("next", GROUP, "info")).get("ns_A").squeeze(-1).clone()
        td = step_mdp(td)
    td = env.reset()
    td.set((GROUP, "action"), torch.zeros(NUM_ENVS, n_agents, 2))
    td = env.step(td)
    after = td.get(("next", GROUP, "info")).get("ns_A").squeeze(-1)
    assert not torch.equal(before, after), "the driver clock was reset by an episode"

    spread = info_of(quiet[0], "ns_A")[:, 0]
    assert float(spread.max() - spread.min()) > 0.5, "phase spread does not cover the cycle"
    return "driver is agent-independent, survives resets, and spans the cycle"


def check_ns_actually_fires():
    """The load-bearing assumption: `process_action` is the hook VMAS calls.

    If this build of VMAS never routes through `BaseScenario.process_action`,
    every other check still passes quietly (the NS would simply be absent), so
    assert positively that thrust gets deflected and the medium gets charged.
    """
    task = make_task(ns_severity=0.8, pact_enabled=False, ns_freeze_driver=1.0)
    n_agents = task.config["n_agents"]
    records = drive(make_env(task), fixed_actions(STEPS, NUM_ENVS, n_agents, 2, seed=19))

    theta = max(float(info_of(r, "ns_theta_applied").abs().max()) for r in records)
    x2 = max(float(info_of(r, "ns_x2").abs().max()) for r in records)
    assert x2 > 1e-3, "the accumulator never charged -- process_action is not being called"
    assert theta > 1e-2, f"thrust is never deflected (max |theta| = {theta})"

    blind = make_task(ns_severity=0.0, pact_enabled=False)
    off_records = drive(
        make_env(blind), fixed_actions(STEPS, NUM_ENVS, n_agents, 2, seed=19)
    )
    diverged = any(
        not torch.equal(a["obs"], b["obs"]) for a, b in zip(records, off_records)
    )
    assert diverged, "severity 0.8 produces the same trajectory as severity 0"
    return f"max |theta| {theta:.3f} rad, max |x2| {x2:.3f}; trajectories diverge"


def check_pact_specs():
    """PACT widens the interface by exactly one action dim and three features."""
    blind = make_task(pact_enabled=False)
    pact = make_task(pact_enabled=True)

    blind_env = make_env(blind)
    pact_env = make_env(pact)

    a_blind = blind_env.full_action_spec_unbatched[(GROUP, "action")].shape[-1]
    a_pact = pact_env.full_action_spec_unbatched[(GROUP, "action")].shape[-1]
    assert a_pact == a_blind + 1, f"action dim {a_blind} -> {a_pact}"

    o_blind = blind.observation_spec(blind_env)[(GROUP, "observation")].shape[-1]
    o_pact = pact.observation_spec(pact_env)[(GROUP, "observation")].shape[-1]
    assert o_pact == o_blind + 3, f"observation dim {o_blind} -> {o_pact}"
    return f"action {a_blind}->{a_pact}, observation {o_blind}->{o_pact}"


def check_gate():
    """The one hard gate, on the real environment, with compensation active."""
    task = make_task(pact_enabled=True, pact_beta_ema=0.0, ns_severity=0.8)
    n_agents = task.config["n_agents"]
    env = make_env(task)
    # w > -1 so beta > 0: the compensation is live and feeding back into the
    # circulation, which is exactly when a timing bug would show up.
    actions = fixed_actions(STEPS, NUM_ENVS, n_agents, 3, seed=11)

    worst, checked = 1.0, 0
    for record in drive(env, actions):
        env_x2 = info_of(record, "ns_x2")
        pact_x2 = info_of(record, "pact_x2")
        cos, valid = per_step_cosine(pact_x2, env_x2)
        if valid.any():
            worst = min(worst, float(cos[valid].min()))
            checked += int(valid.sum())
    assert checked > 0, "gate never had a valid step -- the accumulator stayed zero"
    assert worst > 0.999, f"per-step cosine {worst} < 0.999 over {checked} steps"
    return f"min per-step cosine {worst:.6f} over {checked} agent-steps"


def check_beta_zero_is_blind():
    """The floor property, end to end."""
    blind = make_task(pact_enabled=False, ns_severity=0.8)
    pact = make_task(pact_enabled=True, pact_beta_ema=0.0, ns_severity=0.8)
    n_agents = blind.config["n_agents"]

    physical = fixed_actions(STEPS, NUM_ENVS, n_agents, 2, seed=13)
    # w = -1 maps to beta = 0 under the affine mode
    with_w = [
        torch.cat((a, -torch.ones(NUM_ENVS, n_agents, 1)), dim=-1) for a in physical
    ]

    blind_records = drive(make_env(blind), physical)
    pact_records = drive(make_env(pact), with_w)

    obs_dim = blind_records[0]["obs"].shape[-1]
    for t, (a, b) in enumerate(zip(blind_records, pact_records)):
        assert torch.equal(
            a["obs"], b["obs"][..., :obs_dim]
        ), f"beta=0 diverges from blind at step {t}"
        assert torch.equal(a["reward"], b["reward"]), f"reward diverges at step {t}"
    return "beta=0 is bit-identical to the blind environment"


def check_partial_reset():
    """Pitfall P2: wrapper state must die exactly when its episode does."""
    task = make_task(pact_enabled=True, ns_severity=0.8)
    n_agents = task.config["n_agents"]
    env = make_env(task)
    env.set_seed(SEED)
    td = env.reset()
    actions = fixed_actions(12, NUM_ENVS, n_agents, 3, seed=17)
    for action in actions:
        td.set((GROUP, "action"), action)
        td = env.step(td)
        td = step_mdp(td)

    reset_mask = torch.zeros(NUM_ENVS, 1, dtype=torch.bool)
    reset_mask[: NUM_ENVS // 2] = True
    td.set("_reset", reset_mask)
    td = env.reset(td)

    td.set((GROUP, "action"), actions[0])
    td = env.step(td)
    info = td.get(("next", GROUP, "info"))
    env_x2 = info.get("ns_x2").squeeze(-1)
    pact_x2 = info.get("pact_x2").squeeze(-1)

    assert torch.allclose(env_x2, pact_x2, atol=1e-5), "env and PACT disagree after a partial reset"
    fresh = env_x2[: NUM_ENVS // 2].abs()
    stale = env_x2[NUM_ENVS // 2 :].abs()
    assert float(fresh.max()) < float(stale.mean()), (
        "reset worlds do not have a fresher accumulator than untouched ones"
    )
    return "partial reset clears exactly the reset worlds, on both sides"


def check_ctde_actor_is_blind_to_the_payload():
    """The CTDE payload reaches the critic and provably not the actor."""
    import tempfile

    from benchmarl.algorithms import MappoCtdeConfig
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models import MlpConfig

    config = ExperimentConfig.get_from_yaml()
    config.loggers = []
    config.create_json = False
    config.render = False
    config.evaluation = False
    config.max_n_iters = 1
    config.max_n_frames = None
    config.on_policy_collected_frames_per_batch = 120
    config.on_policy_n_envs_per_worker = 2
    config.on_policy_minibatch_size = 60
    config.evaluation_episodes = 2
    config.checkpoint_interval = 0
    config.checkpoint_at_end = False

    with tempfile.TemporaryDirectory() as folder:
        config.save_folder = folder
        experiment = Experiment(
            task=make_task(pact_enabled=True),
            algorithm_config=MappoCtdeConfig.get_from_yaml(),
            model_config=MlpConfig.get_from_yaml(),
            seed=0,
            config=config,
        )
        try:
            actor_keys = list(experiment.observation_spec[GROUP].keys())
            assert "ctde_state" not in actor_keys, (
                f"the privileged payload leaked into the actor's observation: {actor_keys}"
            )
            critic = experiment.algorithm.get_critic(GROUP)
            critic_keys = [str(k) for k in critic.in_keys]
            assert any("ctde_state" in k for k in critic_keys), (
                f"the critic does not read the payload: {critic_keys}"
            )
        finally:
            experiment.close()
    return "critic reads ctde_state; actor observation spec does not contain it"


CHECKS = [
    ("severity 0 == stock navigation", check_severity_zero_matches_stock_navigation),
    ("N=1 irreducibility certificate", check_n1_is_the_stationary_task),
    ("driver exogenous + persistent", check_driver_is_exogenous_and_persistent),
    ("the NS actually fires", check_ns_actually_fires),
    ("PACT spec widening", check_pact_specs),
    ("per-step cosine gate", check_gate),
    ("beta=0 floor property", check_beta_zero_is_blind),
    ("partial reset", check_partial_reset),
    ("CTDE payload is critic-only", check_ctde_actor_is_blind_to_the_payload),
]


def main():
    failures = []
    for name, check in CHECKS:
        try:
            detail = check()
            print(f"PASS  {name:<34} {detail}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name:<34} {exc}")
            traceback.print_exc()
    print(f"\n{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed")
    if failures:
        print("Do not start training until these pass: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
