#  Shared plumbing for the PACT evaluation scripts.
#
#  Both Phase 1 (certify sigma*) and the Phase-2 arm report need the same three
#  things: load a trained checkpoint, build a *fresh* environment at a patched
#  task config, and roll the policy through it counting returns the way
#  BenchMARL does.  Keeping them here means the two scripts cannot disagree
#  about what "return" means.

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from tensordict import TensorDictBase
from torchrl.envs import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.environments import task_config_registry
from benchmarl.environments.vmas_ns.common import VmasNsClass
from benchmarl.experiment import Experiment
from benchmarl.utils import _add_rnn_transforms

TASK_NAME = "vmas_ns/navigation_pcw"

#: Config patch applied to every reloaded experiment: we only want its policy.
_QUIET = {
    "loggers": [],
    "create_json": False,
    "render": False,
    "evaluation": False,
    "checkpoint_interval": 0,
    "checkpoint_at_end": False,
}


def load_experiment(checkpoint: str, device: Optional[str] = None) -> Experiment:
    """Reload a BenchMARL experiment from a checkpoint, loggers muted."""
    patch: Dict[str, Any] = dict(_QUIET)
    if device is not None:
        patch.update(
            sampling_device=device,
            train_device=device,
            buffer_device=device,
            # Arms are typically trained on GPU and replayed on CPU; without a
            # map location torch.load would try to restore onto a device this
            # process may not have.
            restore_map_location=device,
        )
    checkpoint = str(Path(checkpoint).resolve())
    return Experiment.reload_from_file(checkpoint, experiment_patch=patch)


def task_from(
    experiment: Optional[Experiment] = None, **overrides
) -> VmasNsClass:
    """The navigation_pcw task, optionally inheriting a checkpoint's own config.

    Inheriting matters: a PACT checkpoint was trained against an environment
    whose action and observation specs are wider, so an evaluation environment
    built from the yaml defaults would not fit its policy.
    """
    if experiment is not None:
        base = copy.deepcopy(experiment.task.config)
        name = experiment.task.name
        task_cls = experiment.task.__class__
    else:
        template = task_config_registry[TASK_NAME].get_from_yaml()
        base = copy.deepcopy(template.config)
        name = template.name
        task_cls = template.__class__
    base.update(overrides)
    return task_cls(name=name, config=base)


def build_env(
    task: VmasNsClass,
    *,
    num_envs: int,
    seed: int,
    device: str,
    model_config=None,
    probe_factory=None,
):
    """A fresh evaluation env with the task's transforms plus an optional probe.

    ``probe_factory`` is a callable taking the freshly-built base env and
    returning extra transforms -- that indirection exists because the Phase-1
    probe needs the env's action bounds to clip exactly as the env would.

    Returns ``(env, probes)``.
    """
    base_env = task.get_env_fun(
        num_envs=num_envs,
        continuous_actions=True,
        seed=seed,
        device=device,
    )()
    probes = list(probe_factory(base_env)) if probe_factory is not None else []
    transforms = list(task.get_env_transforms(base_env)) + probes
    transforms.append(task.get_reward_sum_transform(base_env))
    env = TransformedEnv(base_env, Compose(*transforms)).to(device)
    if model_config is not None and model_config.is_rnn:
        group_map = task.group_map(env)
        env = _add_rnn_transforms(lambda: env, group_map, model_config)()
    return env, probes


def episode_returns(rollout: TensorDictBase, group: str) -> torch.Tensor:
    """Per-environment episode return, truncated at each env's first ``done``.

    Mirrors BenchMARL's evaluation accounting: sum the per-agent reward over the
    episode, then average over agents.  Anything a vectorised env produces after
    its first done belongs to a new episode and must not be counted.
    """
    done = rollout.get(("next", "done"))
    while done.dim() > 2:
        done = done.any(dim=-1)
    already_done = torch.cat(
        (torch.zeros_like(done[:, :1]), done[:, :-1]), dim=1
    ).cumsum(dim=1)
    valid = (already_done == 0).to(torch.float32)  # (E, T)

    reward = rollout.get(("next", group, "reward"))  # (E, T, N, 1)
    while valid.dim() < reward.dim():
        valid = valid.unsqueeze(-1)
    return (reward * valid).sum(dim=1).mean(dim=tuple(range(1, reward.dim() - 1)))


@torch.no_grad()
def evaluate(
    experiment: Experiment,
    task: VmasNsClass,
    *,
    num_envs: int,
    seed: int,
    device: str,
    probe_factory=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Roll the checkpoint's policy through a fresh env.  Returns per-env returns."""
    env, probes = build_env(
        task,
        num_envs=num_envs,
        seed=seed,
        device=device,
        model_config=experiment.model_config,
        probe_factory=probe_factory,
    )
    group = next(iter(task.group_map(env)))
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            rollout = env.rollout(
                max_steps=task.max_steps(env),
                policy=experiment.policy,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
        returns = episode_returns(rollout, group)
        extras = _extra_stats(rollout, group, probes)
    finally:
        env.close()
    return returns.cpu(), extras


def _extra_stats(rollout, group, extra_transforms) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    info_key = ("next", group, "info")
    if info_key in rollout.keys(include_nested=True):
        info = rollout.get(info_key)
        for key in ("ns_abs_theta", "ns_c", "ns_A", "pact_beta", "pact_x2"):
            if key in info.keys():
                stats[key] = float(info.get(key).to(torch.float32).mean())
    for transform in extra_transforms:
        if hasattr(transform, "sat_frac"):
            stats["sat_frac"] = transform.sat_frac
    return stats


def bootstrap_ci(
    values: torch.Tensor, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap CI over per-environment returns."""
    generator = torch.Generator().manual_seed(seed)
    n = values.numel()
    if n < 2:
        return float("nan"), float("nan")
    idx = torch.randint(0, n, (n_boot, n), generator=generator)
    means = values.reshape(-1)[idx].mean(dim=1)
    lo = torch.quantile(means, alpha / 2)
    hi = torch.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def fmt_ci(values: torch.Tensor) -> str:
    lo, hi = bootstrap_ci(values)
    return f"{float(values.mean()):+7.4f} [{lo:+.4f}, {hi:+.4f}]"
