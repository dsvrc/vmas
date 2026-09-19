#  Thin shims over the parts of torchrl's loss modules that the baselines have
#  to reach into.
#
#  Every baseline in `baselines/` reimplements a published objective on top of a
#  stock torchrl loss, which means calling a handful of methods that torchrl
#  marks private (`_log_weight`, `_get_entropy`, `_clip_bounds`).  BenchMARL
#  pins `torchrl>=0.10,<0.12` and those signatures moved inside that window, so
#  every such call goes through here rather than being spelled out in eight
#  files.  If a future torchrl breaks one of them, this is the single file that
#  reports it, by name, instead of eight tracebacks from inside a training run.

from __future__ import annotations

from typing import Any, Tuple

import torch


def log_weight(loss_module, tensordict, adv_shape) -> Tuple[Any, Any, Any]:
    """``(log_weight, dist, kl_approx)`` from a torchrl PPO loss.

    ``log_weight`` is ``log pi_new(a|s) - log pi_old(a|s)`` already summed over
    the action dimensions, shaped like the advantage.  For a BenchMARL group
    that is ``(*batch, n_agents, 1)``: one joint ratio per agent, which is
    exactly HARL's ``torch.prod(exp(new - old), dim=-1)``.
    """
    try:
        return loss_module._log_weight(tensordict, adv_shape=adv_shape)
    except TypeError:  # torchrl < 0.11 had no adv_shape argument
        return loss_module._log_weight(tensordict)


def entropy(loss_module, dist, adv_shape) -> torch.Tensor:
    try:
        return loss_module._get_entropy(dist, adv_shape=adv_shape)
    except TypeError:
        return loss_module._get_entropy(dist)


def entropy_coeff(loss_module) -> float:
    """The entropy multiplier, under either spelling torchrl has used."""
    for name in ("entropy_coeff", "entropy_coef"):
        value = getattr(loss_module, name, None)
        if value is not None and not callable(value):
            return value
    return 0.0


def clip_bounds(loss_module):
    """``(log(1-eps), log(1+eps))`` -- torchrl clamps in log space."""
    bounds = getattr(loss_module, "_clip_bounds", None)
    if bounds is not None:
        return bounds
    eps = float(loss_module.clip_epsilon)
    return (
        torch.tensor(1.0 - eps).log().item(),
        torch.tensor(1.0 + eps).log().item(),
    )


def leaf_params(params) -> list:
    """``[(nested_key, leaf_tensor), ...]`` for a ``TensorDictParams``."""
    return list(params.items(True, True))


def callback_base():
    """``benchmarl.experiment.callback.Callback``, imported at CALL time.

    It must not be imported at module level from anywhere under
    ``benchmarl/algorithms/``.  ``benchmarl/__init__.py`` imports
    ``benchmarl.algorithms`` first; if an algorithm module then imports
    ``benchmarl.experiment``, that package's ``experiment.py`` runs
    ``from benchmarl.algorithms import IppoConfig, MappoConfig`` against a
    half-built ``benchmarl.algorithms`` -- whose ``__init__`` has only reached
    the line that started all this -- and ``import benchmarl`` dies with

        ImportError: cannot import name 'IppoConfig' from partially
        initialized module 'benchmarl.algorithms'

    ``mappo_ctde.py`` avoids the same cycle through ``benchmarl.environments``
    the same way.  ``baselines/verify.py`` checks the rule statically, because
    it is invisible until the package is actually imported.
    """
    from benchmarl.experiment.callback import Callback

    return Callback


def attach_callback(experiment, callback) -> None:
    """Add a callback to a live ``Experiment``, whatever container it holds.

    ``Experiment.__init__`` keeps whatever it was handed, and the two entry
    points hand it different things: a notebook builds an ``Experiment``
    directly and gets the ``[]`` default, while ``load_experiment_from_hydra``
    -- the path every launcher in this repo uses -- defaults to ``callbacks=()``,
    a TUPLE.  So ``experiment.callbacks.append(...)`` is an ``AttributeError``
    on the cluster and works fine in a notebook.

    Rebinding the attribute covers both.  ``CallbackNotifier`` only ever
    iterates the container, so replacing it is safe; the callback's
    ``experiment`` back-reference is set here because the notifier only does
    that for the callbacks present at construction.
    """
    callback.experiment = experiment
    experiment.callbacks = list(experiment.callbacks) + [callback]
