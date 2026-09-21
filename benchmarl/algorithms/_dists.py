#  Three things every baseline that reaches inside a policy distribution needs.
#
#  A published objective that is not PPO's -- a proximal term, a natural
#  gradient, a supervised fit to the actions a policy took -- has to get at the
#  distribution's PARAMETERS, not just at a log-probability.  torchrl hands
#  back a ``TanhNormal``, an ``IndependentNormal`` or a ``Categorical``
#  depending on the action space and on ``use_tanh_normal``, and the first of
#  those wraps its base distribution, so "what are loc and scale" is three
#  lines of unwrapping rather than an attribute access.
#
#  ``lcpo.py`` has the same three helpers inline; it predates this module and
#  is left alone because it has been run end to end.  Everything added after it
#  uses these.

from __future__ import annotations

from typing import Tuple

import torch

from benchmarl.algorithms._baseline_math import categorical_kl, gaussian_kl


def params_of(dist) -> Tuple[str, object]:
    """``("categorical", probs)`` or ``("gaussian", (loc, scale))``.

    ``TanhNormal`` is a transformed distribution: the tanh-and-affine map onto
    the action box is a FIXED bijection shared by every policy in a run, so the
    parameters of the base Normal identify the policy completely and any
    divergence between two of them is the divergence between the transformed
    pair.
    """
    probs = getattr(dist, "probs", None)
    if probs is not None:
        return ("categorical", probs)
    loc = getattr(dist, "loc", None)
    scale = getattr(dist, "scale", None)
    if loc is None or scale is None:
        base = dist
        while hasattr(base, "base_dist"):
            base = base.base_dist
        loc, scale = base.loc, base.scale
    return ("gaussian", (loc, scale))


def entropy_of(dist) -> torch.Tensor:
    """``dist.entropy()`` where it exists, one-sample Monte Carlo where not.

    ``TanhNormal`` has no closed-form entropy; torchrl's own PPO loss falls
    back to exactly this estimator.
    """
    try:
        entropy = dist.entropy()
        if entropy.isfinite().all():
            return entropy
    except NotImplementedError:
        pass
    sample = dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()
    return -dist.log_prob(sample)


def kl_of(old: Tuple[str, object], new: Tuple[str, object]) -> torch.Tensor:
    """``KL(pi_old || pi_new)``, summed over the action dimensions."""
    kind, old_p = old
    _, new_p = new
    if kind == "categorical":
        return categorical_kl(old_p, new_p)
    return gaussian_kl(*old_p, *new_p)
