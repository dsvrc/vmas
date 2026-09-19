#  The discrete linear extended state observer, with nothing but torch behind
#  it, so `baselines/verify.py` can exercise it without a simulator.
#
#  BASELINES.md B10's first non-learning compensator: "estimate the lumped
#  disturbance from the agent's own residual with a first-order observer and
#  cancel it next step.  No peer information, no basis, one gain."
#
#  Linear ADRC's ESO, in the discrete form with both observer poles placed
#  together at ``1 - w`` on the unit disc -- Gao's bandwidth parameterisation
#  (Gao, "Scaling and bandwidth-parameterization based controller tuning", ACC
#  2003; discrete form in Miklosovic, Radke & Gao, "Discrete implementation and
#  generalization of the extended state observer", ACC 2006):
#
#      e  = y(t-1) - z1
#      z1 <- z1 + z2 + beta1 * e         beta1 = 2w
#      z2 <- z2 + beta2 * e              beta2 = w^2
#      pred(t) = z1 + z2
#
#  ``z1`` is the observed disturbance, ``z2`` its rate, and ``z1 + z2`` the
#  one-step-ahead extrapolation -- which is what an observer buys over a plain
#  low-pass on the same signal, and the only reason it can partly cover the
#  one-step lag.
#
#  The gains follow from the pole placement and nothing else.  With
#  ``x = [z1, z2]`` and a disturbance that the observer models as a double
#  integrator, the estimation error obeys
#
#      e1 <- (1 - beta1) e1 + e2
#      e2 <- -beta2 e1 + e2
#
#  whose characteristic polynomial is ``z^2 - (2 - beta1) z + (1 - beta1 +
#  beta2)``.  Setting that equal to ``(z - (1-w))^2`` gives
#
#      2 - beta1        = 2 (1 - w)      ->  beta1 = 2w
#      1 - beta1 + beta2 = (1 - w)^2     ->  beta2 = w^2
#
#  which is the sampled-data ESO of Miklosovic, Radke & Gao with sampling
#  period 1 and ``beta = e^{-w} ~ 1 - w``.  `baselines/verify.py` checks the
#  pole placement directly, because a wrong gain here does not crash -- it
#  produces an observer that simply tracks badly, and no training curve tells
#  you which.  The first draft of this file used ``beta1 = 1 - (1-w)^2``, which
#  is ``2w - w^2``, and that check is what found it.

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

__all__ = ["eso_gains", "eso_step"]


def eso_gains(bandwidth: float) -> Tuple[float, float]:
    """``(beta1, beta2)`` for an observer bandwidth ``w`` in ``(0, 1)``."""
    w = float(bandwidth)
    if not (0.0 < w < 1.0):
        raise ValueError(
            f"eso bandwidth={w}: both observer poles sit at 1 - w on the unit "
            "disc, so w must be in (0, 1). w -> 0 is an observer that never "
            "moves; w -> 1 is one that copies the last measurement and has no "
            "filtering left."
        )
    return 2.0 * w, w**2


def eso_step(
    z1: Tensor, z2: Tensor, y: Tensor, beta1: float, beta2: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """One observer update.  Returns ``(z1, z2, prediction)``."""
    e = y - z1
    z1_next = z1 + z2 + beta1 * e
    z2_next = z2 + beta2 * e
    return z1_next, z2_next, z1_next + z2_next
