#  The exogenous driver, the severity dial, the loading ratio and the harm.
#  NS-1.1, NS-1.3, NS-1.4, NS-2.1 .. NS-2.5, and the I.2 category-C signature.
#
#  torch only.  No vmas.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from road_ns.structure import RoadStructure

__all__ = ["DialParams", "driver_A", "sensitivity", "dial_g", "loading", "performance",
           "harm", "excess"]


# ---------------------------------------------------------------------------
#  NS-2.4 -- the anchor.  A published constant, not a round number.
# ---------------------------------------------------------------------------
#: Highway Capacity Manual capacity adjustment factor for heavy rain: capacity
#: falls to ~0.86 of dry.  So the loss at sigma = 1 is 0.14.  Every sigma > 1
#: MUST be labelled a beyond-physical stress test wherever it appears.
HCM_HEAVY_RAIN_LOSS = 0.14


@dataclass(frozen=True)
class DialParams:
    """Declared constants.  ``severity`` is the only experimental variable."""

    severity: float = 1.0

    # -- the driver (I.3 reference form) ------------------------------------
    period: int = 100
    """Steps per weather cycle."""
    wet_fraction: float = 0.5
    """Fraction of the cycle the storm occupies.  The remainder is EXACTLY dry,
    which is what makes NS-2.5's placebo regime provable rather than merely
    small."""

    loss_at_sigma1: float = HCM_HEAVY_RAIN_LOSS

    # -- per-element sensitivity -------------------------------------------
    sens_lo: float = 0.4
    sens_hi: float = 2.0
    """Facility-specific capacity adjustment, mean 1, clipped to [0.4, 2.0], as
    published capacity adjustment factors are.  It matters more than it looks: a
    uniform derating leaves the binding element unchanged at every severity,
    which makes the ceiling decomposition sigma-invariant and hides the effect
    you are trying to show."""

    g_floor: float = 1e-3

    # -- the performance function (III.1 "regime, not convention") ---------
    alpha: float = 2.28
    """``f(u) = 1 + alpha * u``.  LINEAR, not the HCM/BPR quartic: those
    coefficients are calibrated near capacity, and this medium runs well below
    it, where a quartic term is numerically dead.  In the low-utilisation regime
    queueing delay is rho/(1-rho) ~ rho, i.e. linear.

    The value is calibrated by a stated one-line procedure against an
    observable -- mean(realized/nominal - 1) divided by mean loading -- and
    MUST be re-derived per instance rather than inherited.  2.28 is the URB
    reference figure and is a PLACEHOLDER here until the probe measures it; see
    ``calibrate_alpha``."""

    mean_preserve: bool = False
    """NS-5.1: normalise the multiplier by its own cycle mean so only the shape
    varies and total capacity is unchanged."""


# ---------------------------------------------------------------------------
#  NS-1.3 -- the exogenous driver
# ---------------------------------------------------------------------------


def driver_A(step: Tensor, p: DialParams) -> Tensor:
    """``A(t) in [0, 1]``: a smooth storm that starts and ends at EXACTLY zero.

    A function of observable time alone. No agent's action can influence it, and
    it reaches the agents only by shrinking capacity -- never by adding a term
    to the loss.

        phi = (t mod P) / P
        A   = sin^2( pi * min(phi/w, 1) )  if phi < w  else  0
    """
    phi = (step.to(torch.float32) % p.period) / float(p.period)
    ramp = torch.sin(math.pi * torch.clamp(phi / p.wet_fraction, max=1.0)) ** 2
    return torch.where(phi < p.wet_fraction, ramp, torch.zeros_like(ramp))


def sensitivity(structure: RoadStructure, p: DialParams, seed: int = 0) -> Tensor:
    """Per-element capacity-adjustment sensitivity ``s_a``, mean 1.

    Deterministic in the element's own declared attributes -- capacity and
    class -- so it is structure, not run data, and it is identical across arms
    and seeds.  Narrow single-lane urban elements lose proportionally more
    capacity in rain than wide multi-lane ones, which is the direction published
    adjustment factors go.
    """
    _, cls = structure.element_classes()
    cap = structure.capacity
    # smaller capacity -> more sensitive; normalised to mean 1 then clipped
    raw = (cap.median() / cap.clamp_min(1e-9)).sqrt()
    raw = raw + 0.15 * (cls.to(torch.float32) - cls.to(torch.float32).mean())
    s = raw / raw.mean()
    return s.clamp(p.sens_lo, p.sens_hi)


# ---------------------------------------------------------------------------
#  NS-2 -- the severity dial
# ---------------------------------------------------------------------------


def dial_g(a: Tensor, s: Tensor, p: DialParams) -> Tensor:
    """``g_a = clip(1 - sigma * L * A(t) * s_a, g_floor, 1)``.  ``(B, A)``.

    NS-2.1 identity at zero is EXACT: ``sigma = 0`` makes the product exactly
    ``0.0`` and ``1 - 0.0`` exactly ``1.0``, at every driver value, so the stock
    task is recovered bit for bit.
    NS-2.2 monotone in sigma at every driver value, since the subtracted term is
    non-negative and linear in sigma.
    NS-2.3 never generous: the upper clip at 1 keeps only the harmful half.
    """
    a = a.reshape(-1, 1)
    g = 1.0 - p.severity * p.loss_at_sigma1 * a * s.reshape(1, -1)
    g = g.clamp(p.g_floor, 1.0)
    if p.mean_preserve:
        g = g / _cycle_mean_g(s, p).reshape(1, -1)
    return g


def _cycle_mean_g(s: Tensor, p: DialParams) -> Tensor:
    """Analytic cycle mean of ``g`` per element, from the declared driver only.

    Never from run data: that would turn a declared model class into a fit.
    """
    t = torch.arange(p.period, dtype=torch.float32, device=s.device)
    phi = (t % p.period) / float(p.period)
    ramp = torch.sin(math.pi * torch.clamp(phi / p.wet_fraction, max=1.0)) ** 2
    a = torch.where(phi < p.wet_fraction, ramp, torch.zeros_like(ramp))
    g = (
        1.0 - p.severity * p.loss_at_sigma1 * a.reshape(-1, 1) * s.reshape(1, -1)
    ).clamp(p.g_floor, 1.0)
    return g.mean(0).clamp_min(1e-6)


# ---------------------------------------------------------------------------
#  NS-1.1 -- the loading ratio
# ---------------------------------------------------------------------------


def loading(
    element_load: Tensor,
    g: Tensor,
    structure: RoadStructure,
    route_elements: Sequence[Sequence[int]],
) -> Tuple[Tensor, Tensor]:
    """``u_i = max over elements a used by i of load_a / (capacity_a * g_a)``.

    **MAX, not mean.**  Congestion is a property of the worst element;
    averaging over dozens dilutes the signal toward zero.  Switching this sensor
    from mean to max raised applied compensation sixfold on the POWER instance.

    Args:
        element_load: ``(B, A)`` vehicles currently on each element.
        g:            ``(B, A)`` capacity multiplier.
        route_elements: per agent, the element indices it traverses.

    Returns:
        ``(u, binding)``, each ``(B, N)`` -- the loading and which element sets it.
    """
    ratio = element_load / (structure.capacity.reshape(1, -1) * g).clamp_min(1e-12)
    B = ratio.shape[0]
    n = len(route_elements)
    u = torch.zeros(B, n, dtype=ratio.dtype, device=ratio.device)
    binding = torch.zeros(B, n, dtype=torch.long, device=ratio.device)
    for i, elems in enumerate(route_elements):
        if len(elems) == 0:
            continue
        idx = torch.as_tensor(list(elems), dtype=torch.long, device=ratio.device)
        m, k = ratio[:, idx].max(dim=-1)
        u[:, i] = m
        binding[:, i] = idx[k]
    return u, binding


def performance(u: Tensor, p: DialParams) -> Tensor:
    """``f(u) = 1 + alpha * u`` -- realized cost as a multiple of free-flow."""
    return 1.0 + p.alpha * u


# ---------------------------------------------------------------------------
#  I.2 -- the category-C signature
# ---------------------------------------------------------------------------


def harm(u_derated: Tensor, u_nominal: Tensor, p: DialParams) -> Tensor:
    """``harm_i = f(u_i derated) / f(u_i nominal)``  -- a RATIO, never a penalty.

    Two exact identities, and they are the whole point:

    * ``u_i = 0  =>  harm = 1`` exactly -- a lone agent is untouched at ANY
      severity, because capacity sits in the denominator of a loading ratio, so
      the driver multiplies every peer's contribution and leaves a solitary
      agent's term at zero.
    * ``g = 1  =>  u_derated == u_nominal  =>  harm = 1`` exactly -- no severity,
      or a placebo day.

    The environment applies this by dividing the achievable speed, so the reward
    function is never touched: the agent is paid exactly what it was paid
    before, for a journey the medium made slower.
    """
    return performance(u_derated, p) / performance(u_nominal, p)


def excess(u_derated: Tensor, g_binding: Tensor) -> Tensor:
    """``Delta_i = u_i * (1 - g)`` -- the loading excess over the sigma=0
    counterfactual, which I.5 partitions by who can move it."""
    return u_derated * (1.0 - g_binding)


def calibrate_alpha(mean_relative_excess: float, mean_loading: float) -> float:
    """The stated one-line procedure of III.1.

    ``alpha = mean(realized/nominal - 1) / mean(u)``.  A measurement against an
    observable, not a knob turned toward a result -- and it belongs in the
    ablation table either way.
    """
    if mean_loading <= 0:
        raise ValueError("cannot calibrate alpha at zero loading")
    return mean_relative_excess / mean_loading
