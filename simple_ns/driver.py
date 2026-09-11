#  The exogenous driver, the severity dial, and the declared class constants.
#
#  ---------------------------------------------------------------------------
#  Which cell of the classification this is
#  ---------------------------------------------------------------------------
#  NS_design_guide.md sorts non-stationarity three ways:
#
#      (A) learning-induced   co-learners keep changing        vanishes if
#                             policies                          partners frozen
#      (B) exogenous          the world drifts on its own       a LONE agent
#                                                               feels it
#      (C) interaction-       an exogenous driver exists but    a lone agent
#          mediated           reaches i ONLY through others     feels NOTHING
#
#  and PACT_NS_SPEC's II.6 cuts (C) again by whether the harm has an inverse.
#  ``road_ns`` is (C, no inverse): a congested lanelet cannot be un-congested,
#  so the method may claim identification and STEERING only, and the recoverable
#  margin is bounded by the coordination gap.
#
#  This module is (C, INVERTIBLE).  The disturbance is an additive force in the
#  agent's own action space, so a correct estimate cancels it exactly -- II.6's
#  first row, where the method may claim identification AND compensation.  That
#  is the cell where a return curve can fall a long way and be brought back,
#  which is what makes it the demonstration cell rather than the bounded one.
#
#  ---------------------------------------------------------------------------
#  The mechanism, and why it is not a gremlin
#  ---------------------------------------------------------------------------
#  Several agents act on a shared medium -- a rigid payload in ``balance`` and
#  ``transport``, the surrounding fluid in ``sampling`` and ``navigation``.  The
#  medium transmits each agent's exertion to the others: a redundant actuator
#  feels its partners fighting it through the body it is holding, a rotorcraft
#  flies through its neighbours' downwash.  Both are named effects with their own
#  literature (force fight; rotor-wake interaction).
#
#  How strongly the medium transmits is not constant.  Bearing and servo
#  compliance grow over a deployment; air density falls as the day warms.  That
#  is the exogenous driver A(t): a function of observable time that no agent can
#  influence, and which reaches an agent only by scaling what its NEIGHBOURS do
#  to it.  With one agent there are no neighbours and the term is identically
#  zero at every severity -- structurally, not approximately.
#
#  torch only.  No vmas.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "DialParams",
    "driver_A",
    "class_constants",
    "beta_star",
    "cycle_mean_A",
]


@dataclass(frozen=True)
class DialParams:
    """Declared constants.  ``severity`` is the only experimental variable."""

    severity: float = 1.0
    """sigma.  0.0 makes the disturbance EXACTLY zero at every driver value, so
    the stock scenario is recovered bit for bit."""

    # -- the driver (I.3 reference form) ------------------------------------
    period: int = 100
    """Steps per driver cycle."""
    wet_fraction: float = 0.5
    """Fraction of the cycle the disturbance occupies.  The remainder is EXACTLY
    zero, which is what makes NS-2.5's placebo regime provable rather than
    merely small."""

    loss_at_sigma1: float = 0.14
    """The severity scale at sigma = 1, as a fraction of the action range.

    **This is a stated calibration target, NOT a published constant, and the
    difference matters.**  ``road_ns`` anchors sigma = 1 on the Highway Capacity
    Manual's heavy-rain capacity adjustment factor -- a number a traffic engineer
    defends -- so NS-2.4 is satisfied outright there.  ``balance``,
    ``transport`` and ``sampling`` are synthetic arenas with no published
    constant to anchor to, so the honest statement is a procedure instead:

        at sigma = 1, at the driver's peak, with every peer exerting its full
        action range, the disturbance reaching an agent is 0.14 of its own
        action range.

    0.14 is carried over from the HCM figure only so the two families share a
    severity scale and the ladders are readable against each other.  Say this in
    the paper; do not present it as an anchor it is not.  See
    ``simple_ns/README.md``."""

    rho: float = 0.9
    """Transmission memory.  The medium does not respond instantly: compliance
    and wake both integrate.  The leak is applied to the PUBLIC channels, not to
    the private disturbance, so the model stays exactly linear in quantities the
    agent can compute -- see ``coupling.Channels``."""

    # -- the declared classes (P-1.1, P-1.2) --------------------------------
    n_types: int = 3
    """Number of declared agent classes, ``r``.  INDEPENDENT of the number of
    agents: adding agents adds no parameters.  A class is a public property of
    the hardware -- here the actuator model each agent is built from -- and it is
    what the reduction projects the unknown transmission field onto."""

    recv_spread: float = 0.6
    """Spread of the per-class RECEIVER susceptibility.  Public, and it is what
    makes the coupling operator asymmetric: a compliant unit is compliant to its
    neighbours' load, not weak in isolation."""

    send_spread: float = 0.8
    """Spread of the per-class SENDER gain.  This is beta*, the quantity the
    estimator has to identify, and it is NOT handed to the agent."""

    kernel_lambda: float = 0.35
    """Transmission length scale, in world units.  ``w(d) = 1/(1 + (d/lam)^2)``:
    a neighbour twice the length scale away transmits a fifth as much.  Declared
    structure -- the falloff of a wake or of a compliant linkage -- evaluated on
    observed geometry, exactly as ``road_ns``'s loading is the declared operator
    evaluated on observed occupancy."""

    y_clip: float = 10.0
    """P-2.1's declared outlier bound on the sensor."""


# ---------------------------------------------------------------------------
#  NS-1.3 -- the exogenous driver
# ---------------------------------------------------------------------------


def driver_A(step: Tensor, p: DialParams) -> Tensor:
    """``A(t) in [0, 1]``: a smooth episode that starts and ends at EXACTLY zero.

    A function of observable time alone.  No agent's action can influence it, and
    it reaches the agents only by scaling the cross-agent term -- never by adding
    a term to the loss.

        phi = (t mod P) / P
        A   = sin^2( pi * min(phi/w, 1) )  if phi < w  else  0

    The same form ``road_ns`` uses, so the two families share a driver and the
    placebo argument is the same argument.
    """
    phi = (step.to(torch.float32) % p.period) / float(p.period)
    if p.wet_fraction <= 0.0:
        # A permanently quiet cycle: the placebo arm.  Returned as an exact zero
        # rather than reached through a division, so the dial is provably inert
        # for every sigma instead of merely very small.
        return torch.zeros_like(phi)
    ramp = torch.sin(math.pi * torch.clamp(phi / p.wet_fraction, max=1.0)) ** 2
    return torch.where(phi < p.wet_fraction, ramp, torch.zeros_like(ramp))


def cycle_mean_A(p: DialParams) -> float:
    """Mean driver value over one full cycle.  I.6 wants the level reported, not
    just the peak: half the cycle is exactly quiet, so this is well under 0.5."""
    t = torch.arange(p.period, dtype=torch.float32)
    return float(driver_A(t, p).mean())


# ---------------------------------------------------------------------------
#  the declared class constants
# ---------------------------------------------------------------------------


def class_constants(p: DialParams) -> tuple[Tensor, Tensor]:
    """``(recv, send)``, each ``(r,)`` with mean exactly 1.

    Deterministic in the class index alone -- no RNG, no run data -- so they are
    identical across arms, seeds and severities, and the operator is *declared*
    rather than fitted (NS-1.2).  A cosine fan is used simply because it is
    reproducible and spreads the values without a magic table.

    ``recv`` is public: it is the receiver's own compliance, a property of its
    own hardware, and it multiplies the load reaching it.
    ``send`` is NOT public: it is how much a unit of a class-m neighbour's
    exertion actually costs today, and it is what the estimator must recover.
    """
    r = int(p.n_types)
    if r < 1:
        raise ValueError(f"n_types must be >= 1, got {r}")
    idx = torch.arange(r, dtype=torch.float32)
    if r == 1:
        return torch.ones(1), torch.ones(1)
    fan = torch.cos(math.pi * idx / (r - 1))  # +1 .. -1, deterministic
    recv = 1.0 + p.recv_spread * fan
    send = 1.0 - p.send_spread * fan  # deliberately ANTI-aligned with recv
    return recv / recv.mean(), send / send.mean()


def beta_star(a: Tensor, p: DialParams) -> Tensor:
    """The true per-class transmission gain right now.  ``(B, r)``.

        beta*_m(t) = sigma * L * A(t) * send_m

    This is the whole of what drifts, and it is the whole of what the estimator
    is asked to track.  Three properties hold by construction:

    * ``sigma = 0`` makes it exactly ``0.0`` at every driver value, so the
      disturbance is exactly zero and the stock scenario is recovered bit for
      bit (NS-2.1).
    * monotone in sigma at every driver value, since it is linear in sigma with
      a non-negative coefficient (NS-2.2).
    * exactly zero wherever ``A(t)`` is exactly zero, which is half of every
      cycle (NS-2.5).
    """
    _, send = class_constants(p)
    scale = p.severity * p.loss_at_sigma1
    return scale * a.reshape(-1, 1) * send.reshape(1, -1).to(a.device)
