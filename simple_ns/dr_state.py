#  The domain-randomisation distribution over sigma, as a SHARED, MUTABLE
#  object.
#
#  BASELINES.md B9's `ns_dr_enabled` draws sigma ~ U[low, high] once per
#  episode and never changes.  DORAEMON (X5) is the same arm with an ADAPTIVE
#  distribution: it solves a constrained optimisation problem between training
#  rounds and hands the environment a new Beta(a, b) each time.  So the
#  environment has to be able to READ a distribution the algorithm WRITES,
#  while a run is in progress.
#
#  That is what this module is: one process-wide registry, written by
#  `benchmarl/algorithms/doraemon.py` and read by
#  `simple_ns/layer.py::_draw_sigma`.  It works because a VMAS environment is
#  a vectorised torch object in THIS process -- there is no worker process and
#  no pickling between the algorithm and the scenario -- which is the same
#  reason the PACT debug CSV can be written from inside the scenario.
#
#  Nothing here is imported by anything unless `ns_dr_dist=beta`, and with the
#  default `uniform` the whole module is inert: `current()` returns None and
#  `_draw_sigma` takes exactly the branch it took before this file existed.
#
#  See `baselines/docs/doraemon.md`.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["BetaDr", "current", "publish", "clear", "n_updates"]


@dataclass(frozen=True)
class BetaDr:
    """``Y ~ Beta(a, b)`` rescaled onto ``[low, high]``.

    DORAEMON's ``DomainRandDistribution`` with ``dr_type='beta'``: it carries
    the two shape parameters and the fixed support, and the draw is
    ``y = x * (high - low) + low`` with ``x ~ Beta(a, b)`` -- verbatim
    ``DomainRandDistribution.sample``.
    """

    a: float
    b: float
    low: float
    high: float

    def __post_init__(self) -> None:
        if not (self.a > 0.0 and self.b > 0.0):
            raise ValueError(f"Beta needs a, b > 0; got a={self.a} b={self.b}")
        if not (self.low <= self.high):
            raise ValueError(f"needs low <= high; got [{self.low}, {self.high}]")

    def describe(self) -> str:
        return f"Beta(a={self.a:.4g}, b={self.b:.4g}) on [{self.low:g}, {self.high:g}]"


#: The live distribution, or None when no algorithm has published one.
_CURRENT: Optional[BetaDr] = None
_N_UPDATES: int = 0


def current() -> Optional[BetaDr]:
    """The distribution the environment should draw from, or ``None``."""
    return _CURRENT


def publish(distribution: BetaDr) -> None:
    """Install a new distribution.  Every subsequent episode reset uses it."""
    global _CURRENT, _N_UPDATES
    if not isinstance(distribution, BetaDr):
        raise TypeError(f"expected a BetaDr, got {type(distribution).__name__}")
    _CURRENT = distribution
    _N_UPDATES += 1


def clear() -> None:
    """Forget the published distribution.  Used by the tests and at teardown."""
    global _CURRENT, _N_UPDATES
    _CURRENT = None
    _N_UPDATES = 0


def n_updates() -> int:
    """How many times a distribution has been published in this process."""
    return _N_UPDATES
