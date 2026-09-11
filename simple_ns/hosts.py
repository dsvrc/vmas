#  The hosts.  Two lines each, because the layer needs only positions and a
#  force action and every VMAS scenario has both.
#
#  Nothing in the installed vmas is modified.  BenchMARL is handed a scenario
#  INSTANCE, so ``vmas.make_env``'s name lookup is never taken and the stock
#  ``vmas/balance`` and ``simple_ns/balance`` coexist in one process -- which is
#  what lets the sigma = 0 arm be checked against the stock scenario step for
#  step rather than argued about.
#
#  A note on the two files already in the vmas tree.  ``vmas/scenarios/
#  balance_ns.py`` (Force-Fight Coupling) and ``navigation_ns.py`` (Wake
#  Turbulence Coupling) are earlier attempts at the same idea and they are NOT
#  used here, for three reasons:
#
#    * their severity is a module-level constant (``FFC_SEVERITY = 5.0``), so the
#      dial cannot be read from the task configuration and NS-3.1 is violated --
#      there is no way to run a sigma = 0 arm from a config override;
#    * the channel is a multiplicative DERATE of the delivered force
#      (``force *= 1 - load**2``), which has no inverse, so the method could only
#      claim steering -- the same bounded cell ``road_ns`` is already in;
#    * they print debug output every 200 steps from inside the physics.
#
#  The mechanism they describe is right and the story survives; this layer keeps
#  it and moves the channel into the invertible cell.

from __future__ import annotations

from vmas.scenarios.balance import Scenario as BalanceScenario
from vmas.scenarios.navigation import Scenario as NavigationScenario
from vmas.scenarios.sampling import Scenario as SamplingScenario
from vmas.scenarios.transport import Scenario as TransportScenario

from simple_ns.layer import ExertionMixin, PactMixin

__all__ = ["HOSTS", "make_scenario"]


# -- balance: three supports under one rotatable beam carrying a package. The
#    beam is the shared medium and it transmits each support's push to the
#    others -- redundant actuators fighting through a rigid body.
class BalanceNs(ExertionMixin, BalanceScenario):
    pass


class BalancePact(PactMixin, BalanceScenario):
    pass


# -- transport: agents push a heavy package to a goal. Same medium, same story,
#    and the package mass makes the transmission strong.
class TransportNs(ExertionMixin, TransportScenario):
    pass


class TransportPact(PactMixin, TransportScenario):
    pass


# -- sampling: agents fly over a field and collect. The medium is the fluid, and
#    the transmission is rotor wake: a neighbour's thrust pushes you around.
class SamplingNs(ExertionMixin, SamplingScenario):
    pass


class SamplingPact(PactMixin, SamplingScenario):
    pass


# -- navigation: point-to-point with collision avoidance. Same wake story, and it
#    is the cheapest host, so it is the one to debug on.
class NavigationNs(ExertionMixin, NavigationScenario):
    pass


class NavigationPact(PactMixin, NavigationScenario):
    pass


HOSTS = {
    "balance": (BalanceNs, BalancePact),
    "transport": (TransportNs, TransportPact),
    "sampling": (SamplingNs, SamplingPact),
    "navigation": (NavigationNs, NavigationPact),
}


def make_scenario(pact: bool, host: str):
    """Build a scenario instance: stock host + dial (+ compensator)."""
    if host not in HOSTS:
        raise ValueError(f"unknown host {host!r}; expected one of {sorted(HOSTS)}")
    plain, with_pact = HOSTS[host]
    return with_pact() if pact else plain()
