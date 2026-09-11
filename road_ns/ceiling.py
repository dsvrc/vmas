#  I.5 -- the ceiling decomposition.  Compute this FIRST.
#
#  No policy, no training, no simulator.  NS-4.1: commit the output before any
#  method code exists; that is what makes it a prediction rather than a post-hoc
#  explanation.  If the coordination gap is small, this environment is a poor
#  showcase -- say so and pick another.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from road_ns.dial import DialParams, dial_g, driver_A, excess, loading, sensitivity
from road_ns.structure import RoadStructure

__all__ = ["Decomposition", "decompose", "fleet_scan"]


@dataclass(frozen=True)
class Decomposition:
    irreducible: float
    own: float
    peer: float
    decentralized_ceiling: float
    non_coordinating_ceiling: float
    u_mean: float
    u_p95: float
    g_mean: float
    n_controllable: int
    n_background: int

    @property
    def coordination_gap(self) -> float:
        return self.peer

    def row(self) -> str:
        return (
            f"{self.n_controllable:>4d}/{self.n_controllable + self.n_background:<4d} "
            f"{self.irreducible:7.1%} {self.own:7.1%} {self.peer:7.1%} "
            f"{self.decentralized_ceiling:8.1%} {self.u_mean:7.3f} {self.g_mean:6.3f}"
        )


def element_load(
    structure: RoadStructure,
    routes: Sequence[Sequence[int]],
    route_of: Sequence[int],
    weight: float = 1.0,
) -> Tensor:
    """``load_a`` = expected number of participants **occupying** element ``a``.

    One row, ``(1, A)``.

    A vehicle is on exactly one element at a time, so it contributes to element
    ``a`` the fraction of its journey spent there -- at constant speed, the
    element's share of its route length.  The unit therefore sums to 1 across
    each participant's route.

    Getting this wrong is not cosmetic: crediting a vehicle's whole unit to
    every element of its route against a SPATIAL capacity (how many vehicles
    fit) reported ``u_mean = 1.77``, i.e. a medium impossibly far over capacity,
    because the load was a flow and the denominator was an occupancy.  Load and
    capacity must be the same kind of quantity.

    **This is a time average, and the runtime loading is a realisation.**  At
    run time a vehicle is on exactly ONE element, so the load is an integer
    occupancy and ``u_i`` is the max over the route of that integer; here it is
    the expected occupancy.  Because the sensor takes a MAX (NS-1.1), the
    instantaneous value sits systematically above the average one -- measured
    ``u_mean`` 0.170 here against 0.395 in simulation at the same 16-of-40
    fleet.  Neither is wrong and they must not be reconciled by changing one:
    the decomposition is an expectation over the schedule, which is what makes
    it computable with no policy and no simulator, and that is the whole point
    of computing it first (NS-4.1).  Quote them as what they are.
    """
    load = torch.zeros(1, structure.n_elements)
    for r in route_of:
        elems = list(routes[r])
        if not elems:
            continue
        lens = structure.length[elems]
        share = lens / lens.sum().clamp_min(1e-12)
        for a, f in zip(elems, share):
            load[0, a] += weight * float(f)
    return load


def decompose(
    structure: RoadStructure,
    routes: Sequence[Sequence[int]],
    controllable: Sequence[int],
    background: Sequence[int],
    p: DialParams,
    driver_value: float = 1.0,
) -> Decomposition:
    """Partition the loading excess by **who can move it**.

    Excess is proportional to load, so a contributor class's share of the load
    is its share of the excess.  The attribution is therefore exact, and it is
    computed by re-running the loading function with one class of contributor at
    a time -- which is why this needs no policy and no training.

        Delta_fixed  background participants no agent controls
        Delta_own    the agent's own unit, amplified by 1/g
        Delta_peer   other controllable agents, through W, by 1/g
    """
    s = sensitivity(structure, p)
    a = torch.tensor([driver_value], dtype=torch.float32)
    g = dial_g(a, s, p)  # (1, A)
    g_nom = torch.ones_like(g)

    all_routes = list(controllable) + list(background)
    load_all = element_load(structure, routes, all_routes)
    load_fixed = element_load(structure, routes, background)

    route_elems = [routes[r] for r in controllable]
    u_all, binding = loading(load_all, g, structure, route_elems)
    u_nom, _ = loading(load_all, g_nom, structure, route_elems)
    g_bind = torch.gather(g.expand(u_all.shape[0], -1), 1, binding)

    total_excess = excess(u_all, g_bind)  # (1, N)

    # Shares of the BINDING element's load, by contributor class.
    #
    # The own term must be measured in the SAME units as the load: an agent
    # contributes to its binding element the fraction of its journey spent
    # there, not a whole unit.  Crediting it a full unit against an
    # occupancy-weighted load overstated ``own`` roughly fivefold and drove the
    # peer share to 0.0% -- which reads exactly like "this domain has no
    # coordination gap" and is instead an arithmetic error.
    cap_b = structure.capacity[binding]
    denom = load_all[0, binding].clamp_min(1e-12)
    share_fixed = load_fixed[0, binding] / denom
    own_at_binding = torch.zeros_like(share_fixed)
    for i, r in enumerate(controllable):
        elems = list(routes[r])
        if not elems:
            continue
        total_len = structure.length[elems].sum().clamp_min(1e-12)
        own_at_binding[:, i] = structure.length[binding[:, i]] / total_len
    share_own = own_at_binding / denom
    share_peer = (1.0 - share_fixed - share_own).clamp_min(0.0)

    d_fixed = (total_excess * share_fixed).sum()
    d_own = (total_excess * share_own).sum()
    d_peer = (total_excess * share_peer).sum()
    tot = (d_fixed + d_own + d_peer).clamp_min(1e-12)

    return Decomposition(
        irreducible=float(d_fixed / tot),
        own=float(d_own / tot),
        peer=float(d_peer / tot),
        decentralized_ceiling=float(1.0 - d_fixed / tot),
        non_coordinating_ceiling=float(1.0 - (d_fixed + d_peer) / tot),
        u_mean=float(u_all.mean()),
        u_p95=float(u_all.reshape(-1).quantile(0.95)),
        g_mean=float(g.mean()),
        n_controllable=len(controllable),
        n_background=len(background),
    )


def fleet_scan(
    structure: RoadStructure,
    routes: Sequence[Sequence[int]],
    p: DialParams,
    n_total: int = 20,
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 0,
    driver_value: float = 1.0,
) -> List[Decomposition]:
    """NS-4.2: the gap against controllable share.

    The prediction that it rises with the controllable fraction is one no
    competing credit-assignment method makes, and it costs no training to test:
    irreducible load disappears as the fleet share rises while peer load does
    not.
    """
    # road_traffic's own agent->loop assignment where it applies; a
    # deterministic fallback otherwise.  Never random: the fleet layout is
    # structure, and randomising it would make the decomposition a sample.
    from road_ns.structure import loop_assignment

    try:
        assign = loop_assignment(n_total)
    except ValueError:
        assign = [i % len(routes) for i in range(n_total)]
    out = []
    for f in fractions:
        k = max(1, int(round(f * n_total)))
        out.append(
            decompose(
                structure, routes, assign[:k], assign[k:], p, driver_value=driver_value
            )
        )
    return out
