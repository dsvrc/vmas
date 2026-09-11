#  Build-order step 6: wire the layer into a VMAS host.
#
#      BaseScenario
#        +-- <host>                 vmas/road_traffic  OR  road_ns/flow
#              +-- SeverityMixin    <- I.4: EVERY arm gets this, unmodified
#                    +-- PactMixin  <- adds the compensator only
#
#  I.4/NS-3.1: the dial is read from the task configuration and sits BELOW the
#  method in the hierarchy.  A dial only the method's arm experienced is
#  worthless as evidence.
#
#  ---------------------------------------------------------------------------
#  Why the layer is a MIXIN and not a subclass of road_traffic
#  ---------------------------------------------------------------------------
#  Two hosts now carry the same medium:
#
#    ``road_ns/road_traffic``   SigmaRL's scenario.  Faithful, and 173 ms/frame
#                               at N=40 -- 58 hours for one arm of one seed.
#                               Kept as the provenance row, run at low N.
#    ``road_ns/lanelet_flow``   the same map, capacities, routes, dial and harm
#                               channel on an affordable vehicle model.
#                               0.24 ms/frame at N=16.  See road_ns/flow.py.
#
#  Everything between the map and the method is identical across the two, which
#  is the only reason the comparison between them means anything.  Writing the
#  dial twice would have guaranteed it drifted.
#
#  A host must supply exactly two things:
#
#      _ns_route_of()    -> (B, N) long, each agent's route index
#      _ns_element_of()  -> (B, N) long, the element each agent occupies
#
#  Nothing in the installed vmas is modified.  BenchMARL is handed a scenario
#  INSTANCE, so ``vmas.make_env``'s name lookup is never taken and
#  ``vmas/road_traffic`` and ``road_ns/*`` coexist.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from vmas.simulator.core import Agent, World
from vmas.scenarios.road_traffic import Scenario as RoadTrafficScenario

from pact1.core import Basis, PactParams, RLS, confidence, herd_index, steer
from road_ns.dial import (
    DialParams,
    dial_g,
    driver_A,
    harm,
    loading_by_route,
)
from road_ns.flow import LaneletFlow
from road_ns.structure import PATH_TO_LOOP, load_structure
from road_ns.dial import sensitivity

__all__ = [
    "SeverityMixin",
    "PactMixin",
    "SeverityScenario",
    "PactScenario",
    "FlowSeverityScenario",
    "FlowPactScenario",
    "make_scenario",
    "NS_KWARGS",
    "PACT_KWARGS",
    "HOSTS",
]


NS_KWARGS = (
    "ns_severity",
    "ns_period",
    "ns_wet_fraction",
    "ns_alpha",
    "ns_mean_preserve",
    "ns_observe_loading",
    "ns_route_set",
    "ns_exclude_self",
)

PACT_KWARGS = (
    "pact_enabled",
    "pact_trust",
    "pact_kappa",
    "pact_mu",
    "pact_p0",
    "pact_y_clip",
    "pact_warmup",
    "pact_shift_mode",
    "pact_shift_clip",
)


def _pop(kwargs: Dict[str, Any], keys) -> Dict[str, Any]:
    return {k: kwargs.pop(k) for k in keys if k in kwargs}


class SeverityMixin:
    """Coupling-Under-Drift, on top of whatever host it is mixed into.

    Overrides only ``make_world``, ``reset_world_at``, ``process_action``,
    ``post_step``, ``observation`` and ``info``.  ``reward`` and ``done`` are
    **inherited untouched** (NS-1.4): the agent is paid exactly what it was paid
    before, for a journey the medium made slower.  That is enforced by
    inheritance, not asserted in prose.

    The host must supply ``_ns_route_of()`` and ``_ns_element_of()``, each
    returning ``(B, N)``.  They are deliberately NOT declared here as abstract
    stubs: this mixin sits first in the MRO, so a stub here would shadow the
    host adapter's real implementation and raise on the first step.
    """

    # ------------------------------------------------------------------
    #  construction
    # ------------------------------------------------------------------

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        raw_ns = _pop(kwargs, NS_KWARGS)
        self._pact_raw = _pop(kwargs, PACT_KWARGS)

        self.ns = DialParams(
            severity=float(raw_ns.get("ns_severity", 1.0)),
            period=int(raw_ns.get("ns_period", 100)),
            wet_fraction=float(raw_ns.get("ns_wet_fraction", 0.5)),
            alpha=float(raw_ns.get("ns_alpha", 2.28)),
            mean_preserve=bool(raw_ns.get("ns_mean_preserve", False)),
        )
        self._observe_loading = bool(raw_ns.get("ns_observe_loading", True))
        self._route_set = str(raw_ns.get("ns_route_set", "loops"))
        # I.2.  Default ON -- see dial.loading_by_route for what leaving the
        # agent's own vehicle in the load costs (a lone agent reading harm 1.034
        # at sigma=1, i.e. slowing itself down).
        self._exclude_self = bool(raw_ns.get("ns_exclude_self", True))

        # Built BEFORE the host's make_world: a host that follows the centre
        # line needs the route geometry while it is building its world.  Moved
        # onto the training device here, once -- leaving `capacity` on the CPU
        # is what made every severity arm die on the first CUDA step.
        self.struct = load_structure().to(device)
        self.routes = self.struct.declared_routes(self._route_set)

        world = super().make_world(batch_dim, device, **kwargs)

        self.sens = sensitivity(self.struct, self.ns).to(device)
        self._route_mask = self.struct.route_mask(self.routes).to(device)
        self._loop_table = torch.tensor(
            [x - 1 for x in PATH_TO_LOOP], device=device, dtype=torch.long
        )

        agents = list(world.agents)
        self.n_ag = len(agents)
        self._agent_index = {a.name: i for i, a in enumerate(agents)}
        for a in agents:
            if a.action_size < 1:
                raise ValueError(f"{a.name} has no longitudinal action to derate")

        B = batch_dim
        f = dict(device=device, dtype=torch.float32)
        self._step = torch.zeros(B, device=device, dtype=torch.long)
        self._u = torch.zeros(B, self.n_ag, **f)
        self._u_prev = torch.zeros(B, self.n_ag, **f)
        self._harm = torch.ones(B, self.n_ag, **f)
        self._harm_prev = torch.ones(B, self.n_ag, **f)
        self._g_bind = torch.ones(B, self.n_ag, **f)
        self._A = torch.zeros(B, **f)
        self._ns_route = torch.zeros(B, self.n_ag, device=device, dtype=torch.long)
        self._alive = torch.zeros(B, self.n_ag, device=device, dtype=torch.bool)

        # NS-3.3: count what the layer actually touched, and refuse to report a
        # severity arm if either is zero.  A silently inert disturbance is the
        # one failure mode indistinguishable from a clean null result.
        #
        # Accumulated as TENSORS.  These used to be `int(...)` per agent per
        # step, which is a device sync per agent per step -- 80 of them at N=40,
        # to maintain a counter nobody reads until the run ends.
        self._n_seen = torch.zeros((), device=device, dtype=torch.long)
        self._n_harmed = torch.zeros((), device=device, dtype=torch.long)
        self._n_untouched = torch.zeros((), device=device, dtype=torch.long)

        self._on_built(world, device)
        print(self.struct.banner())
        print(
            f"severity layer  sigma={self.ns.severity} period={self.ns.period} "
            f"wet={self.ns.wet_fraction} alpha={self.ns.alpha} "
            f"routes={len(self.routes)} ({self._route_set}) N={self.n_ag} "
            f"exclude_self={self._exclude_self}"
        )
        return world

    def _on_built(self, world: World, device: torch.device) -> None:
        """Hook for the layer above the dial."""
        return

    # ------------------------------------------------------------------
    #  lifecycle
    # ------------------------------------------------------------------

    def reset_world_at(
        self, env_index: Optional[int] = None, agent_index: Optional[int] = None
    ):
        out = super().reset_world_at(env_index, agent_index)
        if env_index is None:
            self._u.zero_()
            self._u_prev.zero_()
            self._harm.fill_(1.0)
            self._harm_prev.fill_(1.0)
            self._alive.zero_()
        else:
            self._u[env_index] = 0.0
            self._u_prev[env_index] = 0.0
            self._harm[env_index] = 1.0
            self._harm_prev[env_index] = 1.0
            self._alive[env_index] = False
        # NS-3.4: the driver's clock is NOT reset.  Weather does not restart
        # because a training episode ended.
        self._on_reset(env_index)
        return out

    def _on_reset(self, env_index: Optional[int]) -> None:
        return

    # ------------------------------------------------------------------
    #  the medium
    # ------------------------------------------------------------------

    def _begin_step(self) -> None:
        """Compute the medium's state for this step.

        Occupancy is read from where the vehicles ARE, so the loading is the
        medium's actual state rather than an expectation.
        """
        here = self._ns_element_of()  # (B, N)
        self._ns_route = self._ns_route_of()  # (B, N)

        B = here.shape[0]
        load = torch.zeros(B, self.struct.n_elements, device=here.device)
        load.scatter_add_(1, here, torch.ones_like(here, dtype=load.dtype))

        # I.5's Delta_fixed: demand no agent controls.  Hosts that have none
        # simply do not define the hook, and the decomposition's irreducible
        # share is then 0 by construction -- which is a fact about the fleet,
        # not about the medium, and must be reported as one.
        bg = getattr(self, "_ns_background_load", None)
        if bg is not None:
            extra = bg(self.struct.n_elements)
            if extra is not None:
                load = load + extra

        a = driver_A(self._step, self.ns)
        g = dial_g(a, self.sens, self.ns)  # (B, A)

        # PER WORLD.  VMAS draws the fleet layout independently per parallel
        # env, so using env 0's routes for all of them computes the loading of a
        # fleet that does not exist -- and reads as a plausible number.
        #
        # `self_element` is the I.2 fix: u is PEER loading.
        se = here if self._exclude_self else None
        # Built ONCE: the derated and nominal loadings run over the same routes,
        # and this (B, N, A) gather is the dominant cost of the medium.
        mask = self._route_mask[self._ns_route]
        u_der, binding = loading_by_route(
            load, g, self.struct, self._route_mask, self._ns_route, se, mask
        )
        u_nom, _ = loading_by_route(
            load,
            torch.ones_like(g),
            self.struct,
            self._route_mask,
            self._ns_route,
            se,
            mask,
        )

        self._A = a
        self._u = u_der
        self._g_bind = torch.gather(g, 1, binding)
        self._harm = harm(u_der, u_nom, self.ns)

    # ------------------------------------------------------------------
    #  the harm channel
    # ------------------------------------------------------------------

    def _command(self, agent: Agent, index: int) -> Tensor:
        """The command that reaches the medium.  The dial never touches it;
        the layer above (PACT) overrides this."""
        return agent.action.u

    def process_action(self, agent: Agent) -> None:
        # The host first: a centre-line host localises the whole fleet on agent
        # 0 and the medium reads the result.
        super().process_action(agent)
        if agent is self.world.agents[0]:
            self._begin_step()
            self._after_medium()

        i = self._agent_index[agent.name]
        u = self._command(agent, i).clone()

        # NS-1.4: divide the achievable speed.  u[:, 0] is a direct VELOCITY
        # command for KinematicBicycle, so a congested, rain-derated element
        # simply caps how fast the vehicle can go.  The reward function is never
        # touched.  At sigma=0, harm == 1.0 exactly and this is v / 1.0 -- bit
        # for bit the stock command.
        u[:, 0] = u[:, 0] / self._harm[:, i].clamp_min(1e-6)
        agent.action.u = u

        if self.ns.severity > 0:
            self._n_harmed += (self._harm[:, i] > 1.0).sum()
            self._n_untouched += (self._harm[:, i] == 1.0).sum()
        self._n_seen += u.shape[0]

    def _after_medium(self) -> None:
        return

    def post_step(self) -> None:
        super().post_step()
        self._u_prev = self._u
        self._harm_prev = self._harm
        self._alive.fill_(True)
        self._step = self._step + 1

    # ------------------------------------------------------------------
    #  sensor and read-out
    # ------------------------------------------------------------------

    def observation(self, agent: Agent):
        obs = super().observation(agent)
        if not self._observe_loading:
            return obs
        i = self._agent_index[agent.name]
        # II.2: the agent's own relative excess over its own nominal, ONE STEP
        # STALE.  Proprioception, not privilege: every driver knows how long its
        # own journey took and what it would have taken on an empty road.
        y = (self._harm_prev[:, i : i + 1] - 1.0).clamp(-1.0, 10.0)
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["relative_excess"] = y
            return obs
        return torch.cat([obs, y], dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = dict(super().info(agent))
        i = self._agent_index[agent.name]
        info.update(
            {
                "ns_u": self._u[:, i : i + 1],
                "ns_harm": self._harm[:, i : i + 1],
                "ns_g": self._g_bind[:, i : i + 1],
                "ns_A": self._A.unsqueeze(-1),
                "ns_excess": (
                    self._u[:, i : i + 1] * (1.0 - self._g_bind[:, i : i + 1])
                ),
            }
        )
        return info

    # ------------------------------------------------------------------
    #  NS-3.3 -- fail loudly when the layer is inert
    # ------------------------------------------------------------------

    def severity_report(self) -> str:
        return (
            f"severity: rewards seen {int(self._n_seen)}, records harmed "
            f"{int(self._n_harmed)}, untouched {int(self._n_untouched)}"
        )

    def assert_layer_fired(self) -> None:
        """Refuse to report this as a severity arm if the dial never reached
        anything.  A silently inert disturbance is indistinguishable from a
        clean null result -- in exactly the arm you most need to trust."""
        if self.ns.severity <= 0:
            return
        if int(self._n_harmed) == 0:
            raise RuntimeError(
                f"ns_severity={self.ns.severity} but NOT ONE record was harmed "
                f"over {int(self._n_seen)} agent-steps. The layer is not "
                "reaching the physics; this is a wiring bug, not a null result."
            )


class PactMixin(SeverityMixin):
    """PACT-1 on top of the dial.

    II.6, adapted: this host's action is continuous and the route is fixed
    between trips, so there is no discrete option set to rank.  The channel is a
    **differential pace shift** -- an agent whose route is predicted more
    congested than the fleet's average eases off, one on a clear route presses
    on.  A uniform shift accomplishes nothing and only a differential one helps,
    which is the commons in miniature.

    There is no inverse (you cannot subtract seconds off a congested lanelet),
    so this instance claims **identification and steering only**.
    """

    def _on_built(self, world: World, device: torch.device) -> None:
        raw = self._pact_raw
        self.pact_enabled = bool(raw.get("pact_enabled", False))
        self.pact_params = PactParams(
            mu=float(raw.get("pact_mu", 0.999)),
            p0=float(raw.get("pact_p0", 10.0)),
            kappa=float(raw.get("pact_kappa", 1.0)),
            y_clip=float(raw.get("pact_y_clip", 10.0)),
            shift_mode=str(raw.get("pact_shift_mode", "centred")),
            shift_clip=float(raw.get("pact_shift_clip", 0.5)),
        )
        self._trust_const = float(raw.get("pact_trust", 0.9))
        self._warmup = int(raw.get("pact_warmup", 200))

        names, cls = self.struct.element_classes()
        self.basis = Basis(
            self.struct.capacity.to(device),
            cls.to(device),
            len(names),
            self.routes,
            self.pact_params,
        )
        self.basis.prune(self.n_ag)
        self._ref = self.basis.geometric_reference(self.n_ag).to(device)
        self._scale = self.basis.scale_reference(self.n_ag).to(device)
        dim = 1 + len(self.basis.live)
        # One estimator per parallel world: each is an independent deployment.
        self.rls = RLS(
            self.n_ag, dim, self.pact_params, batch=world.batch_dim, device=device
        )
        self._dim = dim

        B = world.batch_dim
        # 1.0 is the identity for a multiplicative pace factor, so an
        # unconfigured or gated-off compensator leaves the command untouched.
        self._shift = torch.ones(B, self.n_ag, device=device)
        self._pred = torch.zeros(B, self.n_ag, device=device)
        self._trust = torch.zeros(B, self.n_ag, device=device)
        self._conf = torch.zeros(B, self.n_ag, device=device)
        self._herd = torch.zeros(B, device=device)

        # Gate 1: the vectorised basis must equal the brute-force definition.
        # Index order and self-exclusion are exactly the kind of wiring bug that
        # leaves every diagnostic looking healthy.
        #
        # The probe fleet is route INDICES, so it must wrap at len(routes).  It
        # used to be `range(min(n_ag, 2 * len(routes)))`, which indexes route 13
        # of 7 for any fleet of 7 or more -- i.e. gate 1 aborted every PACT arm
        # with an IndexError before a single step was taken.
        n_probe = min(max(self.n_ag, 2), 2 * len(self.routes))
        self.basis.verify([i % len(self.routes) for i in range(n_probe)])
        print(
            f"PACT            enabled={self.pact_enabled} trust={self._trust_const} "
            f"kappa={self.pact_params.kappa} mu={self.pact_params.mu} "
            f"shift={self.pact_params.shift_mode}/"
            f"{self.pact_params.shift_clip} "
            f"r_live={len(self.basis.live)}/{len(names)} warmup={self._warmup}"
        )

    def _on_reset(self, env_index: Optional[int]) -> None:
        if not hasattr(self, "_shift"):
            return
        if env_index is None:
            self._shift.fill_(1.0)
        else:
            self._shift[env_index] = 1.0

    def _after_medium(self) -> None:
        if not self.pact_enabled:
            self._shift.fill_(1.0)
            return

        # (B, N, dim) -- per world, because each world has its own fleet layout.
        psi = self.basis.design(self._ns_route, self._ref, self._scale)

        # II.2: the target is the agent's own relative excess, one step stale.
        y = (self._harm_prev - 1.0).clamp(-1.0, self.pact_params.y_clip)
        if bool(self._alive.any()):
            self.rls.update(psi, y)

        self._pred = self.rls.predict(psi)  # (B, N)
        self._conf = confidence(psi, self.rls.P, self.pact_params, self._dim)

        # P-5.1's inverted prior lives in ``pact_trust``: near full reliance, not
        # half.  P-5.2 gates it on PREDICTION uncertainty, never on tr(P).
        ready = self.rls.n_updates.min(dim=-1).values >= self._warmup  # (B,)
        self._trust = (
            torch.where(ready, self._trust_const, 0.0).unsqueeze(-1) * self._conf
        )

        # a multiplicative factor on the pace command; 1.0 == untouched
        self._shift = steer(
            torch.ones_like(self._pred), self._pred, self._trust, self.pact_params
        )
        self._herd = herd_index(self._ns_route, len(self.routes))  # (B,), no sync

    def _command(self, agent: Agent, index: int) -> Tensor:
        if not self.pact_enabled:
            return agent.action.u
        u = agent.action.u.clone()
        u[:, 0] = u[:, 0] * self._shift[:, index]
        return u

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = super().info(agent)
        if not self.pact_enabled:
            return info
        i = self._agent_index[agent.name]
        one = lambda t: t[:, i : i + 1]  # noqa: E731
        info.update(
            {
                "pact_trust": one(self._trust),
                "pact_conf": one(self._conf),
                "pact_pred": one(self._pred),
                "pact_shift": one(self._shift),
                "pact_updates": self.rls.n_updates[:, i : i + 1],
                "pact_skipped": self.rls.n_skipped[:, i : i + 1],
                "pact_herd": self._herd.reshape(-1, 1).expand_as(self._u[:, i : i + 1]),
            }
        )
        return info


# ===========================================================================
#  host adapters -- the whole of the contract, per host
# ===========================================================================


class _RoadTrafficHost:
    """``vmas/road_traffic``.  The provenance host: faithful, and expensive."""

    def _ns_route_of(self) -> Tensor:
        """Each agent's route index, from road_traffic's own ``path_id``.

        ``path_to_loop`` maps a 1-based reference-path id to one of seven loops,
        and a rotation of a loop leaves its element SET unchanged -- so the forty
        distinct path ids collapse to seven routes as far as the operator is
        concerned.
        """
        pid = self.ref_paths_agent_related.path_id.to(torch.long)  # (B, N)
        return self._loop_table[pid.clamp(0, self._loop_table.numel() - 1)]

    def _ns_element_of(self) -> Tensor:
        """Nearest element to each vehicle.

        Read from where the vehicles ARE rather than from road_traffic's
        reference-path bookkeeping, so a change there cannot silently repoint
        the medium.
        """
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        return self.struct.locate(pos)


class _FlowHost:
    """``road_ns/lanelet_flow``.  Both answers are O(1) gathers it already has."""

    def _ns_route_of(self) -> Tensor:
        return self._route_of

    def _ns_element_of(self) -> Tensor:
        return self._elem_of


class SeverityScenario(SeverityMixin, _RoadTrafficHost, RoadTrafficScenario):
    pass


class PactScenario(PactMixin, _RoadTrafficHost, RoadTrafficScenario):
    pass


class FlowSeverityScenario(SeverityMixin, _FlowHost, LaneletFlow):
    pass


class FlowPactScenario(PactMixin, _FlowHost, LaneletFlow):
    pass


HOSTS = {
    "road_traffic": (SeverityScenario, PactScenario),
    "lanelet_flow": (FlowSeverityScenario, FlowPactScenario),
}


def make_scenario(pact: bool, host: str = "lanelet_flow") -> SeverityMixin:
    """Build a scenario instance: host + dial (+ compensator)."""
    if host not in HOSTS:
        raise ValueError(f"unknown host {host!r}; expected one of {sorted(HOSTS)}")
    plain, with_pact = HOSTS[host]
    return with_pact() if pact else plain()
