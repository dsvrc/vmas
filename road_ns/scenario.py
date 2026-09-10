#  Build-order step 6: wire the layer into vmas/road_traffic.
#
#      Scenario (vmas.scenarios.road_traffic)
#        +-- SeverityScenario     <- I.4: EVERY arm gets this, unmodified
#              +-- PactScenario   <- adds the compensator only
#
#  I.4/NS-3.1: the dial is read from the task configuration and sits BELOW the
#  method in the hierarchy.  A dial only the method's arm experienced is
#  worthless as evidence.
#
#  Nothing in the installed vmas is modified.  BenchMARL is handed a scenario
#  INSTANCE, so ``vmas.make_env``'s name lookup is never taken and
#  ``vmas/road_traffic`` and ``road_ns/road_traffic`` coexist -- which is what
#  lets the smoke test diff them step for step.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from vmas.simulator.core import Agent, World
from vmas.scenarios.road_traffic import Scenario as RoadTrafficScenario

from pact1.core import Basis, PactParams, RLS, confidence, herd_index, steer, trust_from_logit
from road_ns.dial import (
    DialParams,
    dial_g,
    driver_A,
    harm,
    loading,
    loading_by_route,
    performance,
    sensitivity,
)
from road_ns.structure import PATH_TO_LOOP, load_structure

__all__ = ["SeverityScenario", "PactScenario", "make_scenario", "NS_KWARGS", "PACT_KWARGS"]


NS_KWARGS = (
    "ns_severity",
    "ns_period",
    "ns_wet_fraction",
    "ns_alpha",
    "ns_mean_preserve",
    "ns_observe_loading",
    "ns_route_set",
)

PACT_KWARGS = (
    "pact_enabled",
    "pact_trust",
    "pact_kappa",
    "pact_mu",
    "pact_p0",
    "pact_y_clip",
    "pact_warmup",
)


def _pop(kwargs: Dict[str, Any], keys) -> Dict[str, Any]:
    return {k: kwargs.pop(k) for k in keys if k in kwargs}


class SeverityScenario(RoadTrafficScenario):
    """``road_traffic`` under Coupling-Under-Drift.

    Overrides only ``make_world``, ``reset_world_at``, ``process_action``,
    ``post_step``, ``observation`` and ``info``.  ``reward`` and ``done`` are
    **inherited untouched** (NS-1.4): the agent is paid exactly what it was paid
    before, for a journey the medium made slower.  That is enforced by
    inheritance, not asserted in prose.
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

        world = super().make_world(batch_dim, device, **kwargs)

        self.struct = load_structure()
        self.routes = self.struct.declared_routes(self._route_set)
        self.sens = sensitivity(self.struct, self.ns).to(device)
        # Precomputed once.  Both of these were being rebuilt every step, and
        # the route lookup forced a device sync per step on top of that.
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
        self._route_of = torch.zeros(B, self.n_ag, device=device, dtype=torch.long)
        self._alive = torch.zeros(B, self.n_ag, device=device, dtype=torch.bool)

        # NS-3.3: count what the layer actually touched, and refuse to report a
        # severity arm if either is zero.  A silently inert disturbance is the
        # one failure mode indistinguishable from a clean null result.
        self._n_rewards_seen = 0
        self._n_records_harmed = 0
        self._n_rewards_untouched = 0

        self._on_built(world, device)
        print(self.struct.banner())
        print(
            f"severity layer  sigma={self.ns.severity} period={self.ns.period} "
            f"wet={self.ns.wet_fraction} alpha={self.ns.alpha} "
            f"routes={len(self.routes)} ({self._route_set}) N={self.n_ag}"
        )
        return world

    def _on_built(self, world: World, device: torch.device) -> None:
        """Hook for the layer above the dial."""
        return

    # ------------------------------------------------------------------
    #  lifecycle
    # ------------------------------------------------------------------

    def reset_world_at(self, env_index: Optional[int] = None, agent_index: Optional[int] = None):
        out = super().reset_world_at(env_index, agent_index)
        if env_index is None:
            self._u.zero_(); self._u_prev.zero_()
            self._harm.fill_(1.0); self._harm_prev.fill_(1.0)
            self._alive.zero_()
        else:
            self._u[env_index] = 0.0; self._u_prev[env_index] = 0.0
            self._harm[env_index] = 1.0; self._harm_prev[env_index] = 1.0
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

    def _current_routes(self) -> Tensor:
        """Each agent's route index, from road_traffic's own ``path_id``.

        ``path_to_loop`` maps a 1-based reference-path id to one of seven loops,
        and a rotation of a loop leaves its element SET unchanged -- so the forty
        distinct path ids collapse to seven routes as far as the operator is
        concerned.
        """
        pid = self.ref_paths_agent_related.path_id.to(torch.long)  # (B, N)
        return self._loop_table[pid.clamp(0, self._loop_table.numel() - 1)]

    def _begin_step(self) -> None:
        """Compute the medium's state for this step.

        Occupancy is read from where the vehicles ARE -- nearest element to each
        position -- so the loading is the medium's actual state rather than an
        expectation, and it depends on nothing inside road_traffic's
        reference-path bookkeeping.
        """
        agents = self.world.agents
        pos = torch.stack([a.state.pos for a in agents], dim=1)  # (B, N, 2)
        here = self.struct.locate(pos)  # (B, N)
        self._route_of = self._current_routes()

        B = pos.shape[0]
        load = torch.zeros(B, self.struct.n_elements, device=pos.device)
        load.scatter_add_(1, here, torch.ones_like(here, dtype=load.dtype))

        a = driver_A(self._step, self.ns)
        g = dial_g(a, self.sens, self.ns)  # (B, A)

        # PER WORLD.  VMAS draws path_id independently per parallel env, so
        # using env 0's routes for all of them computes the loading of a fleet
        # that does not exist -- and reads as a plausible number while doing it.
        u_der, binding = loading_by_route(
            load, g, self.struct, self._route_mask, self._route_of
        )
        u_nom, _ = loading_by_route(
            load, torch.ones_like(g), self.struct, self._route_mask, self._route_of
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
            self._n_records_harmed += int((self._harm[:, i] > 1.0).sum())
            self._n_rewards_untouched += int((self._harm[:, i] == 1.0).sum())
        self._n_rewards_seen += u.shape[0]

        super().process_action(agent)

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
                "ns_excess": (self._u[:, i : i + 1] * (1.0 - self._g_bind[:, i : i + 1])),
            }
        )
        return info

    # ------------------------------------------------------------------
    #  NS-3.3 -- fail loudly when the layer is inert
    # ------------------------------------------------------------------

    def severity_report(self) -> str:
        return (
            f"severity: rewards seen {self._n_rewards_seen}, records harmed "
            f"{self._n_records_harmed}, untouched {self._n_rewards_untouched}"
        )

    def assert_layer_fired(self) -> None:
        """Refuse to report this as a severity arm if the dial never reached
        anything.  A silently inert disturbance is indistinguishable from a
        clean null result -- in exactly the arm you most need to trust."""
        if self.ns.severity <= 0:
            return
        if self._n_records_harmed == 0:
            raise RuntimeError(
                f"ns_severity={self.ns.severity} but NOT ONE record was harmed "
                f"over {self._n_rewards_seen} agent-steps. The layer is not "
                "reaching the physics; this is a wiring bug, not a null result."
            )


class PactScenario(SeverityScenario):
    """PACT-1 on top of the dial.

    II.6, adapted: ``road_traffic`` assigns a reference path at reset and its
    action is continuous, so there is no discrete option set to rank.  The
    channel is a **differential pace shift** -- an agent whose route is
    predicted more congested than the fleet's average eases off, one on a clear
    route presses on.  A uniform shift accomplishes nothing and only a
    differential one helps, which is the commons in miniature.

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
        )
        self._trust_const = float(raw.get("pact_trust", 0.9))
        self._warmup = int(raw.get("pact_warmup", 200))

        names, cls = self.struct.element_classes()
        self.basis = Basis(
            self.struct.capacity.to(device), cls.to(device), len(names), self.routes,
            self.pact_params,
        )
        self.basis.prune(self.n_ag)
        self._ref = self.basis.geometric_reference(self.n_ag).to(device)
        self._scale = self.basis.scale_reference(self.n_ag).to(device)
        dim = 1 + len(self.basis.live)
        # One estimator per parallel world: each is an independent deployment.
        self.rls = RLS(self.n_ag, dim, self.pact_params, batch=world.batch_dim)
        self._dim = dim

        B = world.batch_dim
        # 1.0 is the identity for a multiplicative pace factor, so an
        # unconfigured or gated-off compensator leaves the command untouched.
        self._shift = torch.ones(B, self.n_ag, device=device)
        self._pred = torch.zeros(B, self.n_ag, device=device)
        self._trust = torch.zeros(B, self.n_ag, device=device)
        self._conf = torch.zeros(B, self.n_ag, device=device)
        self._herd = torch.zeros(world.batch_dim, device=device)

        # Gate 1: the vectorised basis must equal the brute-force definition.
        # Index order and self-exclusion are exactly the kind of wiring bug that
        # leaves every diagnostic looking healthy.
        self.basis.verify(list(range(min(self.n_ag, len(self.routes) * 2))))
        print(
            f"PACT            enabled={self.pact_enabled} trust={self._trust_const} "
            f"kappa={self.pact_params.kappa} mu={self.pact_params.mu} "
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
        psi = self.basis.design(self._route_of, self._ref, self._scale)

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
        self._herd = herd_index(self._route_of, len(self.routes))  # (B,), no sync

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


def make_scenario(pact: bool) -> SeverityScenario:
    """Build a scenario instance: road_traffic + dial (+ compensator)."""
    return PactScenario() if pact else SeverityScenario()
