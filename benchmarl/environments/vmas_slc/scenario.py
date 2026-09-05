#  The VMAS scenarios carrying Shared Link Contention, and PACT on top of them.
#
#  The class hierarchy is the one ``NS_FORM_SPEC`` B.5 mandates:
#
#      StockScenario                 vmas/sampling, vmas/discovery, ...
#        +-- SlcMixin                <- the dial.  EVERY arm runs this.
#              +-- PactMixin         <- the compensator only.
#
#  Severity is read from the *task* config, never from the method's block: a
#  dial only the method's arm experienced is worthless as evidence.
#
#  Nothing in the installed ``vmas`` package is modified.  ``VmasEnv`` is handed
#  a scenario *instance*, so ``vmas.make_env``'s ``scenarios.load(<name>.py)``
#  path is never taken -- upgrading vmas cannot silently revert the
#  non-stationarity, and ``vmas/sampling`` and ``vmas_slc/sampling`` coexist so
#  the smoke test can diff them step for step.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import torch
from torch import Tensor

from vmas.simulator.core import Agent, World
from vmas.simulator.scenario import BaseScenario

from benchmarl.environments.vmas_slc.pact_core import PactCompensator, PactParams
from benchmarl.environments.vmas_slc.slc_core import (
    apply_harm,
    build_operator,
    channel_load,
    dial_g,
    driver_A,
    effective_driver,
    exertion,
    harm_coefficient,
    loading,
    SHIFT_BUSY,
    SHIFT_QUIET,
    SlcParams,
)

__all__ = [
    "SlcMixin",
    "PactMixin",
    "SLC_KWARGS",
    "PACT_KWARGS",
    "make_slc_scenario",
    "STOCK_SCENARIOS",
]

#: Task-config keys the non-stationarity consumes.  These are TASK physics and
#: reach every arm.
SLC_KWARGS = (
    "slc_severity",
    "slc_driver_period",
    "slc_phase_spread",
    "slc_a_ref",
    "slc_quiet_scale",
    "slc_p_quiet",
    "slc_snr_ref",
    "slc_g_min_at_sigma1",
    "slc_mean_preserve",
    "slc_n_chan",
    "slc_aclr",
    "slc_leak_span",
    "slc_duty_lo",
    "slc_duty_hi",
    "slc_exposure_lo",
    "slc_exposure_hi",
    "slc_lfix_frac",
    "slc_u_nominal",
    "slc_capacity_mode",
    "slc_capacity_ref_agents",
    "slc_phi_floor",
    "slc_phi_slope",
    "slc_v_ref",
    "slc_phi_reads_executed",
    "slc_harm_at_nominal",
    "slc_harm_cap",
    "slc_harm_enabled",
    "slc_observe_loading",
)

#: Task-config keys the compensator consumes.  These never reach the dial.
PACT_KWARGS = (
    "pact_enabled",
    "pact_mode",
    "pact_gate",
    "pact_r",
    "pact_mu",
    "pact_p0",
    "pact_p_max_mult",
    "pact_max_trust",
    "pact_ff_gain",
    "pact_own_gain",
    "pact_fit_floor",
    "pact_ready_updates",
    "pact_fit_ema",
    "pact_warmup_updates",
    "pact_level_tau",
    "pact_max_delta",
    "pact_denom_floor",
    "pact_u_cap",
)


def _pop_params(kwargs: Dict[str, Any], keys) -> Dict[str, Any]:
    return {k[k.index("_") + 1 :]: kwargs.pop(k) for k in keys if k in kwargs}


def _as_action_vector(value, action_size: int, device) -> Tensor:
    """``u_range`` / ``u_multiplier`` may be a scalar or a per-component list."""
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(list(value), device=device, dtype=torch.float32)
    return torch.full((action_size,), float(value), device=device, dtype=torch.float32)


# =============================================================================
#  The dial
# =============================================================================


class SlcMixin:
    """Shared Link Contention over any holonomic VMAS scenario.

    Overrides only ``make_world``, ``reset_world_at``, ``process_action``,
    ``post_step``, ``observation`` and ``info``.  ``reward`` and ``done`` are
    **inherited untouched** -- the agent earns less strictly because it
    physically achieves less, never because the reward was reshaped.  That is
    enforced by inheritance rather than asserted in prose.
    """

    # -- construction -------------------------------------------------------

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        # Pop BOTH blocks before the stock scenario's check_kwargs_consumed
        # runs.  The method's keys are consumed here but only ever *read* by the
        # layer above, so the dial cannot accidentally depend on them.
        self.slc_params = SlcParams(**_pop_params(kwargs, SLC_KWARGS))
        self._pact_raw = _pop_params(kwargs, PACT_KWARGS)
        if not isinstance(self, PactMixin) and self._pact_raw.get("enabled", False):
            raise ValueError(
                "pact_enabled=True on a scenario built without the compensator "
                "layer.  The task class decides the class hierarchy from this "
                "same flag, so the two disagreeing means the env was constructed "
                "by hand -- use make_slc_scenario()."
            )
        world = super().make_world(batch_dim, device, **kwargs)

        agents: List[Agent] = list(world.agents)
        n = len(agents)
        for a in agents:
            if a.action_size != world.dim_p:
                raise ValueError(
                    f"SLC's harm channel is a gain on the holonomic force vector; "
                    f"agent {a.name} has action_size {a.action_size} != "
                    f"{world.dim_p}.  Wire a different channel before using a "
                    "non-holonomic scenario."
                )

        self.slc_op = build_operator(self.slc_params, n, device)
        self._slc_agent_index = {a.name: i for i, a in enumerate(agents)}

        B = batch_dim
        f = dict(device=device, dtype=torch.float32)
        self._slc_step = torch.zeros(B, device=device, dtype=torch.long)
        if self.slc_params.phase_spread:
            self._slc_phase = torch.arange(B, **f) / max(B, 1)
        else:
            self._slc_phase = torch.zeros(B, **f)
        self._slc_shift = torch.full((B,), SHIFT_BUSY, device=device, dtype=torch.long)

        self._slc_phi_prev = torch.full((B, n), self.slc_params.phi_floor, **f)
        self._slc_u_prev = torch.zeros(B, n, **f)
        self._slc_g_prev = torch.ones(B, n, **f)
        self._slc_alive = torch.zeros(B, n, device=device, dtype=torch.bool)

        self._slc_phi = self._slc_phi_prev.clone()
        self._slc_u = self._slc_u_prev.clone()
        self._slc_c = torch.zeros(B, n, **f)
        self._slc_g_agent = torch.ones(B, n, **f)
        self._slc_g = torch.ones(B, self.slc_op.n_chan, **f)
        self._slc_k_star = torch.zeros(B, n, device=device, dtype=torch.long)
        self._slc_A = torch.zeros(B, **f)
        self._slc_A_eff = torch.zeros(B, **f)
        self._slc_sat = torch.zeros(B, n, **f)

        # the action box, in physical units: _set_action clamps the raw action
        # to u_range and then multiplies by u_multiplier, and f_range is None in
        # these scenarios, so nothing downstream would catch an over-range
        # compensated command.  We must clamp it ourselves or PACT would be
        # allowed authority the baseline does not have.
        self._slc_u_bound = torch.stack(
            [
                _as_action_vector(a.u_range, a.action_size, device)
                * _as_action_vector(a.u_multiplier, a.action_size, device)
                for a in agents
            ],
            dim=0,
        )  # (N, action_size)

        # integrator constants, for the no-loop exertion source
        self._slc_dt = float(getattr(world, "dt", getattr(world, "_dt", 0.1)))
        world_drag = float(getattr(world, "drag", getattr(world, "_drag", 0.25)))
        self._slc_drag = torch.tensor(
            [world_drag if a.drag is None else float(a.drag) for a in agents],
            device=device,
            dtype=torch.float32,
        ).view(1, n, 1)
        self._slc_mass = torch.tensor(
            [float(a.mass) for a in agents], device=device, dtype=torch.float32
        ).view(1, n, 1)

        print(self.slc_op.banner())
        return world

    # -- lifecycle ----------------------------------------------------------

    def reset_world_at(self, env_index: Optional[int] = None) -> None:
        super().reset_world_at(env_index)
        p = self.slc_params
        if env_index is None:
            self._slc_phi_prev.fill_(p.phi_floor)
            self._slc_u_prev.zero_()
            self._slc_g_prev.fill_(1.0)
            self._slc_alive.zero_()
            draw = torch.rand(self._slc_shift.shape, device=self._slc_shift.device)
            self._slc_shift.copy_((draw < p.p_quiet).to(torch.long))
        else:
            self._slc_phi_prev[env_index] = p.phi_floor
            self._slc_u_prev[env_index] = 0.0
            self._slc_g_prev[env_index] = 1.0
            self._slc_alive[env_index] = False
            draw = torch.rand((), device=self._slc_shift.device)
            self._slc_shift[env_index] = int(draw < p.p_quiet)
        # NOTE: _slc_step is deliberately NOT reset.  The emitter runs off a
        # global clock; an episode boundary that rewound it would make the
        # driver endogenous to the agents' own failures.
        self._slc_on_reset(env_index)

    def _slc_on_reset(self, env_index: Optional[int]) -> None:
        """Hook for layers above the dial."""
        return

    def to(self, device: torch.device):
        super().to(device)
        self.slc_op = self.slc_op.to(device)
        self._slc_u_bound = self._slc_u_bound.to(device)
        return self

    # -- the medium ---------------------------------------------------------

    def _slc_begin_step(self) -> None:
        """Compute the medium's state for step ``t``.

        ``Phi(t)`` is read from the world state at the *top* of the step -- the
        result of step ``t-1``'s integration.  It is therefore already
        determined (no algebraic loop) yet genuinely unknown to any agent but
        its owner, which is exactly the gap the compensator must close.
        """
        p = self.slc_params
        phi = exertion(self._slc_exertion_source(), p)

        a = driver_A(self._slc_step, self._slc_phase, p.driver_period)
        a_eff = effective_driver(a, self._slc_shift, p)
        g = dial_g(a_eff, p, self.slc_op)

        load = channel_load(phi, self.slc_op)
        u, k_star = loading(load, g, self.slc_op)
        c = harm_coefficient(u, p)

        g_agent = (
            g.unsqueeze(1)
            .masked_fill(~self.slc_op.D.unsqueeze(0), float("inf"))
            .min(dim=-1)
            .values
        )

        self._slc_phi, self._slc_g, self._slc_u = phi, g, u
        self._slc_k_star, self._slc_c = k_star, c
        self._slc_A, self._slc_A_eff = a, a_eff
        self._slc_g_agent = g_agent
        self._slc_sat = torch.zeros_like(self._slc_sat)

    def _slc_exertion_source(self) -> Tensor:
        """The velocity ``Phi`` is read from -- ``(B, N, 2)``.

        A.6, the loop question, and it is decided here in one place.

        ``phi_reads_executed=True`` (default) reads the realised velocity, which
        is the result of the previous step's *compensated* command: pushing back
        is pushing, the medium is a commons, and T4 applies.

        ``False`` reads the velocity the agent's **intended** command would have
        produced -- one step of the same integrator, run on the pre-compensation
        action.  Compensation then never enters ``Phi``, so the loop is cut.
        Both branches live on the same velocity scale, so the contrast isolates
        the *loop* rather than a level shift in exertion.
        """
        agents = self.world.agents
        vel = torch.stack([a.state.vel for a in agents], dim=1)
        if self.slc_params.phi_reads_executed:
            return vel
        act = torch.stack([a.action.u for a in agents], dim=1)
        return vel * (1.0 - self._slc_drag) + act / self._slc_mass * self._slc_dt

    def _slc_command(self, agent: Agent, index: int) -> Tensor:
        """The command that reaches the medium.  The dial does not touch it;
        layers above (PACT) override this."""
        return agent.action.u

    def process_action(self, agent: Agent) -> None:
        if agent is self.world.agents[0]:
            self._slc_begin_step()
            self._slc_after_medium()
        i = self._slc_agent_index[agent.name]

        cmd = self._slc_command(agent, i)
        bound = self._slc_u_bound[i].view(1, -1)
        clamped = cmd.clamp(-bound, bound)
        self._slc_sat[:, i] = (cmd != clamped).any(-1).to(self._slc_sat.dtype)

        agent.action.u = apply_harm(clamped, self._slc_c[:, i])
        super().process_action(agent)

    def _slc_after_medium(self) -> None:
        """Hook that runs once per step, after the medium's state is known and
        before any agent's action is processed."""
        return

    def post_step(self) -> None:
        super().post_step()
        self._slc_phi_prev = self._slc_phi
        self._slc_u_prev = self._slc_u
        self._slc_g_prev = self._slc_g_agent
        self._slc_alive.fill_(True)
        self._slc_step = self._slc_step + 1

    # -- sensor and read-out ------------------------------------------------

    def observation(self, agent: Agent):
        obs = super().observation(agent)
        if not self.slc_params.observe_loading:
            return obs
        i = self._slc_agent_index[agent.name]
        u = self._slc_u[:, i : i + 1]
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["link_loading"] = u
            return obs
        return torch.cat([obs, u], dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = dict(super().info(agent))
        i = self._slc_agent_index[agent.name]
        info.update(
            {
                "slc_u": self._slc_u[:, i : i + 1],
                "slc_c": self._slc_c[:, i : i + 1],
                "slc_phi": self._slc_phi[:, i : i + 1],
                "slc_g": self._slc_g_agent[:, i : i + 1],
                "slc_A": self._slc_A.unsqueeze(-1),
                "slc_A_eff": self._slc_A_eff.unsqueeze(-1),
                "slc_quiet": (self._slc_shift == SHIFT_QUIET)
                .to(torch.float32)
                .unsqueeze(-1),
                "slc_sat": self._slc_sat[:, i : i + 1],
                "slc_excess": (
                    self._slc_u[:, i : i + 1] * (1.0 - self._slc_g_agent[:, i : i + 1])
                ),
            }
        )
        return info


# =============================================================================
#  The compensator
# =============================================================================


class PactMixin(SlcMixin):
    """PACT, sitting strictly above the dial.

    The host RL algorithm is untouched: no loss terms, no critic changes, no
    extra action dimensions.  The compensator reads the agent's own stale
    loading and the fleet's one-step-delayed exertion broadcast, and rescales
    the commanded force.  Because the correction is a pure gain along the
    commanded direction it never changes the agent's heading.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        world = super().make_world(batch_dim, device, **kwargs)
        self.pact_params = PactParams(**self._pact_raw)
        self.pact = PactCompensator(
            self.pact_params,
            self.slc_params,
            self.slc_op,
            batch_dim,
            device,
        )
        self._pact_delta = torch.zeros(
            batch_dim, self.slc_op.n_agents, world.dim_p, device=device
        )
        self._pact_clip = torch.zeros(
            batch_dim, self.slc_op.n_agents, device=device
        )
        print(self.pact.banner())
        return world

    def _slc_on_reset(self, env_index: Optional[int]) -> None:
        if getattr(self, "pact", None) is not None:
            self.pact.reset(env_index)
            if env_index is None:
                self._pact_delta.zero_()
                self._pact_clip.zero_()
            else:
                self._pact_delta[env_index] = 0.0
                self._pact_clip[env_index] = 0.0

    def to(self, device: torch.device):
        super().to(device)
        self.pact = self.pact.to(device)
        return self

    def _slc_after_medium(self) -> None:
        if not self.pact_params.enabled:
            self._pact_delta.zero_()
            self._pact_clip.zero_()
            return

        # The agent's OWN current exertion: it knows its own velocity and its
        # own action exactly.  Peers get only the one-step-delayed broadcast.
        # Reading self._slc_phi here would be an oracle -- do not.
        phi_own_now = exertion(self._slc_exertion_source(), self.slc_params)

        out = self.pact.step(
            u_prev=self._slc_u_prev,
            phi_bcast=self._slc_phi_prev,
            phi_own_now=phi_own_now,
            g_prev=self._slc_g_prev,
            g_now=self._slc_g_agent,
            alive=self._slc_alive,
        )
        actions = torch.stack([a.action.u for a in self.world.agents], dim=1)
        delta, clipped = self.pact.compensate(actions, out["c_hat"])
        self._pact_delta = delta
        self._pact_clip = clipped.to(self._pact_clip.dtype)

    def _slc_command(self, agent: Agent, index: int) -> Tensor:
        if not self.pact_params.enabled:
            return agent.action.u
        return agent.action.u + self._pact_delta[:, index]

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = super().info(agent)
        if not self.pact_params.enabled or not self.pact.diag:
            return info
        i = self._slc_agent_index[agent.name]
        d = self.pact.diag
        one = lambda t: t[:, i : i + 1]  # noqa: E731
        info.update(
            {
                "pact_applied_trust": one(d["applied_trust"]),
                "pact_fit_gain": one(d["fit_gain"]),
                "pact_base_abs": one(d["base_abs"]),
                "pact_ff_abs": one(d["ff_abs"]),
                "pact_own_abs": one(d["own_abs"]),
                "pact_peer_abs": one(d["peer_abs"]),
                "pact_u_hat": one(d["u_hat"]),
                "pact_u_err": one(d["u_hat"]) - self._slc_u[:, i : i + 1],
                "pact_trP": one(d["trP"]),
                "pact_clamp": one(d["clamp_frac"]),
                "pact_state": one(d["state"]),
                "pact_n_updates": one(d["n_updates"]),
                "pact_own_gain_coef": one(d["own_gain_coef"]),
                "pact_delta_abs": self._pact_delta[:, i].abs().mean(-1, keepdim=True),
                "pact_delta_clip": self._pact_clip[:, i : i + 1],
            }
        )
        return info


# =============================================================================
#  Concrete scenarios
# =============================================================================


def _stock(name: str) -> Type[BaseScenario]:
    import importlib

    module = importlib.import_module(f"vmas.scenarios.{name}")
    return module.Scenario


#: The tasks SLC ships on.  Chosen because ``Phi`` cannot be driven to its floor
#: without forfeiting reward (A.5) and because ``n_agents`` is free, which is
#: what makes the C.4 N-scaling prediction testable.
STOCK_SCENARIOS = ("sampling", "discovery", "navigation")


def make_slc_scenario(stock_name: str, pact: bool) -> BaseScenario:
    """Build a scenario instance: stock task + dial (+ compensator)."""
    if stock_name not in STOCK_SCENARIOS:
        raise ValueError(
            f"unknown SLC base scenario {stock_name!r}; expected one of "
            f"{STOCK_SCENARIOS}"
        )
    base = _stock(stock_name)
    mixin = PactMixin if pact else SlcMixin
    cls = type(
        f"{'Pact' if pact else 'Slc'}{base.__name__}_{stock_name}",
        (mixin, base),
        {},
    )
    return cls()
