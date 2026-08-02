#  Navigation-PCW: VMAS ``navigation`` under a category-C non-stationarity.
#
#  ---------------------------------------------------------------------------
#  The story
#  ---------------------------------------------------------------------------
#  N holonomic vehicles fly/swim in a **confined arena** (an indoor flight cage,
#  a test tank).  Their thrusters do not push against nothing: every command
#  feeds momentum into the enclosed medium, and because the volume is closed
#  that momentum cannot escape -- it accumulates as a slowly-decaying **bulk
#  circulation** of the whole arena.  A vehicle sitting in that circulation is
#  dragged around its yaw axis until its passive stabiliser balances the drag,
#  so it ends up flying at a heading offset it never commanded.  Thrust is
#  produced in the **body frame**, so a yawed vehicle pushes in the wrong
#  direction.  Indoors / underwater there is no usable magnetometer, so the
#  offset is unobservable to the vehicle: pose comes from external motion
#  capture (world frame) but the actuator does not.
#
#  How hard a given circulation drags on a hull is set by the **ambient
#  density**, which cycles over the operating day.  That thermal cycle is the
#  exogenous driver ``A(t)``.
#
#  ---------------------------------------------------------------------------
#  Why this is category C and not A or B
#  ---------------------------------------------------------------------------
#  * The driver **multiplies** the sum over the other agents and is never an
#    additive term of its own.  With ``N = 1`` the sum is empty, the circulation
#    is identically zero for all t, and the environment is byte-identical to
#    stationary VMAS navigation *however large the driver grows*.  -> not (B).
#  * The driver runs off a global clock that is never reset, so difficulty keeps
#    drifting even with teammates frozen at an optimal policy.  -> not (A).
#  * No agent appears in its own exertion sum, so no agent can influence its own
#    liability -- individually exogenous, collectively endogenous.
#
#  ---------------------------------------------------------------------------
#  Dynamics-only, never reward shaping
#  ---------------------------------------------------------------------------
#  This scenario inherits ``reward``, ``observation`` and ``done`` from VMAS
#  ``navigation`` **without overriding them**.  The reward function is the
#  original one byte for byte; the agent earns less strictly because it
#  physically achieves less.  ``observation`` being inherited is likewise the
#  proof that the agent is blind to the disturbance by construction.

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from vmas.scenarios.navigation import Scenario as NavigationScenario

from benchmarl.environments.vmas_ns.pcw_core import (
    angular_impulse,
    driver_A,
    leak_step,
    PcwParams,
    peer_mean,
    rotate,
    wrap_angle,
)

#: Names of the extra ``info`` entries this scenario publishes.  All are
#: privileged (they are *not* in the observation) and all are in native units.
PCW_INFO_KEYS = (
    "ns_theta_applied",
    "ns_theta_next",
    "ns_x2",
    "ns_c",
    "ns_c_next",
    "ns_A",
    "ns_msg",
    "ns_abs_theta",
)


class Scenario(NavigationScenario):
    """VMAS ``navigation`` + the propwash-circulation-wake non-stationarity.

    Extra kwargs on top of the stock ``navigation`` ones:

    ``ns_severity`` (float)
        The one severity dial.  ``0.0`` -> the stock task, bit for bit.
    ``ns_gain`` (float), ``ns_rho`` (float), ``ns_driver_period`` (int),
    ``ns_phase_spread`` (bool)
        Fixed structural constants.  Calibrate once, then leave alone.
    ``ns_freeze_driver`` (float or None)
        Phase-1 freeze knob: hold the driver at a constant value.
    """

    # ------------------------------------------------------------------
    # construction / reset
    # ------------------------------------------------------------------

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Pop our kwargs *before* super(), which asserts that every kwarg it
        # was handed has been consumed.
        self.pcw = PcwParams(
            severity=float(kwargs.pop("ns_severity", 0.0)),
            gain=float(kwargs.pop("ns_gain", 4.0)),
            rho=float(kwargs.pop("ns_rho", 0.8)),
            driver_period=int(kwargs.pop("ns_driver_period", 2000)),
            phase_spread=bool(kwargs.pop("ns_phase_spread", True)),
            freeze_driver=kwargs.pop("ns_freeze_driver", None),
        )
        if self.pcw.freeze_driver is not None:
            self.pcw.freeze_driver = float(self.pcw.freeze_driver)

        if not hasattr(NavigationScenario, "process_action"):
            raise RuntimeError(
                "This VMAS build has no BaseScenario.process_action hook; "
                "navigation_pcw needs vmas>=1.3.4."
            )

        world = super().make_world(batch_dim, device, **kwargs)
        self._pcw_init(batch_dim, device, world)
        return world

    def _pcw_init(self, batch_dim: int, device, world) -> None:
        agents = world.agents
        self.pcw_n_agents = len(agents)
        self.pcw_agent_index = {a.name: i for i, a in enumerate(agents)}
        self.pcw_batch_dim = batch_dim
        self.pcw_global_step = 0

        zeros_an = torch.zeros(batch_dim, self.pcw_n_agents, device=device)
        # x2(t): circulation the medium carries into the *current* step.
        self.pcw_x2 = zeros_an.clone()
        # theta(t): deflection actually applied during the last stepped step.
        self.pcw_theta_applied = zeros_an.clone()
        # theta(t+1): deflection that *will* be applied next step.  This is the
        # Phase-1 privileged signal -- publishing it one step early is what
        # lets a decentralised controller act on it without time travel.
        self.pcw_theta_next = zeros_an.clone()
        # the per-agent scalar broadcast this step
        self.pcw_msg = zeros_an.clone()
        self.pcw_u_raw = torch.zeros(batch_dim, self.pcw_n_agents, 2, device=device)

        self.pcw_A = self._pcw_driver(device)
        self.pcw_c = self.pcw_A * self.pcw.severity
        # what was in force during the step just taken (equal to the above
        # before any step has been taken)
        self.pcw_A_applied = self.pcw_A.clone()
        self.pcw_c_applied = self.pcw_c.clone()

    def _pcw_driver(self, device=None) -> Tensor:
        return driver_A(
            self.pcw_global_step,
            self.pcw_batch_dim,
            period=self.pcw.driver_period,
            phase_spread=self.pcw.phase_spread,
            freeze=self.pcw.freeze_driver,
            device=device if device is not None else self.pcw_x2.device,
        )

    def reset_world_at(self, env_index: int = None):
        super().reset_world_at(env_index)
        if not hasattr(self, "pcw_x2"):
            # super().make_world() has not run yet; nothing to reset.
            return
        # The medium of a reset arena is still.  The *clock* is deliberately NOT
        # reset: the driver persists across episodes, which is what makes this a
        # non-stationarity rather than a per-episode randomisation.
        if env_index is None:
            self.pcw_x2.zero_()
            self.pcw_theta_applied.zero_()
            self.pcw_theta_next.zero_()
            self.pcw_msg.zero_()
            self.pcw_u_raw.zero_()
        else:
            self.pcw_x2[env_index] = 0.0
            self.pcw_theta_applied[env_index] = 0.0
            self.pcw_theta_next[env_index] = 0.0
            self.pcw_msg[env_index] = 0.0
            self.pcw_u_raw[env_index] = 0.0

    # ------------------------------------------------------------------
    # the non-stationarity itself
    # ------------------------------------------------------------------

    def process_action(self, agent):
        """Deflect the agent's thrust by the circulation-induced yaw offset.

        VMAS calls this for every agent after *all* agents' actions have been
        set and before ``world.step()``, so on the first agent we snapshot the
        whole raw command matrix and do the bookkeeping once; the per-agent
        calls then only apply their own rotation.  Snapshotting matters: agent 0
        has already been rewritten by the time agent 1 is processed, and the
        exertion functional must read the *executed commands*, not the
        already-deflected ones.
        """
        if agent is self.world.agents[0]:
            self._pcw_begin_step()

        i = self.pcw_agent_index[agent.name]
        deflected = rotate(self.pcw_u_raw[:, i], self.pcw_theta_applied[:, i])
        if agent.action.u.shape[-1] == 2:
            agent.action.u = deflected
        else:
            agent.action.u = torch.cat(
                (deflected, agent.action.u[..., 2:]), dim=-1
            )

    def _pcw_begin_step(self) -> None:
        agents = self.world.agents
        # Executed commands (post-clamp, pre-deflection) and the positions they
        # were issued from.  Both are exactly what a PACT agent shares.
        self.pcw_u_raw = torch.stack(
            [a.action.u[..., :2] for a in agents], dim=1
        ).clone()
        pos = torch.stack([a.state.pos for a in agents], dim=1)

        # 1. Harm for THIS step, from the circulation charged by earlier steps.
        #    theta = c(t) * x2(t).
        self.pcw_theta_applied = self.pcw_c.unsqueeze(-1) * self.pcw_x2
        self.pcw_A_applied = self.pcw_A
        self.pcw_c_applied = self.pcw_c

        # 2. This step's exertion charges the medium -> x2(t+1).
        self.pcw_msg = angular_impulse(pos, self.pcw_u_raw)
        phi = peer_mean(self.pcw_msg)
        self.pcw_x2 = leak_step(
            self.pcw_x2, phi, rho=self.pcw.rho, gain=self.pcw.gain
        )

        # 3. The exogenous clock advances -> c(t+1).  Never reset by episodes.
        self.pcw_global_step += 1
        self.pcw_A = self._pcw_driver()
        self.pcw_c = self.pcw_A * self.pcw.severity
        self.pcw_theta_next = self.pcw_c.unsqueeze(-1) * self.pcw_x2

    # ------------------------------------------------------------------
    # privileged read-out (info only -- never the observation)
    # ------------------------------------------------------------------

    def info(self, agent) -> Dict[str, Tensor]:
        info = super().info(agent)
        i = self.pcw_agent_index[agent.name]
        info.update(
            {
                # deflection that was in force during the step just taken
                "ns_theta_applied": self.pcw_theta_applied[:, i],
                # deflection that will be in force during the *next* step: the
                # Phase-1 privileged signal, in native units (radians)
                "ns_theta_next": self.pcw_theta_next[:, i],
                # driver-free accumulator for the next step: the Phase-2 gate target
                "ns_x2": self.pcw_x2[:, i],
                # the hidden scalar beta must learn to track
                "ns_c": self.pcw_c_applied,
                "ns_c_next": self.pcw_c,
                "ns_A": self.pcw_A_applied,
                # what this agent broadcast (comms accounting)
                "ns_msg": self.pcw_msg[:, i],
                # |theta| for calibration: this is the number to look at when
                # deciding whether `ns_gain` is sized to the policy's real
                # operating scale
                "ns_abs_theta": wrap_angle(self.pcw_theta_applied[:, i]).abs(),
            }
        )
        return info
