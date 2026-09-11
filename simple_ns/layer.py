#  The layer, and the four hosts it stacks onto.
#
#      BaseScenario
#        +-- <stock vmas scenario>      balance / transport / sampling / navigation
#              +-- ExertionMixin        <- I.4: EVERY arm gets this, unmodified
#                    +-- PactMixin      <- adds the compensator only
#
#  NS-3.1: the dial sits BELOW the method in the hierarchy and is read from the
#  task configuration.  A dial only the method's arm experienced is worthless as
#  evidence.
#
#  ---------------------------------------------------------------------------
#  Why one mixin covers all four hosts
#  ---------------------------------------------------------------------------
#  ``road_ns`` needed a per-host adapter because its medium is a lanelet map and
#  each host had to answer "which element is this vehicle on".  Here the medium is
#  the space between the agents, so the layer needs only what every VMAS scenario
#  already has: positions and a 2-D force action.  Adding a fifth host is a
#  two-line class, not a port.
#
#  ---------------------------------------------------------------------------
#  Where the disturbance is applied, and why there
#  ---------------------------------------------------------------------------
#  In ``process_action``, i.e. below the action interface and above the physics.
#  That is the single place every host reads an action, which is I.4's corollary:
#  apply the harm where every host shares one hook, not in a training loop that
#  only some hosts have.  ``reward``, ``done`` and ``observation`` are inherited
#  untouched -- the agent is paid exactly what it was paid before, and earns less
#  only because it physically achieved less (NS-1.4).

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from vmas.simulator.core import Agent, World

from pact1.core import PactParams, RLS, compensate, confidence
from simple_ns.coupling import Coupling
from simple_ns.driver import DialParams, beta_star, driver_A

__all__ = ["ExertionMixin", "PactMixin", "NS_KWARGS", "PACT_KWARGS"]


NS_KWARGS = (
    "ns_severity",
    "ns_period",
    "ns_wet_fraction",
    "ns_loss_at_sigma1",
    "ns_rho",
    "ns_n_types",
    "ns_recv_spread",
    "ns_send_spread",
    "ns_kernel_lambda",
    "ns_y_clip",
    "ns_observe_residual",
    "ns_direct",
)

PACT_KWARGS = (
    "pact_enabled",
    "pact_trust",
    "pact_mu",
    "pact_p0",
    "pact_warmup",
    "pact_channels",
    "pact_oracle",
    "pact_corr_clip",
)


def _pop(kwargs: Dict[str, Any], keys) -> Dict[str, Any]:
    return {k: kwargs.pop(k) for k in keys if k in kwargs}


class ExertionMixin:
    """Interaction-mediated, invertible non-stationarity.

    Overrides only ``make_world``, ``reset_world_at``, ``process_action``,
    ``post_step``, ``observation`` and ``info``.
    """

    # ------------------------------------------------------------------
    #  construction
    # ------------------------------------------------------------------

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        raw = _pop(kwargs, NS_KWARGS)
        self._pact_raw = _pop(kwargs, PACT_KWARGS)

        self.ns = DialParams(
            severity=float(raw.get("ns_severity", 1.0)),
            period=int(raw.get("ns_period", 100)),
            wet_fraction=float(raw.get("ns_wet_fraction", 0.5)),
            loss_at_sigma1=float(raw.get("ns_loss_at_sigma1", 0.14)),
            rho=float(raw.get("ns_rho", 0.9)),
            n_types=int(raw.get("ns_n_types", 3)),
            recv_spread=float(raw.get("ns_recv_spread", 0.6)),
            send_spread=float(raw.get("ns_send_spread", 0.8)),
            kernel_lambda=float(raw.get("ns_kernel_lambda", 0.35)),
            y_clip=float(raw.get("ns_y_clip", 10.0)),
        )
        self._observe_residual = bool(raw.get("ns_observe_residual", True))
        # The (B) CONTROL.  See _disturbance for what it changes and why the
        # comparison between it and the default is the paper's central pair.
        self._direct = bool(raw.get("ns_direct", False))

        world = super().make_world(batch_dim, device, **kwargs)

        agents = list(world.agents)
        self.n_ag = len(agents)
        self._agent_index = {a.name: i for i, a in enumerate(agents)}
        self.coupling = Coupling(self.n_ag, self.ns, device=device)
        self._action_dim = int(agents[0].action_size)
        # the action box, per agent, so the clip is the host's own
        self._u_range = torch.tensor(
            [
                [float(x) for x in (
                    a.u_range if isinstance(a.u_range, (list, tuple))
                    else [a.u_range] * self._action_dim
                )]
                for a in agents
            ],
            device=device,
            dtype=torch.float32,
        )  # (N, D)

        # NS-2.4: sigma = 1 is a declared physical statement, so the raw channel
        # scale -- which depends on N, on the kernel, and above all on how this
        # host arranges its agents -- is divided out.  The reference geometry is
        # the host's OWN spawn distribution; see Coupling.load_norm for what a
        # uniform-over-arena reference cost when it was tried.
        self._arena = float(
            max(getattr(world, "x_semidim", None) or 1.0,
                getattr(world, "y_semidim", None) or 1.0)
        )
        self._load_norm = self.coupling.load_norm(self._spawn_reference(world))

        B, N, r, D = batch_dim, self.n_ag, self.ns.n_types, self._action_dim
        f = dict(device=device, dtype=torch.float32)
        self._step = torch.zeros(B, device=device, dtype=torch.long)
        self._u_prev = torch.zeros(B, N, D, **f)
        self._Q = torch.zeros(B, N, r, 2, **f)
        self._ehat = torch.zeros(B, N, 2, **f)
        self._x = torch.zeros(B, N, r, **f)
        self._load = torch.zeros(B, N, **f)
        self._d = torch.zeros(B, N, D, **f)
        self._y = torch.zeros(B, N, **f)
        self._y_prev = torch.zeros(B, N, **f)
        self._A = torch.zeros(B, **f)
        self._clipped = torch.zeros(B, N, **f)

        # NS-3.3: count what the layer actually touched, as tensors so there is
        # no device sync on the per-step path.
        self._n_seen = torch.zeros((), device=device, dtype=torch.long)
        self._n_hit = torch.zeros((), device=device, dtype=torch.long)

        self._on_built(world, device)
        print(self.coupling.banner())
        print(
            f"exertion layer  sigma={self.ns.severity} period={self.ns.period} "
            f"wet={self.ns.wet_fraction} L={self.ns.loss_at_sigma1} "
            f"N={self.n_ag} D={self._action_dim} "
            f"channel={'DIRECT (B, control)' if self._direct else 'COUPLED (C)'}"
        )
        return world

    def _spawn_reference(self, world: World, draws: int = 24) -> Tensor:
        """``(draws, N, 2)`` positions from the host's own ``reset_world_at``.

        The global RNG is snapshotted and restored, so drawing the reference does
        not shift the run's stream -- otherwise the blind and pact arms would see
        different environments for a reason that has nothing to do with either.
        """
        # `env_make_world` sets `self._world` from our return value, one line
        # after this runs, so the host's reset would otherwise assert.  Setting it
        # here is exactly what the caller is about to do.
        self._world = world
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(20260911)
            out = []
            for _ in range(draws):
                super().reset_world_at(None)
                out.append(
                    torch.stack([a.state.pos for a in world.agents], dim=1)[0]
                    .detach()
                    .cpu()
                )
            return torch.stack(out)
        finally:
            torch.random.set_rng_state(state)

    def _on_built(self, world: World, device: torch.device) -> None:
        """Hook for the layer above the dial."""
        return

    # ------------------------------------------------------------------
    #  lifecycle
    # ------------------------------------------------------------------

    def reset_world_at(self, env_index: Optional[int] = None):
        out = super().reset_world_at(env_index)
        if env_index is None:
            for t in (self._u_prev, self._Q, self._ehat, self._x, self._load,
                      self._d, self._y, self._y_prev, self._clipped):
                t.zero_()
        else:
            for t in (self._u_prev, self._Q, self._ehat, self._x, self._load,
                      self._d, self._y, self._y_prev, self._clipped):
                t[env_index] = 0.0
        # NS-3.4: the driver's clock is NOT reset.  The bearing does not un-wear
        # and the afternoon does not un-warm because an episode ended.
        self._on_reset(env_index)
        return out

    def _on_reset(self, env_index: Optional[int]) -> None:
        return

    # ------------------------------------------------------------------
    #  the disturbance
    # ------------------------------------------------------------------

    def _positions(self) -> Tensor:
        return torch.stack([a.state.pos for a in self.world.agents], dim=1)

    def _disturbance(self) -> None:
        """Compute this step's additive disturbance for the whole fleet."""
        pos = self._positions()
        a = driver_A(self._step, self.ns)
        self._A = a

        Q, q, ehat, x = self.coupling.step_channels(pos, self._u_prev, self._Q)
        self._Q, self._ehat, self._x = Q, ehat, x

        beta = beta_star(a, self.ns)  # (B, r)
        # normalised to the declared reference, and scaled to the receiver's own
        # action range, so "0.14 of the action range at the peak" is literal
        self._load = (beta.unsqueeze(1) * x).sum(-1) / self._load_norm
        self._load = self._load * self._u_range[:, 0].reshape(1, -1)

        if self._direct:
            #  THE (B) CONTROL -- exogenous, not interaction-mediated.
            #
            #  Same driver, same severity, same scale, same reward, same ladder.
            #  The ONLY change is that the disturbance no longer passes through
            #  the neighbours: every agent gets it directly, along its own
            #  commanded direction, with no sum over j != i anywhere.
            #
            #  That single flag moves the instance from cell (C) to cell (B), and
            #  the pair is the experiment the classification rests on:
            #
            #    * A LONE agent now feels it, so the N=1 test separates the two
            #      cells by measurement rather than by argument.
            #    * PACT's PEER CHANNELS carry no information about it.  The
            #      disturbance is the same for every agent, so the regression
            #      puts all of it in the INTERCEPT and the class channels go to
            #      zero: fit gain over an intercept-only null collapses to ~0.
            #
            #  Note what that does and does not predict.  PACT still helps on
            #  (B) -- the intercept tracks a level shift perfectly well -- so
            #  "the method fails on B" would be the wrong claim and the
            #  experiment would refute it.  The right claim is sharper: on (B) a
            #  single per-agent adaptive bias does the whole job, so nothing
            #  about the problem is multi-agent; on (C) the peer channels are
            #  load-bearing and removing them removes the recovery.  That is
            #  what `pact_channels=intercept` isolates, and it is the pair the
            #  classification actually rests on.
            u_cmd = torch.stack(
                [ag.action.u for ag in self.world.agents], dim=1
            )
            nrm = u_cmd.norm(dim=-1, keepdim=True)
            direction = torch.where(
                nrm > 1e-12, u_cmd / nrm.clamp_min(1e-12), torch.zeros_like(u_cmd)
            )
            mag = self.ns.severity * self.ns.loss_at_sigma1 * a.reshape(-1, 1)
            self._load = mag * self._u_range[:, 0].reshape(1, -1)
            self._d = -direction * self._load.unsqueeze(-1)
            # the sensor still measures what it measures
            self._ehat = -direction[..., :2]
        else:
            d = ehat * self._load.unsqueeze(-1)
            if self._action_dim != 2:
                pad = torch.zeros(
                    *d.shape[:-1], self._action_dim - 2, device=d.device, dtype=d.dtype
                )
                d = torch.cat([d, pad], dim=-1)
            self._d = d

        self._after_disturbance()

    def _after_disturbance(self) -> None:
        return

    def _correction(self, index: int) -> Optional[Tensor]:
        """The compensator's correction for agent ``index``, or None.  The dial
        never produces one; the layer above (PACT) overrides this."""
        return None

    def process_action(self, agent: Agent) -> None:
        if agent is self.world.agents[0]:
            self._disturbance()

        i = self._agent_index[agent.name]
        u_cmd = agent.action.u

        corr = self._correction(i)
        u_sent = u_cmd if corr is None else u_cmd - corr

        # NS-1.4: the disturbance is an unmodelled force added to what the
        # actuator was asked for.  The reward function is never touched.  At
        # sigma = 0 the load is exactly 0.0, so this is u + 0.0 -- bit for bit
        # the stock command.
        u_exec = u_sent + self._d[:, i]

        lo, hi = -self._u_range[i], self._u_range[i]
        # clamp(NaN) is NaN, so sanitise first -- see _after_disturbance
        clipped = torch.nan_to_num(u_exec, nan=0.0, posinf=0.0, neginf=0.0).clamp(lo, hi)
        self._clipped[:, i] = (clipped != u_exec).any(-1).to(torch.float32)
        agent.action.u = clipped

        # II.2 / P-2.1: proprioception.  The agent knows what it SENT and can
        # measure what its actuator delivered, so the residual is observable
        # without any privileged quantity.  Note this is the residual of the
        # command actually sent, so compensating does not blind the estimator --
        # which is what lets trust stay armed once it is working.
        self._y[:, i] = ((clipped - u_sent) * self._ehat[:, i, : self._action_dim]).sum(-1)

        if self.ns.severity > 0:
            self._n_hit += (self._load[:, i].abs() > 0).sum()
        self._n_seen += u_cmd.shape[0]

        super().process_action(agent)

    def post_step(self) -> None:
        super().post_step()
        # Peers' EXECUTED exertion, which is what the medium actually transmits
        # and what P-4.1 lets an agent see.  Feeding the INTENDED action instead
        # was a measured bug on the Ant instance.
        self._u_prev = torch.stack(
            [a.action.u for a in self.world.agents], dim=1
        ).clone()
        self._y_prev = self._y.clone()
        self._step = self._step + 1

    # ------------------------------------------------------------------
    #  sensor and read-out
    # ------------------------------------------------------------------

    def observation(self, agent: Agent):
        obs = super().observation(agent)
        if not self._observe_residual:
            return obs
        i = self._agent_index[agent.name]
        # ONE STEP STALE, and clipped to a declared bound (P-2.1).
        y = self._y_prev[:, i : i + 1].clamp(-1.0, self.ns.y_clip)
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["residual"] = y
            return obs
        return torch.cat([obs, y], dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        try:
            info = dict(super().info(agent))
        except (AttributeError, NotImplementedError):
            info = {}
        i = self._agent_index[agent.name]
        info.update(
            {
                "ns_load": self._load[:, i : i + 1],
                "ns_dmag": self._d[:, i].norm(dim=-1, keepdim=True),
                "ns_A": self._A.unsqueeze(-1),
                "ns_y": self._y[:, i : i + 1],
                "ns_clipped": self._clipped[:, i : i + 1],
                "ns_x_std": self._x[:, i].std(dim=-1, keepdim=True),
            }
        )
        return info

    # ------------------------------------------------------------------
    #  NS-3.3 -- fail loudly when the layer is inert
    # ------------------------------------------------------------------

    def severity_report(self) -> str:
        return (
            f"exertion: actions seen {int(self._n_seen)}, disturbed "
            f"{int(self._n_hit)}"
        )

    def assert_layer_fired(self) -> None:
        if self.ns.severity <= 0:
            return
        if int(self._n_hit) == 0:
            raise RuntimeError(
                f"ns_severity={self.ns.severity} but NOT ONE action was "
                f"disturbed over {int(self._n_seen)} agent-steps. The layer is "
                "not reaching the physics; this is a wiring bug, not a null "
                "result."
            )


class PactMixin(ExertionMixin):
    """PACT-1 on top of the dial, in II.6's INVERTIBLE row.

    The disturbance is additive in the agent's own action space and its direction
    is public, so a correct scalar estimate cancels it exactly.  The method may
    therefore claim identification **and compensation** here -- unlike
    ``road_ns``, where no inverse exists and the claim is steering only.

    Everything except the channel is the same object as in ``road_ns``: the RLS
    with its dead-row skip, the inverted trust prior, and the prediction-based
    confidence gate all come from ``pact1.core`` unchanged.  That is deliberate.
    A cross-cell comparison run on two separately implemented methods measures
    the implementations.
    """

    def _on_built(self, world: World, device: torch.device) -> None:
        raw = self._pact_raw
        self.pact_enabled = bool(raw.get("pact_enabled", False))
        self.pact_params = PactParams(
            mu=float(raw.get("pact_mu", 0.999)),
            p0=float(raw.get("pact_p0", 10.0)),
            y_clip=self.ns.y_clip,
        )
        self._trust_const = float(raw.get("pact_trust", 0.9))
        self._warmup = int(raw.get("pact_warmup", 200))
        #  The ablation that carries the (B) vs (C) claim.  "intercept" keeps the
        #  identical method, the identical trust, the identical channel and the
        #  identical floor property, and deletes ONLY the peer channels -- so
        #  what is left is a per-agent adaptive bias with no knowledge of who the
        #  neighbours are.  On (B) it should lose nothing.  On (C) it should lose
        #  most of the recovery, because there the disturbance differs per agent
        #  and a fleet-average cannot represent it.
        #  ANT_complete_story.md 2.2's free-answer controller, as an ARM rather
        #  than as a probe outside the environment.  It replaces the estimate
        #  with the TRUE disturbance magnitude and changes nothing else -- same
        #  channel, same trust, same clipping -- so it measures the CEILING: a
        #  learner cannot beat a controller that already knows the answer, and
        #  where this fails, nothing can.  That is what separates "the method is
        #  weak here" from "this severity is past sigma* and the row is about the
        #  environment".
        #
        #  It has to live here, not in a wrapper.  Computed outside the env the
        #  best available answer is one step stale, which understates the ceiling
        #  badly enough that PACT appeared to BEAT it -- measured 78.9% against
        #  65.0% of B0 at sigma = 2, which is not a possible result and would
        #  have been caught by a reviewer rather than by us.
        #  A DECLARED bound on the correction, as a fraction of the agent's own
        #  action range.  Not a tuning knob -- a guard rail, and the reason it
        #  exists is a measured trap.
        #
        #  At mu <= 0.95 the covariance blows up and the estimate becomes
        #  meaningless (prediction error 1e8, then 1e34).  Unbounded, that garbage
        #  correction is clamped by the action box into a large saturating kick,
        #  and on transport's heuristic it SCORED BETTER than a correct estimate:
        #  96.2% of B0 at mu=0.9 against 84.7% at mu=0.97.  Selecting mu on
        #  return would therefore have selected a diverged estimator, and the
        #  paper would have reported a random perturbation as a method.
        #
        #  Bounding the correction closes that route: a diverged estimate can now
        #  only fail to help, which is what P-7.1 is supposed to guarantee.  Two
        #  consequences worth stating in the ablation: mu is selected on
        #  PREDICTION ERROR, never on return; and this bound is what makes that
        #  selection safe.
        self._corr_clip = float(raw.get("pact_corr_clip", 0.5))
        self._oracle = bool(raw.get("pact_oracle", False))
        self._channels = str(raw.get("pact_channels", "full"))
        if self._channels not in ("full", "intercept"):
            raise ValueError(
                f"pact_channels must be 'full' or 'intercept', got "
                f"{self._channels!r}"
            )

        self._dim = 1 if self._channels == "intercept" else 1 + self.ns.n_types
        self._ref, self._scale = self.coupling.geometric_reference(arena=self._arena)
        # One estimator per parallel world: each is an independent deployment.
        self.rls = RLS(
            self.n_ag, self._dim, self.pact_params, batch=world.batch_dim, device=device
        )

        B = world.batch_dim
        self._pred = torch.zeros(B, self.n_ag, device=device)
        self._trust = torch.zeros(B, self.n_ag, device=device)
        self._conf = torch.zeros(B, self.n_ag, device=device)
        self._corr = torch.zeros(B, self.n_ag, self._action_dim, device=device)
        self._n_diverged = torch.zeros((), device=device, dtype=torch.long)

        # II.9 gates 1 and 3, at startup, before a single episode is simulated.
        gen = torch.Generator().manual_seed(0)
        pos = ((torch.rand(2, self.n_ag, 2, generator=gen) * 2 - 1) * self._arena).to(device)
        u = (torch.rand(2, self.n_ag, 2, generator=gen) * 2 - 1).to(device)
        print("PACT gate 1/3   " + self.coupling.verify(pos, u))
        print(
            f"PACT            enabled={self.pact_enabled} trust={self._trust_const} "
            f"mu={self.pact_params.mu} channels={self._channels} "
            f"{'ORACLE(ceiling) ' if self._oracle else ''}"
            f"dim={self._dim} warmup={self._warmup} "
            f"channel=EXACT INVERSE (II.6 row 1)"
        )

    def _on_reset(self, env_index: Optional[int]) -> None:
        if not hasattr(self, "_corr"):
            return
        if env_index is None:
            self._corr.zero_()
        else:
            self._corr[env_index] = 0.0

    def _after_disturbance(self) -> None:
        if not self.pact_enabled:
            self._corr.zero_()
            return

        psi = self.coupling.design(self._x, self._ref, self._scale)  # (B,N,1+r)
        if self._channels == "intercept":
            psi = psi[..., :1]

        # II.2: the target is the agent's own residual, ONE STEP STALE.  It never
        # sees another agent's residual (P-4.1).
        y = self._y_prev.clamp(-self.pact_params.y_clip, self.pact_params.y_clip)
        self.rls.update(psi, y)

        self._pred = self.rls.predict(psi)  # (B, N)
        self._conf = confidence(psi, self.rls.P, self.pact_params, self._dim)

        #  P-7.1, enforced rather than hoped for.  "A diverging estimate can fail
        #  to help; it cannot drag the arm below the baseline it wraps" -- but a
        #  NON-FINITE estimate does worse than that: it puts a NaN in the action
        #  and the host asserts, killing the run.  Measured: at mu <= 0.95 the
        #  covariance blows up on this instance and every seed crashed.
        #
        #  So a non-finite prediction is treated as no information: trust goes to
        #  exactly zero for that agent and the floor property returns the
        #  untouched policy.  The estimator stays outside the worst-case decision
        #  path, which is the whole point of the property.
        #  Guard the PREDICTION and the CONFIDENCE: a blown-up covariance makes
        #  psi'Ppsi non-finite too, so checking only the prediction leaves a NaN
        #  route into the correction.
        bad = ~(torch.isfinite(self._pred) & torch.isfinite(self._conf))
        if bool(bad.any()):
            self._pred = torch.where(bad, torch.zeros_like(self._pred), self._pred)
            self._conf = torch.where(bad, torch.zeros_like(self._conf), self._conf)
            self._n_diverged += bad.sum()

        if self._oracle:
            self._pred = self._load.clone()
            self._conf = torch.ones_like(self._conf)

        # P-5.1's inverted prior lives in pact_trust: near full reliance, not
        # half.  P-5.2 gates it on PREDICTION uncertainty, never on tr(P).
        ready = self.rls.n_updates.min(dim=-1).values >= self._warmup  # (B,)
        if self._oracle:
            ready = torch.ones_like(ready)
        self._trust = (
            torch.where(ready, self._trust_const, 0.0).unsqueeze(-1) * self._conf
        )

        dirn = self._ehat
        if self._action_dim != 2:
            pad = torch.zeros(
                *dirn.shape[:-1], self._action_dim - 2,
                device=dirn.device, dtype=dirn.dtype,
            )
            dirn = torch.cat([dirn, pad], dim=-1)
        # `process_action` sends `u_cmd - corr`, which is exactly
        # `compensate(u_cmd, dirn, pred, trust)`.  Stored per fleet so the
        # per-agent hook is a slice.
        corr = (self._trust.unsqueeze(-1) * self._pred.unsqueeze(-1)) * dirn
        #  The last line of defence, and the one that makes P-7.1 a guarantee
        #  rather than an argument: whatever the estimator did, the correction
        #  that reaches the actuator is finite.  A NaN here is not "fails to
        #  help" -- it propagates into the state, the observation and then the
        #  POLICY's own action, and the host asserts.  Measured at mu <= 0.95:
        #  every seed died that way, and guarding only `pred` did not stop it.
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        if self._corr_clip > 0.0:
            lim = self._corr_clip * self._u_range.unsqueeze(0)  # (1, N, D)
            corr = corr.clamp(-lim, lim)
        self._corr = corr

    def _correction(self, index: int) -> Optional[Tensor]:
        if not self.pact_enabled:
            return None
        return self._corr[:, index]

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
                "pact_corr": self._corr[:, i].norm(dim=-1, keepdim=True),
                "pact_updates": self.rls.n_updates[:, i : i + 1],
                "pact_skipped": self.rls.n_skipped[:, i : i + 1],
                # how much of the disturbance the correction actually removed
                "pact_residual": (self._load[:, i : i + 1] - one(self._pred)).abs(),
                "pact_diverged": torch.full_like(
                    one(self._pred), float(self._n_diverged)
                ),
            }
        )
        return info
