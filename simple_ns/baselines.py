#  BASELINES.md B10 -- the non-learning compensators, as ARMS of the same
#  environment.
#
#  Two of them, and both exist to answer a control engineer rather than a MARL
#  reviewer:
#
#  * ``eso``      -- a per-agent extended-state / disturbance observer (linear
#                    ADRC).  "Why does an estimator on a declared basis beat a
#                    disturbance observer that needs no structure at all?"
#                    No peer information, no basis, one gain.
#  * ``rls_raw``  -- the SAME recursive least squares PACT uses, on the raw
#                    N-1 peer draws instead of the r declared classes.  "Why not
#                    learn the coupling instead of declaring it?"  r parameters
#                    against N-1.
#
#  Both are wired as arms of ``ExertionMixin``, not as wrappers, for the reason
#  VMAS_BALANCE.md section 9 records: a compensator computed outside the
#  environment is one step stale, and a one-step-stale ceiling once made PACT
#  look like it beat the free-answer controller.  They share PACT's trust
#  constant, warm-up and correction bound, so a difference between an arm and
#  PACT is a difference in the ESTIMATOR and nothing else.
#
#  See `baselines/docs/eso_dob.md` and `baselines/docs/rls_raw.md`.

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from vmas.simulator.core import Agent, World

from pact1.core import PactParams, RLS, confidence
#  BASELINE_KWARGS is declared in layer.py so that ExertionMixin.make_world
#  pops these keys on EVERY arm, blind included: an unconsumed scenario
#  kwarg is a silent no-op.  Everything shared with PACT (trust, warm-up,
#  mu, p0, the correction bound, the sensor clip) is deliberately read from
#  the PACT block -- these arms are ESTIMATOR swaps, so every other knob is
#  held equal.
from simple_ns.layer import BASELINE_KWARGS, ExertionMixin
from simple_ns.observer import eso_gains, eso_step

__all__ = ["EsoMixin", "RlsRawMixin", "BASELINE_KWARGS"]


class _CompensatorArm(ExertionMixin):
    """Shared plumbing: PACT's channel and bounds, somebody else's estimate.

    The disturbance is applied by ``ExertionMixin`` exactly as it is for every
    other arm.  What an arm supplies is a per-agent scalar ``pred`` and an
    applied reliance ``trust``; the layer turns those into the feed-forward
    ``u / (1 - trust*pred)`` on the droop channel, or into the additive
    correction ``trust*pred*e_hat`` on the shove channel.
    """

    def _on_built(self, world: World, device: torch.device) -> None:
        raw = self._pact_raw
        if bool(raw.get("pact_enabled", False)):
            raise ValueError(
                "pact_enabled=true together with a B10 baseline arm. These are "
                "alternative compensators for the same channel; run one at a "
                "time, or the two feed-forwards multiply."
            )
        #  Held equal with PACT on purpose -- see the module docstring.
        self._trust_const = float(raw.get("pact_trust", 0.9))
        self._warmup = int(raw.get("pact_warmup", 200))
        self._corr_clip = float(raw.get("pact_corr_clip", 0.5))

        B, N, D = world.batch_dim, self.n_ag, self._action_dim
        f = dict(device=device, dtype=torch.float32)
        self._pred = torch.zeros(B, N, **f)
        self._trust = torch.zeros(B, N, **f)
        self._conf = torch.ones(B, N, **f)
        self._corr = torch.zeros(B, N, D, **f)
        self._n_steps = torch.zeros(B, N, **f)
        self._build_estimator(world, device)

    # -- hooks for the concrete arms ------------------------------------

    def _build_estimator(self, world: World, device: torch.device) -> None:
        raise NotImplementedError

    def _estimate(self) -> Tensor:
        """This step's per-agent prediction of the disturbance, ``(B, N)``."""
        raise NotImplementedError

    def _ready(self) -> Tensor:
        """``(B,)`` -- has the estimator seen enough to be relied on?"""
        return self._n_steps.min(dim=-1).values >= self._warmup

    # -- the arm --------------------------------------------------------

    def _on_reset(self, env_index: Optional[int]) -> None:
        if not hasattr(self, "_corr"):
            return
        if env_index is None:
            self._corr.zero_()
        else:
            self._corr[env_index] = 0.0

    def _after_disturbance(self) -> None:
        pred = self._estimate()
        #  Same guard as PACT (P-7.1): a non-finite estimate is treated as NO
        #  information rather than propagated into the action, where it would
        #  reach the observation and kill the run.
        bad = ~torch.isfinite(pred)
        if bool(bad.any()):
            pred = torch.where(bad, torch.zeros_like(pred), pred)
        self._pred = pred
        self._trust = torch.where(
            self._ready(), self._trust_const, 0.0
        ).unsqueeze(-1) * self._conf

        if self.ns.channel == "droop":
            self._corr = torch.zeros_like(self._corr)
            return

        dirn = self._ehat
        if self._action_dim != 2:
            pad = torch.zeros(
                *dirn.shape[:-1], self._action_dim - 2,
                device=dirn.device, dtype=dirn.dtype,
            )
            dirn = torch.cat([dirn, pad], dim=-1)
        corr = (self._trust.unsqueeze(-1) * self._pred.unsqueeze(-1)) * dirn
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        if self._corr_clip > 0.0:
            lim = self._corr_clip * self._u_range.unsqueeze(0)
            corr = corr.clamp(-lim, lim)
        self._corr = corr

    def _correction(self, index: int) -> Optional[Tensor]:
        return self._corr[:, index]

    def _trust_for(self, index: int) -> Tensor:
        return self._trust[:, index]

    def _pred_for(self, index: int) -> Tensor:
        return self._pred[:, index]

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = super().info(agent)
        i = self._agent_index[agent.name]
        applied = (self._trust[:, i] * self._pred[:, i]).abs()
        truth = self._load[:, i].abs()
        info.update(
            {
                "pact_pred": self._pred[:, i : i + 1],
                "pact_trust_applied": self._trust[:, i : i + 1],
                "pact_trust_policy": torch.full_like(
                    self._pred[:, i : i + 1], self._trust_const
                ),
                "pact_corr_vs_d": (applied / truth.clamp_min(1e-9)).unsqueeze(-1),
            }
        )
        return info


class EsoMixin(_CompensatorArm):
    """Linear ADRC's extended state observer, one per agent, no peer information.

    The agent measures its own residual ``y`` (on the droop channel, the
    fractional shortfall ``1 - delivered/commanded``; on the shove channel, the
    projection of the excess onto ``e_hat``).  A discrete second-order extended
    state observer tracks it and extrapolates one step:

        e  = y(t-1) - z1
        z1 <- z1 + z2 + beta1 * e
        z2 <- z2 + beta2 * e
        pred(t) = z1 + z2

    ``z1`` is the observed disturbance and ``z2`` its rate, so ``z1 + z2`` is the
    one-step-ahead extrapolation.  The two observer poles are placed together at
    ``1 - w`` on the unit disc, which is Gao's bandwidth parameterisation in its
    discrete form (Miklosovic, Radke & Gao, ACC 2006):

        beta1 = 2w ,      beta2 = w^2

    so the arm has exactly ONE tuning constant, the observer bandwidth ``w``.
    That is the point of the baseline: no basis, no classes, no peer channel,
    one gain.

    What it cannot do, and what the comparison is for: the residual it filters
    is one step old, and the disturbance is generated by the peers' reaction to
    the same drift, so a fast component arrives before the observer has seen it.
    PACT's channels are computed from the peers' BROADCAST actions before the
    step, which is the structural difference the row is meant to expose.
    """

    def _build_estimator(self, world: World, device: torch.device) -> None:
        self._w = float(_pop_baseline(self._baseline_raw, "eso_bandwidth", 0.3))
        self._beta1, self._beta2 = eso_gains(self._w)
        B, N = world.batch_dim, self.n_ag
        f = dict(device=device, dtype=torch.float32)
        self._z1 = torch.zeros(B, N, **f)
        self._z2 = torch.zeros(B, N, **f)
        print(
            f"ESO/DOB arm    bandwidth w={self._w} -> beta1={self._beta1:.4f} "
            f"beta2={self._beta2:.4f}, trust={self._trust_const}, "
            f"warmup={self._warmup} steps, NO peer information"
        )

    def _on_reset(self, env_index: Optional[int]) -> None:
        super()._on_reset(env_index)
        if not hasattr(self, "_z1"):
            return
        #  The observer state is per-deployment, not per-episode: the rig does
        #  not forget what it learned about the rail because a lift finished.
        #  Only the correction is cleared, above, which the base class does.
        return

    def _estimate(self) -> Tensor:
        y = self._y_prev.clamp(-1.0, self.ns.y_clip)
        self._z1, self._z2, pred = eso_step(
            self._z1, self._z2, y, self._beta1, self._beta2
        )
        self._n_steps = self._n_steps + 1.0
        return pred


class RlsRawMixin(_CompensatorArm):
    """PACT's estimator on the raw peer draws: no classes, N-1 parameters.

    The regressor is

        psi_i = [ 1, x_i1, ..., x_i(i-1), x_i(i+1), ..., x_iN ]
        x_ij(t) = rho * x_ij(t-1) + (1 - rho) * c_ij * ||u_j(t-1)|| / u_range_i

    with ``c_ij = 1`` -- "regress the residual on the N-1 broadcast actions
    directly", as BASELINES.md B10 puts it -- or ``c_ij = W_ij`` when
    ``rls_raw_use_operator`` is set, which hands the arm the declared operator
    and leaves only the class reduction as the difference.  Both are reported;
    the first is the literal baseline and the second is the stronger version of
    it, so the claim does not rest on withholding public information.

    Everything else is PACT's, object for object: the same ``pact1.core.RLS``
    with the same forgetting factor, the same covariance bound, the same
    dead-row skip, the same prediction-uncertainty gate, the same trust prior
    and the same correction bound.  The only thing that changes is the design
    matrix, which is what the ablation is about: ``r`` parameters that do not
    grow with the fleet against ``N - 1`` that do.

    Restricted to the droop channel: a draw on a shared supply is a scalar, so
    "the peer's exertion" is unambiguous. On the shove channel the peer's
    contribution is a vector and "the raw peer action" would need a projection,
    i.e. a piece of declared structure -- which is the thing this arm exists to
    do without.
    """

    def _build_estimator(self, world: World, device: torch.device) -> None:
        if self.ns.channel != "droop":
            raise NotImplementedError(
                "ns_baseline=rls_raw is defined for the droop channel, where a "
                "peer's draw is a scalar. On the shove channel the raw peer "
                "action is a vector and regressing on it needs a declared "
                "projection -- the structure this arm exists to do without."
            )
        self._use_operator = bool(
            _pop_baseline(self._baseline_raw, "rls_raw_use_operator", False)
        )
        raw = self._pact_raw
        self.pact_params = PactParams(
            mu=float(raw.get("pact_mu", 0.95)),
            p0=float(raw.get("pact_p0", 10.0)),
            p_trace_max=float(raw.get("pact_p_trace_max", 100.0)),
            y_clip=self.ns.y_clip,
        )
        N = self.n_ag
        if N < 2:
            raise ValueError("ns_baseline=rls_raw needs at least two agents.")
        self._dim = N  # intercept + (N - 1) peers
        self.rls = RLS(
            N, self._dim, self.pact_params, batch=world.batch_dim, device=device
        )
        B = world.batch_dim
        f = dict(device=device, dtype=torch.float32)
        self._xraw = torch.zeros(B, N, N - 1, **f)
        self._psi_prev = torch.zeros(B, N, self._dim, **f)
        self._have_prev = False
        #  (N, N-1) index of "the peers of i, in order", so the regressor has a
        #  fixed column meaning per agent and the zero diagonal is structural.
        self._peer_idx = torch.stack(
            [
                torch.tensor([j for j in range(N) if j != i], device=device)
                for i in range(N)
            ]
        )
        print(
            f"RLS-raw arm    dim={self._dim} (intercept + {N - 1} peers) against "
            f"PACT's {1 + self.ns.n_types}, mu={self.pact_params.mu}, "
            f"use_operator={self._use_operator}, trust={self._trust_const}, "
            f"warmup={self._warmup}"
        )

    def _channels(self) -> Tensor:
        pos = self._positions()
        draw = self._u_prev.norm(dim=-1)  # (B, N) peer exertion magnitudes
        if self._use_operator:
            W = self.coupling.W(pos)  # (B, N, N), zero diagonal
            weighted = W * draw.unsqueeze(1)  # (B, N, N): row i, column j
        else:
            weighted = draw.unsqueeze(1).expand(-1, self.n_ag, -1)
        idx = self._peer_idx.unsqueeze(0).expand(weighted.shape[0], -1, -1)
        peers = torch.gather(weighted, 2, idx)  # (B, N, N-1)
        #  P-3.3's scaling, in the only form available without classes: make the
        #  regressor dimensionless against the receiver's own action range. No
        #  centring -- the intercept column carries the mean.
        peers = peers / self._u_range[:, 0].reshape(1, -1, 1)
        self._xraw = self.ns.rho * self._xraw + (1.0 - self.ns.rho) * peers
        ones = torch.ones_like(self._xraw[..., :1])
        return torch.cat([ones, self._xraw], dim=-1)

    def _estimate(self) -> Tensor:
        psi = self._channels()
        y = self._y_prev.clamp(-self.pact_params.y_clip, self.pact_params.y_clip)
        #  Pair y(t-1) with the row that PRODUCED it, exactly as PactMixin does
        #  -- pairing it with psi(t) regresses the target on a near-independent
        #  row and was a measured bug on this instance.
        if self._have_prev:
            self.rls.update(self._psi_prev, y)
        self._psi_prev = psi.clone()
        self._have_prev = True
        self._n_steps = self.rls.n_updates
        self._conf = confidence(psi, self.rls.P, self.pact_params, self._dim)
        self._conf = torch.where(
            torch.isfinite(self._conf), self._conf, torch.zeros_like(self._conf)
        )
        return self.rls.predict(psi)

    def _ready(self) -> Tensor:
        return self.rls.n_updates.min(dim=-1).values >= self._warmup


def _pop_baseline(raw: Dict[str, Any], key: str, default):
    return raw.get(key, default)
