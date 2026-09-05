#  PACT on Shared Link Contention -- the compensator, arithmetic only.
#
#  ---------------------------------------------------------------------------
#  What it is, in one paragraph
#  ---------------------------------------------------------------------------
#  Each agent's loading excess decomposes into a term it causes itself and a
#  term its peers cause.  The peer term lies in a low-dimensional subspace
#  spanned by waveforms **computed exactly** from one-step-delayed peer exertion
#  projected onto the environment's own declared coupling operator.  Each agent
#  tracks the subspace's coefficients online by recursive least squares on its
#  own proprioceptive residual, and applies the certified channel inverse driven
#  by that estimate.  The host RL algorithm is **never modified** -- everything
#  lives below the environment interface, so every arm shares hyperparameters
#  and an arm difference cannot be an algorithm difference.
#
#  ---------------------------------------------------------------------------
#  What each agent knows, and what it does not
#  ---------------------------------------------------------------------------
#  KNOWS
#    * ``u_i(t-1)``      its own loading, one step stale.  The sensor.
#    * ``Phi_j(t-1)``    one scalar per robot per step, broadcast.  This is the
#                        coordination channel and it is **paid for by every
#                        arm**: it rides in ``phi_floor``, the heartbeat frame
#                        that already loads the medium whether or not anybody
#                        listens to it.
#    * ``Phi_i(t)``      its own current exertion -- it knows its own velocity.
#    * the declared operator ``E, D, W``, and ``g(t)``, a function of observable
#      time.  The **information-matched baseline gets all of this too**
#      (``--pact_mode ff``), or the gap is information rather than mechanism.
#
#  DOES NOT KNOW
#    * ``Phi_j(t)``      peers' *current* exertion.  This is the whole problem:
#                        the harm applied at step t is built from it.
#    * ``L_k(t)``, ``u_i(t)``, which channel binds.
#
#  ---------------------------------------------------------------------------
#  The floor property
#  ---------------------------------------------------------------------------
#  When the gates say inadmissible, ``trust`` is exactly ``0.0`` and the
#  executed action is **byte-identical** to the information-matched
#  feedforward arm.  A diverging estimate can fail to help; it must never make
#  things worse.  ``test_slc_pact.py`` asserts this bit for bit.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from benchmarl.environments.vmas_slc.slc_core import SlcOperator, SlcParams

__all__ = ["PactParams", "PactCompensator"]

STATE_INERT = 0.0
STATE_ASLEEP = 1.0
STATE_ALIVE = 2.0


@dataclass(frozen=True)
class PactParams:
    """Everything the compensator needs.  ``max_trust`` and ``mu`` are the two
    Phase-1 calibration parameters; both must be **swept, reported with the
    sweep, calibrated on one seed and validated on held-out seeds**.
    """

    enabled: bool = True
    ff_gain: float = 1.0
    """The analytic driver feedforward.  **LOCAL** -- it needs no peer
    information, so it supports no coordination claim.  ``ff_gain = 0`` is the
    peer-only ablation that isolates coordination; ``pact_mode = 'ff'`` with
    ``max_trust = 0`` is the information-matched baseline."""
    own_gain: float = 1.0
    """The own-exertion correction.  Also LOCAL and also exact -- an agent knows
    its own velocity."""
    max_trust: float = 0.80
    """The T4 gain cap.  Binary admissibility x a calibrated constant, never a
    product of heuristic confidences (8.1)."""

    # --- the estimator ------------------------------------------------------
    r: int = 1
    """Peer channels.  Default 1: an r=2 strong/weak split measured *worse*
    conditioned than a single weighted channel (3006 vs 810) because after
    per-channel normalisation both columns collapse to a weighted mean of peer
    exertion fractions and go near-collinear."""
    mu: float = 0.99
    """RLS forgetting factor.  Measured, not assumed -- **re-measure per
    environment**, the optimum follows the drift rate.  Aggressive forgetting
    buys nothing and injects noise straight into the coefficients."""
    p0: float = 10.0
    p_max_mult: float = 10.0
    """Covariance windup bound: ``p_max = p_max_mult * p0 * dim``.  Forgetting
    inflates unexcited directions by ``1/mu`` every update *without bound*; as
    the policy converges excitation dies and the estimator runs away."""

    # --- the gates ----------------------------------------------------------
    gate: str = "fit"
    """``fit`` (default): gate on measured lift over the null.  ``trace``:
    reproduces the silent-disarm failure -- **an ablation, never control**.
    ``always``: no admissibility gate, for the floor-property test."""
    fit_floor: float = 0.0
    ready_updates: int = 200
    fit_ema: float = 2e-3
    """Windowed, not cumulative: a cumulative lift is held down forever by early
    negatives."""
    warmup_updates: int = 50
    """Skip before accumulating fit statistics -- a cold start otherwise
    dominates both sums and their difference is noise."""
    trace_gate_p0: float = 10.0

    # --- the channel inverse ------------------------------------------------
    mode: str = "delta"
    """``delta``: every term is a correction to the *measured* stale loading, so
    the pedestal is removed by construction.  ``level``: 6.4 as written, an
    explicit slow EMA of the standing level.  ``ff``: feedforward only, the
    information-matched baseline arm."""
    level_tau: float = 400.0
    """``mode='level'`` only.  Must be >> the driver period's in-episode
    variation.  A declared constant; it belongs in the ablation table."""
    max_delta: float = 1.0
    """Rail on the correction, in action units.  Track contact separately from
    the action-box rail; they mean different things."""
    denom_floor: float = 0.15
    """Floor on ``1 - c_hat``.  A rail-pinned delta is a **constant bias, not a
    compensation** -- it stops responding to the estimate entirely."""
    u_cap: float = 3.0
    """Sanity clamp on the predicted loading."""


class PactCompensator:
    """Per-agent, per-vectorised-world RLS + channel inverse.

    State is ``(B, N, ...)``: each vectorised world is an independent
    deployment running its own estimator, which also gives ``B`` replicates for
    the diagnostics.  ``beta`` and ``P`` deliberately **survive episode
    resets** -- a deployment's radio characterisation persists across missions.
    The one-step history (``psi_lag``, ``own_lag``) does not, and is cleared per
    reset env.
    """

    def __init__(
        self,
        params: PactParams,
        slc: SlcParams,
        operator: SlcOperator,
        batch_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.p = params
        self.slc = slc
        self.op = operator
        self.B = int(batch_dim)
        self.N = operator.n_agents
        self.device = device
        self.dtype = dtype

        self.r_eff, self.chan_mask, self.psi_range, self.psi_ref = self._build_basis()
        self.dim = 2 + self.r_eff

        B, N, d = self.B, self.N, self.dim
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)  # noqa: E731
        eye = torch.eye(d, device=device, dtype=dtype)

        self.beta = z(B, N, d)
        self.P = params.p0 * eye.expand(B, N, d, d).clone()
        self.beta_null = z(B, N, 2)
        self.P_null = params.p0 * torch.eye(2, device=device, dtype=dtype).expand(
            B, N, 2, 2
        ).clone()

        self.psi_lag = z(B, N, self.r_eff)
        self.own_lag = z(B, N)
        self.have_lag = torch.zeros(B, N, device=device, dtype=torch.bool)

        self.level = z(B, N)
        self.n_updates = z(B, N)

        self.sse_full = z(B, N)
        self.sse_null = z(B, N)
        self.ybar = z(B, N)
        self.yvar = z(B, N)
        self.clamp_hits = z(B, N)

        # E[reg reg'] running mean, for cond(psi) -- can theta be *decomposed*,
        # not merely predicted?
        self.gram = z(N, d, d)
        self.gram_n = 0.0

        self.p_max = params.p_max_mult * params.p0 * d
        self._eye = eye
        self._eye2 = torch.eye(2, device=device, dtype=dtype)

        # last-step diagnostics
        self.diag: Dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    #  the declared basis (2.2, 2.3)
    # ------------------------------------------------------------------

    def _build_basis(self) -> Tuple[int, Tensor, Tensor, Tensor]:
        """Split each agent's peers into ``r`` channels by coupling strength,
        carrying the operator magnitudes as **weights inside** each channel, not
        as flat buckets.

        Per-agent, per-channel scaling by each channel's **own declared range**
        (2.3).  A single scale shared across agents drove ``cond`` to 72,148
        against 57, because an agent whose weak channel is ~100x smaller than
        its strong one contributes a near-zero column and the Gram goes
        singular.  Each agent runs its own estimator, so this leaks nothing.
        """
        W = self.op.W  # (N, N), zero diagonal
        N = self.N
        r = max(int(self.p.r), 1)

        mask = torch.zeros(r, N, N, device=self.device, dtype=self.dtype)
        if r == 1:
            mask[0] = W
        else:
            # split by coupling strength, per agent, at within-agent quantiles
            for i in range(N):
                w = W[i]
                pos = w[w > 0]
                if pos.numel() == 0:
                    continue
                qs = torch.quantile(
                    pos,
                    torch.linspace(0.0, 1.0, r + 1, device=self.device, dtype=self.dtype)[
                        1:-1
                    ],
                )
                edges = torch.cat(
                    [
                        torch.zeros(1, device=self.device, dtype=self.dtype),
                        qs,
                        torch.full((1,), float("inf"), device=self.device, dtype=self.dtype),
                    ]
                )
                for c in range(r):
                    sel = (w > edges[c]) & (w <= edges[c + 1])
                    mask[c, i] = torch.where(sel, w, torch.zeros_like(w))

        # declared range and geometric reference: peers acting uniformly at
        # random, NEVER the sample mean -- the sample mean is run data and turns
        # a declared model class into a fit.
        span = self.slc.phi_span
        mid = self.slc.phi_nominal
        rng = mask.sum(-1) * span  # (r, N)
        ref = mask.sum(-1) * mid

        # drop channels whose coverage is negligible rather than forcing them to
        # survive; report the EFFECTIVE r
        live = (rng > 0).any(-1)
        if not bool(live.any()):
            live = torch.zeros_like(live)
            live[0] = True
        mask = mask[live]
        rng = rng[live]
        ref = ref[live]
        r_eff = int(mask.shape[0])

        return r_eff, mask, rng.transpose(0, 1).contiguous(), ref.transpose(0, 1).contiguous()

    def peer_basis(self, phi: Tensor) -> Tensor:
        """``psi[i, c] = (x[i, c] - x_ref[i, c]) / range[i, c]`` in ``[-.5, .5]``.

        Args:
            phi: ``(B, N)`` one-step-delayed peer exertion (the broadcast).

        Returns:
            ``(B, N, r_eff)``
        """
        # x[b, i, c] = sum_j mask[c, i, j] * phi[b, j]
        x = torch.einsum("cij,bj->bic", self.chan_mask, phi)
        rng = self.psi_range.unsqueeze(0)
        out = (x - self.psi_ref.unsqueeze(0)) / rng.clamp_min(1e-30)
        # a dead channel contributes an exactly-zero column, never a NaN
        return torch.where(rng > 0, out, torch.zeros_like(out))

    def own_basis(self, phi_own: Tensor) -> Tensor:
        """The own column, on the same declared scaling.  Built from the
        *diagonal* read of the operator -- excluded from the coupling basis
        because it is not coupling, but exactly what the inverse needs.

        Args:
            phi_own: ``(B, N)`` the agent's own CURRENT exertion.
        """
        e_own = self.op.E[:, torch.arange(self.N, device=self.device)]  # (C, N)
        link = self.op.D.to(self.dtype)  # (N, C)
        w_own = (link * e_own.transpose(0, 1)).sum(-1) / link.sum(-1)  # (N,)
        rng = (w_own * self.slc.phi_span).clamp_min(1e-30)
        ref = w_own * self.slc.phi_nominal
        return (w_own.view(1, -1) * phi_own - ref.view(1, -1)) / rng.view(1, -1)

    # ------------------------------------------------------------------
    #  the estimator (5)
    # ------------------------------------------------------------------

    def _rls(
        self,
        beta: Tensor,
        P: Tensor,
        psi: Tensor,
        y: Tensor,
        active: Tensor,
        eye: Tensor,
        track_clamp: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """One vectorised RLS step with forgetting.  Returns
        ``(beta, P, prior_residual)``.
        """
        mu = self.p.mu
        prior = (beta * psi).sum(-1)
        resid = y - prior

        Ppsi = torch.einsum("bnij,bnj->bni", P, psi)
        denom = mu + (psi * Ppsi).sum(-1)
        k = Ppsi / denom.unsqueeze(-1).clamp_min(1e-12)

        new_beta = beta + k * resid.unsqueeze(-1)
        new_P = (P - k.unsqueeze(-1) * Ppsi.unsqueeze(-2)) / mu
        new_P = 0.5 * (new_P + new_P.transpose(-1, -2))  # the gates read P

        # ---- covariance windup bound.  Test NON-FINITE FIRST: an
        # ``isfinite(c) and c > thr`` guard lets the most degenerate basis
        # possible pass silently.
        trP = new_P.diagonal(dim1=-2, dim2=-1).sum(-1)
        bad = ~torch.isfinite(trP)
        if bool(bad.any()):
            new_P = torch.where(bad[..., None, None], self.p.p0 * eye, new_P)
            trP = new_P.diagonal(dim1=-2, dim2=-1).sum(-1)
        over = trP > self.p_max
        scale = torch.where(over, self.p_max / trP.clamp_min(1e-30), torch.ones_like(trP))
        new_P = new_P * scale[..., None, None]
        if track_clamp:
            self.clamp_hits = torch.where(active, over.to(self.dtype), self.clamp_hits)

        a = active.unsqueeze(-1)
        beta = torch.where(a, new_beta, beta)
        P = torch.where(a.unsqueeze(-1), new_P, P)
        return beta, P, resid

    # ------------------------------------------------------------------
    #  the step
    # ------------------------------------------------------------------

    def step(
        self,
        u_prev: Tensor,
        phi_bcast: Tensor,
        phi_own_now: Tensor,
        g_prev: Tensor,
        g_now: Tensor,
        alive: Tensor,
    ) -> Dict[str, Tensor]:
        """Advance the estimator and return this step's control quantities.

        Args:
            u_prev:      ``(B, N)`` the sensor -- ``u_i(t-1)``.
            phi_bcast:   ``(B, N)`` peers' broadcast exertion, ``Phi(t-1)``.
            phi_own_now: ``(B, N)`` each agent's own CURRENT exertion, ``Phi_i(t)``.
            g_prev:      ``(B, N)`` capacity ratio on the agent's worst link at t-1.
            g_now:       ``(B, N)`` ditto at t.  Both analytic from observable time.
            alive:       ``(B, N)`` bool -- envs with a valid one-step history.

        Returns:
            dict with ``c_hat`` (the loading-loss estimate the inverse divides
            by) plus the full diagnostic set.
        """
        p = self.p
        psi_now = self.peer_basis(phi_bcast)  # psi_peer(t-1)
        own_now = self.own_basis(phi_own_now)  # own_col(t)   -- CURRENT
        ones = torch.ones_like(own_now)

        # -- 1. score and update on the row formed last step ---------------
        # The regressor is ASYMMETRIC in time: own column CURRENT (the agent
        # knows its own action exactly), peer columns PREVIOUS (it cannot
        # observe peers' current exertion).  Regressing y(t) on the whole of
        # psi(t-1) would lag the own-gain column by one step and corrupt the
        # very coefficient the inverse depends on.
        active = alive & self.have_lag
        reg = torch.cat([ones.unsqueeze(-1), self.own_lag.unsqueeze(-1), self.psi_lag], dim=-1)
        reg_null = torch.cat([ones.unsqueeze(-1), self.own_lag.unsqueeze(-1)], dim=-1)

        self.beta, self.P, resid_full = self._rls(
            self.beta, self.P, reg, u_prev, active, self._eye, track_clamp=True
        )
        self.beta_null, self.P_null, resid_null = self._rls(
            self.beta_null, self.P_null, reg_null, u_prev, active, self._eye2
        )
        self.n_updates = self.n_updates + active.to(self.dtype)

        self._accumulate_fit(u_prev, resid_full, resid_null, active)
        self._accumulate_gram(reg, active)

        # -- 2. this step's control quantities ------------------------------
        beta_peer = self.beta[..., 2:]
        beta_own = self.beta[..., 1]
        beta_0 = self.beta[..., 0]

        ell_now = (beta_peer * psi_now).sum(-1)  # peer part of u(t)
        ell_prev = (beta_peer * self.psi_lag).sum(-1)  # peer part of u(t-1)
        self.level = self.level + (ell_now - self.level) / max(p.level_tau, 1.0)

        # the driver moved between t-1 and t.  u = L/K and K scales by g, so if
        # L were unchanged then u(t) = u(t-1) * g(t-1)/g(t).  Closed form, no
        # estimator, no gate, nothing to converge -- and it needs no peer
        # information, so it supports NO coordination claim.
        ratio = torch.where(
            g_now > 0, g_prev / g_now.clamp_min(1e-12), torch.ones_like(g_now)
        )
        ff_term = u_prev * (ratio - 1.0)
        own_term = beta_own * (own_now - self.own_lag)

        fit_gain = self.fit_gain()
        admissible = self._admissible(fit_gain, alive)
        trust = torch.where(
            admissible,
            torch.full_like(u_prev, p.max_trust),
            torch.zeros_like(u_prev),
        )

        if p.mode == "ff":
            peer_term = torch.zeros_like(u_prev)
            trust = torch.zeros_like(trust)
        elif p.mode == "level":
            peer_term = trust * (ell_now - self.level)
        else:  # "delta"
            peer_term = trust * (ell_now - ell_prev)

        ff = p.ff_gain * ff_term
        own_c = p.own_gain * own_term

        if p.mode == "level":
            u_hat = beta_0 + beta_own * own_now + ff + peer_term
        else:
            u_hat = u_prev + ff + own_c + peer_term

        u_hat = torch.where(alive, u_hat, u_prev).clamp(0.0, p.u_cap)
        c_hat = (self.slc.harm_gain * u_hat).clamp(0.0, self.slc.harm_cap)

        # -- 3. roll the one-step history -----------------------------------
        self.psi_lag = torch.where(alive.unsqueeze(-1), psi_now, self.psi_lag)
        self.own_lag = torch.where(alive, own_now, self.own_lag)
        self.have_lag = self.have_lag | alive

        state = torch.where(
            (self.psi_range.sum(-1) > 0).unsqueeze(0).expand_as(u_prev),
            torch.where(admissible, torch.full_like(u_prev, STATE_ALIVE),
                        torch.full_like(u_prev, STATE_ASLEEP)),
            torch.full_like(u_prev, STATE_INERT),
        )

        self.diag = {
            "u_hat": u_hat,
            "c_hat": c_hat,
            # The FOUR-way split, and it must be reported in full.  Only
            # ``peer_abs`` carries peer information; ``base`` (the agent's own
            # stale sensor reading), ``ff`` and ``own`` are all LOCAL and all
            # available to the information-matched baseline.  Quoting a
            # coordination result without this split is not honest.
            "base_abs": u_prev.abs() if p.mode != "level" else beta_0.abs(),
            "ff_abs": ff.abs(),
            "own_abs": own_c.abs(),
            "peer_abs": peer_term.abs(),
            "applied_trust": trust,
            "fit_gain": fit_gain,
            "ell_now": ell_now,
            "level": self.level,
            "own_gain_coef": beta_own,
            "n_updates": self.n_updates,
            "trP": self.P.diagonal(dim1=-2, dim2=-1).sum(-1),
            "clamp_frac": self.clamp_hits,
            "state": state,
        }
        return self.diag

    # ------------------------------------------------------------------
    #  fit statistics and gates (4.3, 8)
    # ------------------------------------------------------------------

    def _accumulate_fit(
        self, y: Tensor, resid_full: Tensor, resid_null: Tensor, active: Tensor
    ) -> None:
        """The null model is mandatory.  The reported quality metric is the
        **lift over the null**, never a raw R^2: a pooled R^2 of 0.9998 looked
        like a triumph on POWER until the intercept-only model was scored too
        and came in at 0.656.

        Score one-step-ahead (**prior**) predictions, not posterior fits, and
        skip a warmup before accumulating -- a cold start otherwise dominates
        both sums and their difference is noise.
        """
        a = self.p.fit_ema
        acc = active & (self.n_updates >= self.p.warmup_updates)
        af = acc.to(self.dtype)
        self.sse_full = self.sse_full + af * a * (resid_full.pow(2) - self.sse_full)
        self.sse_null = self.sse_null + af * a * (resid_null.pow(2) - self.sse_null)
        self.ybar = self.ybar + af * a * (y - self.ybar)
        self.yvar = self.yvar + af * a * ((y - self.ybar).pow(2) - self.yvar)

    def fit_gain(self) -> Tensor:
        """``R2(full) - R2(intercept + own only)``, windowed.

        **Guarded with NaN, never an epsilon.**  Where the target has no
        variance the lift is genuinely meaningless; a ``1e-12`` floor once
        produced a ``-1011`` that poisoned a column average.  NaN compares False
        against the floor, so an undefined lift is inadmissible by construction.
        """
        return torch.where(
            self.yvar > 1e-10,
            (self.sse_null - self.sse_full) / self.yvar,
            torch.full_like(self.yvar, float("nan")),
        )

    def _admissible(self, fit_gain: Tensor, alive: Tensor) -> Tensor:
        """Separate WHETHER from HOW MUCH.

        RLS returns the least-squares prediction, and for a least-squares
        predictor the residual-minimising gain is exactly 1 because the LS
        prediction *is* the conditional mean.  T4 pulls it below 1 and
        estimation noise pulls it further -- to an **interior optimum**, not to
        nothing.  Multiplying heuristic confidences produced an applied trust
        ~40x below the theoretical gain on POWER.  Binary admissibility x a
        calibrated constant took it back.
        """
        p = self.p
        ready = self.n_updates >= p.ready_updates
        live = (self.psi_range.sum(-1) > 0).unsqueeze(0).expand_as(fit_gain)
        if p.gate == "always":
            return alive & ready & live
        if p.gate == "trace":
            # THE TRAP.  tr(P) is dominated by the LEAST excited direction,
            # which forgetting inflates without bound; once the policy converges
            # this disarms a working compensator while every other number still
            # looks healthy.  Kept only so the ablation can reproduce it.
            trP = self.P.diagonal(dim1=-2, dim2=-1).sum(-1)
            conf = 1.0 / (1.0 + trP / p.trace_gate_p0)
            return alive & ready & live & (conf > 0.5)
        return alive & ready & live & (fit_gain > p.fit_floor)

    def _accumulate_gram(self, reg: Tensor, active: Tensor) -> None:
        if not bool(active.any()):
            return
        g = torch.einsum("bni,bnj->nij", reg * active.unsqueeze(-1).to(self.dtype), reg)
        n = float(active.to(self.dtype).sum(0).mean())
        self.gram = self.gram + g
        self.gram_n += max(n, 1e-9)

    def cond_psi(self) -> float:
        """Can ``theta`` be *decomposed*, not merely predicted?

        Report ``cond`` before claiming to identify anything.  Non-finite is a
        **value**, tested first -- ``if isfinite(c) and c > thr`` lets the most
        degenerate basis possible pass silently.
        """
        if self.gram_n <= 0:
            return float("nan")
        g = self.gram / self.gram_n
        try:
            ev = torch.linalg.eigvalsh(g.to(torch.float64))
        except Exception:
            return float("inf")
        lo = ev[:, 0].clamp_min(0.0)
        hi = ev[:, -1]
        c = torch.where(lo > 0, hi / lo, torch.full_like(hi, float("inf")))
        finite = c[torch.isfinite(c)]
        if finite.numel() == 0:
            return float("inf")
        return float(finite.max())

    # ------------------------------------------------------------------
    #  the channel inverse (6)
    # ------------------------------------------------------------------

    def compensate(self, action: Tensor, c_hat: Tensor) -> Tuple[Tensor, Tensor]:
        """``a' = a / (1 - c_hat)``, railed.

        The correction is a pure **gain** along the commanded direction, so it
        never changes the agent's heading -- it restores the magnitude the
        sampled-data loop lost.

        Returns:
            ``(delta, clipped)``.  ``clipped`` is contact with the *compensation*
            rail; the caller separately tracks contact with the action box.  The
            two mean different things and must never be pooled.
        """
        denom = (1.0 - c_hat).clamp_min(self.p.denom_floor).unsqueeze(-1)
        delta = action * (1.0 / denom - 1.0)
        clipped = (delta.abs() > self.p.max_delta).any(-1)
        delta = delta.clamp(-self.p.max_delta, self.p.max_delta)
        return delta, clipped

    # ------------------------------------------------------------------
    #  lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_index: Optional[int] = None) -> None:
        """Clear the one-step history.  ``beta`` and ``P`` deliberately survive:
        the deployment's radio characterisation is not re-learned every mission.
        """
        if env_index is None:
            self.psi_lag.zero_()
            self.own_lag.zero_()
            self.have_lag.zero_()
        else:
            self.psi_lag[env_index] = 0.0
            self.own_lag[env_index] = 0.0
            self.have_lag[env_index] = False

    def to(self, device: torch.device) -> "PactCompensator":
        for name, value in list(self.__dict__.items()):
            if isinstance(value, Tensor):
                self.__dict__[name] = value.to(device)
        self.device = device
        return self

    # ------------------------------------------------------------------
    #  reporting
    # ------------------------------------------------------------------

    def banner(self) -> str:
        live = int((self.psi_range.sum(-1) > 0).sum())
        return (
            f"PACT      mode={self.p.mode} gate={self.p.gate} r_eff={self.r_eff} "
            f"mu={self.p.mu} max_trust={self.p.max_trust} ff_gain={self.p.ff_gain} "
            f"agents_with_live_coupling={live}/{self.N}"
        )
