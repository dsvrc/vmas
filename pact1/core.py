#  PACT-1 for VMAS road_traffic.  Part II of the spec.
#
#  torch only.  No vmas, no torchrl -- the estimator self-test must run offline.
#
#  ---------------------------------------------------------------------------
#  What transferred, and what did not
#  ---------------------------------------------------------------------------
#  II.6 is the porting decision that most changes what may be claimed, and it
#  had to be made before this file existed.
#
#  URB steers over an agent's DISCRETE route options: it z-scores the predicted
#  cost of each option and subtracts g*kappa*z from that option's logit.  The
#  policy gradient reaches trust because the shift sits inside a softmax.
#
#  road_traffic assigns a reference path at reset and its action is CONTINUOUS
#  (v_command, steering).  There is no option set to rank.  So the channel here
#  is a differential PACE shift: an agent whose own route is predicted more
#  congested than the fleet's average eases off, one on a clear route presses
#  on.  This is the shift the spec itself names for traffic -- "a uniform shift
#  accomplishes exactly nothing and only a differential one helps" -- and it is
#  loop-coupled in the same way, because easing off changes who you share the
#  road with.
#
#  There is still no inverse.  You cannot subtract seconds off a congested
#  lanelet.  So this instance sits in II.6's second row: **identification and
#  steering only**, and the paper must say so.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "PactParams",
    "Basis",
    "RLS",
    "trust_from_logit",
    "confidence",
    "steer",
    "compensate",
]


@dataclass(frozen=True)
class PactParams:
    # -- sensor (P-2.1) -----------------------------------------------------
    p_trace_max: float = 100.0
    """Bound on covariance windup, as a MULTIPLE of the initial trace
    ``p0 * dim``.  Declared, not tuned.

    RLS with forgetting divides P by mu every step, so in a direction the data
    stops exciting, P grows without bound -- and the prediction built from it
    eventually overflows.  P-4.2's dead-row skip is supposed to stop the worst of
    it and cannot: it only catches an EXACTLY zero regressor, and a poorly
    excited one is not zero.

    Measured on a real 3M-frame transport run before this existed: 5.7 million
    agent-steps with a non-finite prediction out of ~12 million, fit_gain NaN
    from iteration 4 onward, and an applied correction worth 4% of the
    disturbance.  The method was not beaten by the baseline; it never ran.

    When the trace exceeds the bound P is shrunk isotropically, which preserves
    symmetry and positive-definiteness and costs one reduction per step.
    Set to 0 to disable, which is how the measurement above is reproduced."""

    y_clip: float = 10.0

    # -- estimator (II.4) ---------------------------------------------------
    mu: float = 0.999
    p0: float = 10.0
    """Declared; sweep, never tune. mu is the bias/variance dial of a tracking
    floor of order sqrt(noise * drift) -- it cannot be tuned away, only
    balanced."""

    # -- trust (II.5) -------------------------------------------------------
    g_max: float = 1.0
    trust_bias: float = 2.2
    """P-5.1, the INVERTED prior: w = 0 sits at 0.90 * g_max, i.e. near full
    reliance, not half.  The estimator already supplies the magnitude, so this
    knob tracks nothing -- whenever the estimate is right, optimal trust is a
    constant.  Starting at half spends the whole budget at half compensation
    with a correct waveform (measured: return 3642 against 5444)."""
    trust_ema: float = 0.05

    # -- channel (II.6) -----------------------------------------------------
    kappa: float = 1.0
    """Declared.  The z-score in ``steer`` is what lets this be a single
    constant rather than a per-instance scale factor -- and a tuned kappa is a
    tuned result."""

    shift_mode: str = "centred"
    """How the predicted excess is turned into a pace shift.  ``centred`` or
    ``zscore``.

    ``zscore`` is URB's literal form -- ``z = zscore(predicted)`` -- and it is
    kept only so the ablation can run it.  It is WRONG for this channel, and
    measurably so.  In URB the shift is subtracted from a LOGIT, where the only
    thing that matters is the ranking and a standardised score is exactly
    right.  Here it multiplies a physical velocity command, and standardising
    throws away the one thing that channel needs to know: how big the
    disturbance actually is.  Feeding predictions that differ by 1e-6 still
    produces ``|z| ~ 1``, so at kappa=1, trust=0.9 the measured result is 14% of
    agent-steps commanded to REVERSE and 30% with pace cut by more than half --
    against a real harm of 7-9% at the storm peak.  A compensator an order of
    magnitude larger than the thing it compensates for, and the same size at
    sigma=0.01 as at sigma=3.

    ``centred`` subtracts the fleet mean and stops:

        shift_i = 1 - g * kappa * (pred_i - mean_j pred_j)

    P-6.1 asked for a DIMENSIONLESS shift, and it already is: P-2.1 defines the
    sensor as a RELATIVE excess, ``realized/nominal - 1``, precisely so that
    path-length scaling is out of it.  The z-score is a second normalisation on
    a quantity that was already normalised, and the scale it destroys is the
    signal.  Centring keeps kappa a single declared constant, keeps the channel
    purely differential (a uniform prediction still moves nobody), and makes the
    applied compensation scale with the severity -- which is what any
    compensator is supposed to do."""

    shift_clip: float = 0.5
    """Safety bound on the pace channel: the shift is clamped to
    ``[1 - shift_clip, 1 + shift_clip]``.

    This bound is an ADAPTATION forced by II.6's second row, and it must be
    declared as one.  In URB the shift ``g * kappa * z`` is subtracted from a
    LOGIT, where a one-sigma prediction is a mild re-ranking.  Here it
    multiplies a physical velocity command, and because ``z`` is standardised
    ACROSS THE FLEET its size does not depend on how large the disturbance
    actually is -- feeding predictions that differ by 1e-6 still yields
    ``|z| ~ 1``.  Unclamped at kappa=1, trust=0.9 that is measured as 14% of
    agent-steps commanded to REVERSE and 30% with pace cut by more than half,
    against a real harm of 7-9% at the storm peak: a compensator an order of
    magnitude larger than the thing it compensates for.

    Under ``shift_mode="centred"`` this almost never binds -- the fleet spread
    in predicted relative excess is a few percent -- so it is a guard rail, not
    a tuning knob.  It exists because nothing else stops a diverging estimate
    from commanding a negative velocity, and P-7.1 only promises safety at
    ``g = 0``.

    The clamp is symmetric about 1, so P-7.1 is untouched: at ``g = 0`` the
    shift is exactly 1 and ``clip(1) == 1`` bit for bit.

    Set to 0.0 to disable it."""

    # -- pruning (P-3.4) ----------------------------------------------------
    min_share: float = 1e-3
    min_variance: float = 1e-8


# ===========================================================================
#  II.3 -- the basis
# ===========================================================================


class Basis:
    """Project the unknown per-element sensitivity field onto ``r`` known
    element classes.

    The classes are public infrastructure -- a lanelet's type and lane count are
    painted on it.  What is not handed over is ``beta*``: how much a peer unit
    on each class actually costs today.  That drifts as the operating point
    moves and must be tracked online.

    P-1.1: ``r`` is independent of the number of agents AND of the number of
    elements.  Here r = number of element classes = 3 on the CPM map.
    """

    def __init__(
        self,
        capacity: Tensor,
        element_class: Tensor,
        n_classes: int,
        routes: Sequence[Sequence[int]],
        p: PactParams,
    ) -> None:
        self.capacity = capacity
        self.element_class = element_class
        self.n_classes = int(n_classes)
        self.routes = [tuple(r) for r in routes]
        self.p = p
        self.live: List[int] = list(range(self.n_classes))
        self._shared: Optional[Tensor] = None

    # -- the waveform ------------------------------------------------------

    @property
    def shared(self) -> Tensor:
        """``S[m, p, q]`` = class-``m`` load one unit on route ``q`` places on
        the elements of route ``p``.  ``(r, P, P)``, built once from structure.

        This is a precomputation, not a shortcut: ``channels`` is on the
        per-step path of the environment AND is called thousands of times by the
        estimator self-test, and rebuilding the incidence matrix per call made
        that test take ten minutes against the spec's budget of seconds.  The
        brute-force definition in ``channels_bruteforce`` remains the authority
        and ``verify`` checks this against it at startup.
        """
        if self._shared is None:
            n_routes, n_elem = len(self.routes), self.capacity.shape[0]
            dev = self.capacity.device
            # On the capacity's device, not the default one: `capacity` arrives
            # already moved to the training device, and a CPU `inc` divided by a
            # CUDA `capacity` is a hard device-mismatch error at startup -- which
            # is why the PACT arm could never have run under DEVICE=cuda.
            inc = torch.zeros(n_routes, n_elem, device=dev)
            for p_i, r in enumerate(self.routes):
                inc[p_i, torch.as_tensor(r, dtype=torch.long, device=dev)] = 1.0
            contrib = inc / self.capacity.unsqueeze(0)  # (P, A)
            S = torch.empty(self.n_classes, n_routes, n_routes, device=dev)
            for m in range(self.n_classes):
                mask = (self.element_class == m).to(torch.float32)
                S[m] = (inc * mask) @ contrib.transpose(0, 1)
            self._shared = S
        return self._shared

    @staticmethod
    def _as_batched(route_of) -> tuple[Tensor, bool]:
        """Accept one fleet or a batch of them.  Returns ``((B, N), batched)``."""
        ix = torch.as_tensor(
            list(route_of) if not isinstance(route_of, Tensor) else route_of,
            dtype=torch.long,
        )
        if ix.dim() == 1:
            return ix.unsqueeze(0), False
        return ix, True

    def channels(self, route_of) -> Tensor:
        """``x[b, i, m] = sum over j != i of the class-m load peer j places on
        the elements agent i traverses``.

        ``(N, r)`` for a single fleet, ``(B, N, r)`` for a batch of them --
        VMAS runs B parallel worlds and each has its own route assignment, so
        collapsing them (by averaging, say) would estimate a coupling no world
        actually has.

        Zero-diagonal by construction (P-3.1): a lone agent reads exactly zero
        on every channel, which is what makes the estimated quantity a coupling
        rather than a self-effect.
        """
        ix, batched = self._as_batched(route_of)
        S = self.shared.to(ix.device)  # (r, P, P)
        # gather the (N, N) sub-block per world: S[m, ix[b,i], ix[b,j]]
        rows = S[:, ix]  # (r, B, N, P)
        sub = torch.gather(
            rows, 3, ix.unsqueeze(0).unsqueeze(2).expand(S.shape[0], *ix.shape, ix.shape[1])
        )  # (r, B, N, N)
        # subtract the j == i term rather than masking: ASSERTED, not argued
        own = torch.diagonal(sub, dim1=-2, dim2=-1)  # (r, B, N)
        out = (sub.sum(dim=-1) - own).permute(1, 2, 0).contiguous()  # (B, N, r)
        return out if batched else out[0]

    def channels_bruteforce(self, route_of: Sequence[int]) -> Tensor:
        """P-3.2: the definition, written straight out as loops.

        The vectorised path above is verified against this at startup and the
        run aborts on mismatch.  Index order and self-exclusion are exactly the
        kind of wiring bug that leaves every diagnostic looking healthy.
        """
        n = len(route_of)
        out = torch.zeros(n, self.n_classes)
        for i in range(n):
            Ei = set(self.routes[route_of[i]])
            for j in range(n):
                if j == i:
                    continue
                Ej = set(self.routes[route_of[j]])
                for a in Ei & Ej:
                    out[i, int(self.element_class[a])] += 1.0 / float(self.capacity[a])
        return out

    def verify(self, route_of: Sequence[int]) -> None:
        fast, slow = self.channels(route_of), self.channels_bruteforce(route_of)
        err = float((fast - slow).abs().max())
        if err > 1e-5:
            raise AssertionError(
                f"vectorised basis disagrees with the brute-force definition by "
                f"{err:.3e}. This is gate 1: abort, it is a wiring bug."
            )

    # -- P-3.3 the geometric reference -------------------------------------

    def geometric_reference(self, n_agents: int, samples: int = 512, seed: int = 0) -> Tensor:
        """The load each agent would see if every peer acted uniformly at random.

        A function of structure and schedule only -- no run data enters.
        Centring on this is what makes the intercept and the class channels
        separable: uncentred, the raw channels carry a large common mean against
        an intercept column of 1, and the split becomes unidentifiable even
        though prediction stays fine (measured condition number 1.3e5).
        """
        # The generator is deliberately CPU and explicit: the reference is
        # structure, so it must not depend on the device or consume the global
        # RNG stream (which would make the pact arm draw different actions from
        # the blind arm and silently break the paired comparison).
        gen = torch.Generator().manual_seed(seed)
        dev = self.capacity.device
        acc = torch.zeros(self.n_classes, device=dev)
        for _ in range(samples):
            assign = torch.randint(
                0, len(self.routes), (n_agents,), generator=gen
            ).to(dev)
            acc += self.channels(assign).mean(dim=0)
        return acc / samples

    def scale_reference(self, n_agents: int, samples: int = 512, seed: int = 0) -> Tensor:
        gen = torch.Generator().manual_seed(seed + 1)
        dev = self.capacity.device
        vals = []
        for _ in range(samples // 8):
            assign = torch.randint(
                0, len(self.routes), (n_agents,), generator=gen
            ).to(dev)
            vals.append(self.channels(assign))
        v = torch.cat(vals, dim=0)
        return v.std(dim=0, unbiased=False).clamp_min(1e-8)

    # -- P-3.4 pruning ------------------------------------------------------

    def prune(self, n_agents: int, seed: int = 0) -> List[int]:
        """Drop channels below a declared share or variance, keeping every
        downstream index aligned."""
        ref = self.geometric_reference(n_agents, seed=seed)
        std = self.scale_reference(n_agents, seed=seed)
        total = ref.sum().clamp_min(1e-30)
        keep = [
            m
            for m in range(self.n_classes)
            if float(ref[m] / total) >= self.p.min_share
            and float(std[m]) >= self.p.min_variance
        ]
        self.live = keep or list(range(self.n_classes))
        return self.live

    def design(self, route_of, ref: Tensor, scale: Tensor) -> Tensor:
        """``psi = [1, centred and scaled live channels]``.

        ``(N, 1 + r_live)`` for one fleet, ``(B, N, 1 + r_live)`` for a batch.
        """
        x = self.channels(route_of)[..., self.live]
        z = (x - ref[self.live]) / scale[self.live]
        return torch.cat([torch.ones_like(z[..., :1]), z], dim=-1)


# ===========================================================================
#  II.4 -- the estimator
# ===========================================================================


class RLS:
    """Per-agent recursive least squares with forgetting.

    P-4.1 decentralized: agent *i* never sees another agent's residual. It sees
    only peers' executed actions, which a connected fleet broadcasts anyway.
    """

    def __init__(
        self, n_agents: int, dim: int, p: PactParams, batch: int = 1, device=None
    ) -> None:
        self.p = p
        self.dim = dim
        self.batch = int(batch)
        self.device = device
        # (B, N, ...) -- each parallel world is an INDEPENDENT deployment running
        # its own estimator.  Sharing one across worlds would average couplings
        # that no single world has.
        #
        # `device` is not optional in practice: psi arrives on the training
        # device, and beta/P left on the CPU is a device mismatch on the first
        # update.
        f = dict(device=device)
        self.beta = torch.zeros(self.batch, n_agents, dim, **f)
        self.P = (
            p.p0 * torch.eye(dim, **f).expand(self.batch, n_agents, dim, dim).clone()
        )
        self.n_updates = torch.zeros(self.batch, n_agents, **f)
        self.n_skipped = torch.zeros(self.batch, n_agents, **f)
        #  how often the windup bound had to act -- a rising count is the signal
        #  that mu is too aggressive for the excitation this policy provides
        self.n_bounded = torch.zeros(self.batch, n_agents, **f)

    def predict(self, psi: Tensor) -> Tensor:
        """``beta' psi``.  ``psi`` is ``(N, d)`` or ``(B, N, d)``."""
        p = psi.unsqueeze(0) if psi.dim() == 2 else psi
        return (self.beta * p).sum(-1)

    def update(self, psi: Tensor, y: Tensor) -> Tensor:
        """One row per agent per world.  Returns the prior residual.

        P-4.2: rows whose regressor is numerically zero are SKIPPED, not fed.
        A dead row carries no information about beta but still divides P by mu,
        inflating the covariance every step and silently tightening the
        effective forgetting factor -- so mu stops meaning what the banner says.
        """
        if psi.dim() == 2:
            psi = psi.unsqueeze(0).expand(self.batch, -1, -1)
        if y.dim() == 1:
            y = y.unsqueeze(0).expand(self.batch, -1)

        #  P-4.2, and it was NOT firing.  psi carries an intercept column of
        #  exactly 1, so `psi.abs().sum(-1) > 0` is true for every row ever
        #  built and `n_skipped` was 0 on every iteration of every run.  The
        #  regressor that matters is the CHANNELS; a row with no channel content
        #  carries nothing about beta and must not divide P by mu.
        chan = psi[..., 1:] if psi.shape[-1] > 1 else psi
        live = chan.abs().sum(dim=-1) > 0
        resid = y - (self.beta * psi).sum(-1)

        Ppsi = torch.einsum("bnij,bnj->bni", self.P, psi)
        denom = self.p.mu + (psi * Ppsi).sum(-1)
        K = Ppsi / denom.unsqueeze(-1).clamp_min(1e-12)
        new_beta = self.beta + K * resid.unsqueeze(-1)
        new_P = (self.P - K.unsqueeze(-1) * Ppsi.unsqueeze(-2)) / self.p.mu
        new_P = 0.5 * (new_P + new_P.transpose(-1, -2))

        #  Bound the windup.  Without this, a direction the policy stops exciting
        #  grows P by 1/mu every step until the prediction overflows -- see
        #  PactParams.p_trace_max for what that measured on a real run.
        if self.p.p_trace_max > 0:
            cap = self.p.p_trace_max * self.p.p0 * self.dim
            tr = new_P.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, N)
            shrink = (cap / tr.clamp_min(1e-12)).clamp(max=1.0)
            new_P = new_P * shrink.unsqueeze(-1).unsqueeze(-1)
            self.n_bounded += (shrink < 1.0).to(torch.float32)

        m = live.unsqueeze(-1)
        self.beta = torch.where(m, new_beta, self.beta)
        self.P = torch.where(m.unsqueeze(-1), new_P, self.P)
        self.n_updates += live.to(torch.float32)
        self.n_skipped += (~live).to(torch.float32)
        return resid


# ===========================================================================
#  II.5 -- trust
# ===========================================================================


def trust_from_logit(w: Tensor, p: PactParams) -> Tensor:
    """``g = g_max * sigmoid(w + bias)``.  The inverted prior of P-5.1."""
    return p.g_max * torch.sigmoid(w + p.trust_bias)


def confidence(psi: Tensor, P: Tensor, p: PactParams, r: int) -> Tensor:
    """``conf = 1 / (1 + r * psi'P psi / (p0 * ||psi||^2))``.

    P-5.2: gate on the uncertainty of the scalar the compensator actually uses,
    NOT on tr(P).  The trace version is a trap -- it is dominated by the least
    excited direction, which forgetting inflates without bound, so once the
    fleet converges it quietly disarms a working compensator while fit R^2 still
    reads 0.9998.
    """
    if psi.dim() == 2 and P.dim() == 4:
        psi = psi.unsqueeze(0).expand(P.shape[0], -1, -1)
    if psi.dim() == 2:
        quad = torch.einsum("ni,nij,nj->n", psi, P, psi)
    else:
        quad = torch.einsum("bni,bnij,bnj->bn", psi, P, psi)
    norm2 = psi.pow(2).sum(-1).clamp_min(1e-12)
    return 1.0 / (1.0 + r * quad / (p.p0 * norm2))


# ===========================================================================
#  II.6 / II.7 -- the channel and the floor property
# ===========================================================================


def steer(v_command: Tensor, predicted: Tensor, g: Tensor, p: PactParams) -> Tensor:
    """Differential pace shift.  ``(N,) -> (N,)``.

    ``predicted`` is each agent's predicted relative excess -- already
    dimensionless by P-2.1.  It is CENTRED on the fleet mean, so a uniform
    prediction moves nobody: only the differential helps, which is the commons
    in miniature.  See ``PactParams.shift_mode`` for why centring rather than
    z-scoring, and what z-scoring measurably costs here.

    P-7.1, the floor property: at ``g = 0`` this returns ``v_command`` bit for
    bit for any ``predicted``, however wrong; and when every prediction is
    identical the shift is defined to be exactly zero rather than NaN. The
    estimator therefore sits entirely outside the worst-case decision path.
    """
    # ACROSS THE FLEET, within each world: the last axis is the fleet.  Pooling
    # worlds would make one world's congestion steer another's vehicles.
    mean = predicted.mean(dim=-1, keepdim=True)
    dev = predicted - mean
    if p.shift_mode == "zscore":
        std = predicted.std(dim=-1, unbiased=False, keepdim=True)
        dev = torch.where(
            std > 1e-12, dev / std.clamp_min(1e-12), torch.zeros_like(predicted)
        )
    elif p.shift_mode != "centred":
        raise ValueError(f"unknown shift_mode {p.shift_mode!r}")
    shift = 1.0 - g * p.kappa * dev
    if p.shift_clip > 0.0:
        # Symmetric about 1, so g = 0 still gives exactly 1.0 and the floor
        # property is preserved bit for bit.  See PactParams.shift_clip.
        shift = shift.clamp(1.0 - p.shift_clip, 1.0 + p.shift_clip)
    return v_command * shift


def compensate(
    u_command: Tensor, direction: Tensor, predicted: Tensor, g: Tensor
) -> Tensor:
    """II.6's FIRST row: the exact channel inverse.  ``(B, N, D) -> (B, N, D)``.

        u_sent = u_command - g * direction * predicted

    Use this where the disturbance is ADDITIVE in the agent's own action space,
    so a correct estimate cancels it rather than merely routing around it.  The
    method may then claim identification AND compensation; where no inverse
    exists, use ``steer`` instead and claim identification and steering only.
    Conflating the two is the one thing II.6 says is not publishable.

    ``direction`` is the unit vector the disturbance arrives along.  It is PUBLIC
    -- computed from the declared operator and the peers' broadcast actions --
    which is precisely what lets a scalar estimate cancel a vector disturbance.
    ``predicted`` is the scalar magnitude the estimator supplies.

    P-7.1, the floor property: at ``g = 0`` this returns ``u_command`` bit for
    bit for any ``predicted``, however wrong, because the correction term is
    multiplied by exactly zero rather than merely by something small.  The
    estimator therefore sits entirely outside the worst-case decision path: a
    diverging estimate can fail to help, and cannot drag the arm below the
    baseline it wraps.

    No clamp here on purpose.  The host clips to its own action range and
    reports the clipped fraction; a clamp inside the method would hide the
    saturation that bounds sigma*.
    """
    return u_command - (g.unsqueeze(-1) * predicted.unsqueeze(-1)) * direction


def herd_index(route_of: Sequence[int], n_routes: int) -> float:
    """P-8.1: normalised Herfindahl over the fleet's choices.

    0 = perfectly spread, 1 = everyone on one option. Rising concentration
    alongside rising trust is the externality becoming visible.

    **Logged, never acted on.** Acting on it would make the method a mechanism
    rather than a per-agent estimator and break the decentralization claim.
    """
    ix = torch.as_tensor(route_of)
    if ix.dim() == 1:
        ix = ix.unsqueeze(0)
    n = ix.shape[-1]
    if n <= 1:
        return 1.0
    counts = torch.zeros(ix.shape[0], n_routes, device=ix.device)
    counts.scatter_add_(1, ix, torch.ones_like(ix, dtype=counts.dtype))
    shares = counts / counts.sum(-1, keepdim=True).clamp_min(1)
    h = (shares**2).sum(-1)
    out = (h - 1.0 / n) / (1.0 - 1.0 / n)
    # A tensor in, a tensor out: the caller logs this every step, and forcing a
    # float here would sync the device on every step to print one number.
    return out if isinstance(route_of, Tensor) and route_of.dim() > 1 else float(out[0])
