#  The declared coupling operator and the PACT basis.
#
#  NS-1.2 and P-1.1/P-1.2/P-3.1/P-3.2/P-3.3.
#
#  ---------------------------------------------------------------------------
#  The reduction
#  ---------------------------------------------------------------------------
#  The unknown quantity is a transmission field: how much of each neighbour's
#  exertion actually reaches you today.  It is projected onto ``r`` declared
#  agent classes, so the number of parameters is ``r``, independent of the number
#  of agents (P-1.1).  What is declared is the GEOMETRY -- who is near whom, and
#  how compliant the receiver is.  What is not declared is ``beta*``: how much a
#  unit of a class-m neighbour's exertion costs, which drifts with the driver.
#
#      Q_m,i(t) = rho Q_m,i(t-1) + (1-rho) * sum_{j != i, type(j)=m} W_ij u_j(t-1)
#      q_i      = sum_m Q_m,i                      (B, N, 2)
#      e_i      = q_i / ||q_i||                    the direction, PUBLIC
#      x_m,i    = <Q_m,i , e_i>                    the channels, PUBLIC  (B, N, r)
#      psi_i    = [1, x_1,i, ..., x_r,i]
#
#      d_i      = e_i * ( beta*(t) . psi_i )       the disturbance, PRIVATE
#
#  Three things this buys, and each one is a requirement rather than a
#  convenience:
#
#  * The model is EXACTLY linear in psi.  The medium's memory lives in the public
#    filtered channels, not in the private disturbance, so the estimator's model
#    class is correct rather than approximately correct.  Putting the leak on the
#    disturbance instead would leave the regressor describing an instantaneous
#    quantity and the target describing a filtered one -- a mismatch that shows
#    up as irreducible bias and reads as "the reduction does not hold here".
#  * The DIRECTION is public and the GAIN is unknown.  That is the honest split:
#    an agent can see where its neighbours pushed, and cannot see how much that
#    push costs it today.  It is also what makes the channel invertible -- knowing
#    the direction is what lets a scalar estimate cancel a vector disturbance.
#  * Every sum is strictly over ``j != i`` (P-3.1), so at N=1 every channel is
#    exactly zero, psi is [1, 0, ..., 0], and the disturbance is exactly zero at
#    any severity.  Category C, structurally.
#
#  torch only.  No vmas.

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from simple_ns.driver import DialParams, class_constants

__all__ = ["Coupling"]


class Coupling:
    """The declared operator, the filtered channels, and the basis."""

    def __init__(self, n_agents: int, p: DialParams, device=None) -> None:
        self.n = int(n_agents)
        self.p = p
        self.device = device
        self.r = int(p.n_types)
        recv, send = class_constants(p)
        self.recv = recv.to(device)
        self.send = send.to(device)
        # Class of agent i is i mod r.  Declared, public, and fixed for the run:
        # it is the actuator model the unit was built with, not something the
        # environment draws.
        self.type_of = torch.arange(self.n, device=device) % self.r
        # (r, N) one-hot over senders, so the per-class sum is one matmul
        self.is_type = (
            self.type_of.reshape(1, -1) == torch.arange(self.r, device=device).reshape(-1, 1)
        ).to(torch.float32)
        self._eye = torch.eye(self.n, device=device, dtype=torch.bool)

    # ------------------------------------------------------------------
    #  NS-1.2 -- the declared operator
    # ------------------------------------------------------------------

    def W(self, pos: Tensor) -> Tensor:
        """``W[b, i, j]`` -- how much of j's exertion reaches i.  ``(B, N, N)``.

            W_ij = recv_{type(i)} * 1 / (1 + (||p_i - p_j|| / lam)^2),   W_ii = 0

        The zero diagonal is **asserted, not argued**: it is what makes the
        estimated quantity a coupling rather than a self-effect, and it is what
        makes a lone agent read exactly zero at any severity.

        Asymmetric because the RECEIVER's compliance scales the row: ``W_ij`` and
        ``W_ji`` differ whenever the two agents are of different classes.  A flat
        proxy -- every neighbour equal -- was measured on the URB instance at a
        fit gain of -0.0045, i.e. worse than an intercept-only null, which is why
        both the distance falloff and the class factor are load-bearing.
        """
        d2 = (pos.unsqueeze(2) - pos.unsqueeze(1)).pow(2).sum(-1)  # (B,N,N)
        w = 1.0 / (1.0 + d2 / (self.p.kernel_lambda ** 2))
        w = w.masked_fill(self._eye, 0.0)
        return w * self.recv[self.type_of].reshape(1, -1, 1)

    # ------------------------------------------------------------------
    #  the filtered channels
    # ------------------------------------------------------------------

    def step_channels(
        self, pos: Tensor, u_prev: Tensor, Q_prev: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Advance the public channels one step.

        Args:
            pos:    ``(B, N, 2)`` current positions
            u_prev: ``(B, N, 2)`` the exertion each agent actually delivered last
                    step.  Peers' EXECUTED actions, which P-4.1 allows -- a
                    connected fleet broadcasts them anyway -- and never any peer's
                    residual.
            Q_prev: ``(B, N, r, 2)`` the filtered channels from last step

        Returns:
            ``(Q, q, ehat, x)`` -- the new filtered channels ``(B,N,r,2)``, their
            sum ``(B,N,2)``, the unit direction ``(B,N,2)`` and the scalar
            channels ``(B,N,r)``.
        """
        W = self.W(pos)  # (B,N,N)
        # per class m: sum_{j != i, type(j)=m} W_ij u_j
        #   (B,N,N) x (r,N) -> (B,N,r,N) would materialise N^2 r; instead mask W
        #   once per class, which is r matmuls of (B,N,N) @ (B,N,2).
        parts = []
        for m in range(self.r):
            Wm = W * self.is_type[m].reshape(1, 1, -1)
            parts.append(torch.bmm(Wm, u_prev))
        raw = torch.stack(parts, dim=2)  # (B,N,r,2)

        rho = self.p.rho
        Q = rho * Q_prev + (1.0 - rho) * raw
        q = Q.sum(dim=2)  # (B,N,2)
        norm = q.norm(dim=-1, keepdim=True)
        # An agent with no live neighbours reads an exactly zero direction, so
        # its disturbance is exactly zero rather than NaN.
        ehat = torch.where(norm > 1e-12, q / norm.clamp_min(1e-12), torch.zeros_like(q))
        x = (Q * ehat.unsqueeze(2)).sum(-1)  # (B,N,r)
        return Q, q, ehat, x

    def step_draw(
        self, pos: Tensor, u_prev: Tensor, Q_prev: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """The ``droop`` channel: each class's FLOW DEMAND reaching agent i.

            Q_m,i(t) = rho Q_m,i(t-1)
                       + (1-rho) * sum_{j != i, type(j)=m} W_ij * ||u_j(t-1)||

        A draw on a shared pressure rail is a SCALAR, not a vector: what costs
        you pressure is how much your neighbours are pulling, not which way they
        are pushing.  So unlike ``step_channels`` there is no direction to
        project onto -- the channels ARE the filtered draws, and the model is
        already linear in them.

        That also makes this channel simpler to identify than the additive one:
        no unit vector has to be estimated or agreed on, and the regressor cannot
        be degraded by a direction that happens to be near zero.

        Returns ``(Q, x)``, both ``(B, N, r)``.
        """
        W = self.W(pos)  # (B,N,N)
        draw = u_prev.norm(dim=-1)  # (B,N) -- the magnitude each peer is pulling
        parts = [
            torch.bmm(W * self.is_type[m].reshape(1, 1, -1), draw.unsqueeze(-1)).squeeze(-1)
            for m in range(self.r)
        ]
        raw = torch.stack(parts, dim=2)  # (B,N,r)
        Q = self.p.rho * Q_prev + (1.0 - self.p.rho) * raw
        return Q, Q

    def design(self, x: Tensor, ref: Tensor, scale: Tensor) -> Tensor:
        """``psi = [1, (x - ref) / scale]``.  ``(B, N, 1 + r)``.

        P-3.3: centre and scale on a geometric reference.  Raw channels carry a
        large common mean against an intercept column of 1; measured on the
        source implementation that gave a design-matrix condition number of
        ~1.3e5, at which the intercept and the class channels trade off and the
        per-class split is unidentifiable even though prediction is fine.
        Centring took it to ~24 and moved beta recovery from unidentifiable to
        0.001 error.
        """
        centred = (x - ref.reshape(1, 1, -1)) / scale.reshape(1, 1, -1).clamp_min(1e-9)
        ones = torch.ones_like(centred[..., :1])
        return torch.cat([ones, centred], dim=-1)

    # ------------------------------------------------------------------
    #  P-3.3 -- the geometric reference
    # ------------------------------------------------------------------

    def geometric_reference(
        self, pos_ref: Tensor, samples: int = 256, seed: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """``(ref, scale)``, each ``(r,)``.  P-3.3's centring reference.

        The channel each agent would see if every peer exerted uniformly at
        random from the SCENARIO'S OWN spawn geometry -- a function of declared
        structure only, so no run data enters.

        ``pos_ref`` has to be the host's own layout for the same reason
        ``load_norm`` does, and getting it wrong here is worse because it is
        silent.  With a uniform-over-arena draw the reference mean for channel 1
        came out at 0.011 with a scale of 0.020, while the actual runtime channel
        sat at 0.251 -- so the centred regressor was **psi = 11.8** instead of
        O(1).  The design matrix is then badly conditioned, the intercept and the
        class channels trade off, and beta comes out uncorrelated with the truth
        (measured cosine -0.29) while fit_gain still reads 0.86.  That is exactly
        P-3.3's warning -- "the split becomes unidentifiable even though
        prediction is fine" -- and it is why beta must be scored against the
        truth and not inferred from the fit.

        Computed with an explicit CPU generator so it cannot consume the run's
        RNG stream and cannot differ between arms.
        """
        gen = torch.Generator().manual_seed(seed)
        cpu = Coupling(self.n, self.p, device=None)
        pos_ref = pos_ref.detach().to("cpu", torch.float32)
        vals = []
        for k in range(samples):
            pos = pos_ref[k % pos_ref.shape[0]].unsqueeze(0)
            ang = torch.rand(1, self.n, generator=gen) * 2 * math.pi
            u = torch.stack([ang.cos(), ang.sin()], dim=-1)
            if self.p.channel == "droop":
                Q = torch.zeros(1, self.n, self.r)
                for _ in range(3):  # let the filtered channel reach steady state
                    Q, x = cpu.step_draw(pos, u, Q)
            else:
                Q = torch.zeros(1, self.n, self.r, 2)
                for _ in range(3):
                    Q, _, _, x = cpu.step_channels(pos, u, Q)
            vals.append(x.reshape(-1, self.r))
        V = torch.cat(vals, dim=0)
        ref = V.mean(dim=0)
        scale = V.std(dim=0, unbiased=False).clamp_min(1e-6)
        dev = self.recv.device
        return ref.to(dev), scale.to(dev)

    # ------------------------------------------------------------------
    #  NS-2.4 -- what sigma = 1 means, and why it must be normalised
    # ------------------------------------------------------------------

    def load_norm(
        self, pos_ref: Tensor, samples: int = 256, seed: int = 7
    ) -> float:
        """The reference value of ``sum_m send_m x_m`` at full peer exertion.

        Dividing the disturbance by this is what makes the severity dial MEAN
        something.  Without it the disturbance depends on the number of agents,
        the arena size and the kernel length scale, so sigma = 1 is a different
        physical severity in every host and the ladders cannot be read against
        each other -- which for a multi-environment paper is fatal, because a
        row that looks harsher would only be a row with more neighbours.

        Measured before this was added: the disturbance at sigma = 1 was 0.002 of
        the action range against the 0.14 the dial claims, i.e. the knob was
        nearly inert and would have read as "the method has nothing to recover".

        With it, ``sigma = 1`` is the declared statement in
        ``DialParams.loss_at_sigma1``: at the driver's peak, with every peer
        exerting its full action range, the disturbance reaching an agent is
        ``L`` of its own action range -- in every host, at every N.

        ``pos_ref`` is the SCENARIO'S OWN spawn distribution -- draws from its
        ``reset_world_at`` before any policy exists.  That is declared structure
        (where the scenario puts its agents) and not run data, and it has to be
        the reference rather than a uniform draw over the arena.

        Measured with a uniform-over-arena reference instead: ``balance`` puts
        its supports about 0.25 apart under a 0.8-long beam, while a uniform draw
        over [-1, 1]^2 separates them by about 1.0.  With a kernel length scale
        of 0.35 that is a factor of ~14 in transmitted force, so sigma = 1
        delivered 0.50 of the action range instead of the 0.035 the dial claims,
        20% of actions saturated against their own range, and the severity ladder
        stopped being monotone.  A dial whose meaning depends on how a host
        happens to arrange its agents is not a dial.

        Computed with an explicit CPU generator, so no arm's RNG stream is
        touched and every arm gets the same number.
        """
        gen = torch.Generator().manual_seed(seed)
        cpu = Coupling(self.n, self.p, device=None)
        send = cpu.send
        pos_ref = pos_ref.detach().to("cpu", torch.float32)
        tot = 0.0
        for k in range(samples):
            pos = pos_ref[k % pos_ref.shape[0]].unsqueeze(0)
            # full exertion, uniform direction: the reference condition named in
            # DialParams.loss_at_sigma1
            ang = torch.rand(1, self.n, generator=gen) * 2 * math.pi
            u = torch.stack([ang.cos(), ang.sin()], dim=-1)
            # the filtered channel at its steady state, which is what a sustained
            # exertion actually produces -- (1-rho) sum over one step understates
            # it by 1/(1-rho)
            if self.p.channel == "droop":
                Q = torch.zeros(1, self.n, self.r)
                for _ in range(3):
                    Q, x = cpu.step_draw(pos, u, Q)
            else:
                Q = torch.zeros(1, self.n, self.r, 2)
                for _ in range(3):
                    Q, _, _, x = cpu.step_channels(pos, u, Q)
            tot += float((send.reshape(1, 1, -1) * x).sum(-1).mean())
        ref = tot / samples
        if not (ref > 0):
            raise RuntimeError(
                "the reference load came out non-positive; the coupling is inert "
                "and sigma would have no meaning"
            )
        return ref

    # ------------------------------------------------------------------
    #  P-3.2 -- gate 1: the vectorised form must equal the definition
    # ------------------------------------------------------------------

    def verify(self, pos: Tensor, u_prev: Tensor, tol: float = 1e-5) -> str:
        """Check ``step_channels`` against a brute-force loop written straight
        off the definition, and abort on mismatch.

        Index order and self-exclusion are exactly the kind of wiring bug that
        leaves every diagnostic looking healthy, so this runs at startup rather
        than in a test file somebody can skip.
        """
        Q0 = torch.zeros(pos.shape[0], self.n, self.r, 2, device=pos.device)
        Q, q, ehat, x = self.step_channels(pos, u_prev, Q0)

        W = self.W(pos)
        slow = torch.zeros_like(Q)
        for b in range(pos.shape[0]):
            for i in range(self.n):
                for j in range(self.n):
                    if j == i:
                        continue  # P-3.1, written out
                    m = int(self.type_of[j])
                    slow[b, i, m] += W[b, i, j] * u_prev[b, j]
        slow = (1.0 - self.p.rho) * slow
        err = float((Q - slow).abs().max())
        if err > tol:
            raise RuntimeError(
                f"vectorised channels differ from the brute-force definition by "
                f"{err:.3e}. Index order or self-exclusion is wrong; every "
                "downstream diagnostic would still look healthy."
            )

        # a lone agent must read exactly zero on every channel
        one = Coupling(1, self.p, device=pos.device)
        _, _, e1, x1 = one.step_channels(
            pos[:, :1], u_prev[:, :1], torch.zeros(pos.shape[0], 1, self.r, 2, device=pos.device)
        )
        if float(x1.abs().max()) != 0.0 or float(e1.abs().max()) != 0.0:
            raise RuntimeError(
                "a lone agent read a non-zero channel; the sum is not strictly "
                "over j != i and this is category B in disguise"
            )
        return (
            f"channels == definition to {err:.2e}; N=1 reads exactly zero on "
            f"all {self.r} channels"
        )

    # ------------------------------------------------------------------
    #  reporting
    # ------------------------------------------------------------------

    def operator_stats(self, pos: Tensor) -> Dict[str, float]:
        """NS-1.2's three properties, measured rather than asserted."""
        W = self.W(pos)[0]
        off = W[~self._eye]
        off = off[off > 0]
        num = (W - W.T).abs()
        den = W + W.T
        mask = (~self._eye) & (den > 0)
        return {
            "diag_max": float(W.diag().abs().max()),
            "spread": float(off.std(unbiased=False) / off.mean().clamp_min(1e-30))
            if off.numel() > 1
            else 0.0,
            "asymmetry": float((num[mask] / den[mask]).mean()) if bool(mask.any()) else 0.0,
        }

    def banner(self) -> str:
        return (
            f"coupling        N={self.n} r={self.r} classes={self.type_of.tolist()}\n"
            f"                recv (public)  {[round(float(v), 3) for v in self.recv]}\n"
            f"                send (unknown) {[round(float(v), 3) for v in self.send]}\n"
            f"                kernel lambda={self.p.kernel_lambda} rho={self.p.rho}"
        )
