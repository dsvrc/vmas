#  Competent scripted controllers, for CALIBRATION only.
#
#  ---------------------------------------------------------------------------
#  Why this file exists
#  ---------------------------------------------------------------------------
#  ANT_complete_story.md 2.2 says the severity has to be chosen against a
#  controller that was actually trying:
#
#      "At severity sigma, could *anything* recover?  If the answer is no, a
#       failed training run tells you nothing.  You would be measuring the
#       environment, not the method."
#
#  ``transport`` and ``navigation`` ship a HeuristicPolicy good enough for that.
#  ``balance`` ships one that is not: measured at sigma = 0, with no disturbance
#  of any kind, it is **on the ground 61% of the time**, never reaches the goal,
#  and lets the package drift AWAY (1.570 -> 1.655).  Its return is pinned at a
#  failure floor, so the severity ladder reads -6.16 / -6.00 / -6.07 at
#  sigma = 0 / 1 / 4 -- noise around a constant, and exactly the "non-monotone"
#  result that made balance look like a bad host.
#
#  It is not a bad host.  It had a bad probe.  A host whose own controller cannot
#  do the task tells you nothing about a disturbance, because there is no
#  performance left to disturb.
#
#  These controllers are NOT baselines and must never be reported as one.  They
#  are instruments for choosing sigma before spending training compute, and they
#  read world state directly, which no policy may do.

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["BalanceController", "controller_for"]


class BalanceController:
    """A PD lift controller for ``vmas/balance``.

    The task: N supports stand under a beam carrying a package, and must raise it
    to a goal that is somewhere ABOVE them while keeping the beam level -- tip it
    and the package rolls off and the episode ends at -10.

    Three decoupled loops, which is how a real gantry or a multi-jack lift is
    actually controlled:

    * **lift** -- a common vertical command driving the beam's height to the
      goal's, plus a feed-forward term holding the static weight.  Without the
      feed-forward the loop has to run a standing error just to hold station,
      which on a 10-unit load under this gravity is most of the authority.
    * **level** -- a torque loop on the beam's angle and angular rate,
      distributed across the supports by lever arm.  Torque from support i is
      ``lever_i x F_i``, so a desired torque is met by adding
      ``tau * lever_i / sum(lever^2)`` to each support's vertical command.  This
      is the loop the disturbance attacks.
    * **track** -- a horizontal command that keeps each support under its own
      slot on the beam while walking the whole beam toward the goal's x.

    Gains are declared here and not tuned per severity: a probe whose gains moved
    with sigma would be measuring the probe.
    """

    def __init__(
        self,
        k_lift: float = 3.0,
        d_lift: float = 1.2,
        ff: float = 0.32,
        k_level: float = 6.0,
        d_level: float = 1.5,
        k_track: float = 1.5,
        d_track: float = 0.8,
        k_slot: float = 4.0,
    ) -> None:
        self.k_lift, self.d_lift, self.ff = k_lift, d_lift, ff
        self.k_level, self.d_level = k_level, d_level
        self.k_track, self.d_track, self.k_slot = k_track, d_track, k_slot

    def __call__(self, sc) -> list:
        line, pkg = sc.line, sc.package
        goal = pkg.goal.state.pos  # (B, 2)
        lp, lv = line.state.pos, line.state.vel
        theta = line.state.rot[:, 0]
        omega = line.state.ang_vel[:, 0]

        pos = torch.stack([a.state.pos for a in sc.world.agents], dim=1)  # (B,N,2)
        vel = torch.stack([a.state.vel for a in sc.world.agents], dim=1)
        n = pos.shape[1]

        # -- lift: drive the beam's height to the goal's, and hold the weight ---
        err_y = goal[:, 1] - lp[:, 1]
        u_common = self.k_lift * err_y - self.d_lift * lv[:, 1] + self.ff  # (B,)

        # -- level: a torque loop, distributed by lever arm --------------------
        # each support's nominal slot along the beam, evenly spaced
        slot = (
            torch.linspace(-0.5, 0.5, n, device=pos.device) * sc.line_length
        ).reshape(1, -1)  # (1,N)
        tau = -self.k_level * theta - self.d_level * omega  # (B,)
        u_lift = u_common.unsqueeze(-1) + tau.unsqueeze(-1) * slot / (
            (slot**2).sum().clamp_min(1e-9)
        )

        # -- track: stay under your slot, and walk the beam toward the goal ----
        c, s = torch.cos(theta), torch.sin(theta)
        slot_x = lp[:, 0:1] + slot * c.unsqueeze(-1)
        u_x = (
            self.k_track * (goal[:, 0:1] - lp[:, 0:1])
            - self.d_track * lv[:, 0:1]
            + self.k_slot * (slot_x - pos[..., 0])
        )

        u = torch.stack([u_x, u_lift], dim=-1)  # (B,N,2)
        lim = sc.world.agents[0].u_range
        lim = lim[0] if isinstance(lim, (list, tuple)) else lim
        u = u.clamp(-float(lim), float(lim))
        return [u[:, i] for i in range(n)]


def controller_for(host: str, sc):
    """A callable ``sc -> [action per agent]`` for ``host``.

    ``balance`` gets the controller above; everything else gets the scenario's
    own shipped HeuristicPolicy, wrapped so the call looks the same.
    """
    if host == "balance":
        ctl = BalanceController()
        return lambda scen: ctl(scen)

    mod = __import__(f"vmas.scenarios.{host}", fromlist=["HeuristicPolicy"])
    pol = mod.HeuristicPolicy(continuous_action=True)

    def run(scen):
        obs = [scen.observation(a) for a in scen.world.agents]
        return [
            pol.compute_action(obs[i], u_range=a.u_range)
            for i, a in enumerate(scen.world.agents)
        ]

    return run
