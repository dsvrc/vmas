#  ``lanelet_flow`` -- the same medium, an affordable host.
#
#  ---------------------------------------------------------------------------
#  Why this file exists
#  ---------------------------------------------------------------------------
#  Measured on the CPM map, CPU, random actions, via ``vmas.make_env``:
#
#      host                                  N    envs   ms/frame
#      vmas/road_traffic                     16     64      29.8
#      vmas/road_traffic                     40     16     173
#      vmas/road_traffic, done() truncating  40     16     183   <- no gain
#      road_ns medium layer ALONE            40    600       0.16
#
#  1.2M frames at 173 ms/frame is 58 hours for ONE arm of ONE seed.  The cost is
#  not the severity layer -- that is 0.09% of it -- and it is not the
#  collision-reset storm either: overriding ``done`` to truncate changed
#  nothing.  It is ``road_traffic``'s per-agent Python geometry: the O(N^2)
#  ``interX`` loop with a device sync per pair, five boundary distances per
#  agent per step, and the short-term reference-path resampling.  That is most
#  of its 4035 lines and no subclass can reach it.
#
#  So this host keeps everything the SPEC actually asks for and throws away the
#  part that has nothing to do with the claim:
#
#      KEPT   the CPM Lab CommonRoad map, parsed by road_ns/structure.py
#             capacity_a = lanes * length / (vehicle + IDM jam gap)
#             W = lanelet-route incidence / capacity      (NS-1.2)
#             the seven reference loops road_traffic itself declares
#             the HCM heavy-rain anchor and the whole dial                (I.3)
#             the harm channel: divide the achievable speed               (NS-1.4)
#             KinematicBicycle -- the SAME vehicle model road_traffic uses
#
#      REPLACED   collision meshes -> vectorised centre-to-centre distances
#                 5-point boundary distance -> signed offset from the centre line
#                 short-term path resampling -> a gather off the progress index
#                 rejection-sampled per-env reset -> one vectorised draw
#                 terminate-on-collision -> truncation + vectorised respawn
#
#  Nothing here loops over agents in Python except where VMAS's own API forces
#  it (``process_action``/``observation``/``reward`` are called per agent), and
#  every one of those is a slice of a tensor computed once per step.
#
#  ---------------------------------------------------------------------------
#  What this costs the story, stated plainly
#  ---------------------------------------------------------------------------
#  You can no longer say "we run SigmaRL's published scenario".  You can still
#  say every load-bearing thing: the medium is a third-party map, the capacities
#  come off it, the operator is written down before any run, and sigma = 1 is a
#  published constant.  Questions Q1, Q2 and Q4 of III.1 are answered by the
#  MAP, not by the vehicle model.  ``road_ns/road_traffic`` stays registered so
#  the real scenario can be reported as a provenance row at low N.

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Box, World
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario

from road_ns.structure import load_structure

__all__ = ["LaneletFlow", "FLOW_KWARGS"]


#: Host knobs.  Popped before the world is built, exactly as the dial's are.
FLOW_KWARGS = (
    "n_agents",
    "n_nearing_agents_observed",
    "flow_n_samples",
    "flow_n_lookahead",
    "flow_lookahead_stride",
    "flow_search_window",
    "flow_respawn",
    "flow_reroute_on_lap",
    "flow_n_background",
    "flow_background_speed",
    "flow_integration",
    "dt",
    "max_speed",
)

# road_traffic's own constants, kept so the two hosts' returns are on the same
# scale.  r_p_normalizer = 100 there; these are the already-divided values.
R_PROGRESS = 10 / 100
R_VEL = 5 / 100
P_DEVIATE = -2 / 100
P_NEAR_BOUNDARY = -20 / 100
P_NEAR_AGENTS = -20 / 100
P_COLLIDE_AGENTS = -100 / 100
P_COLLIDE_BOUNDARY = -100 / 100
P_CHANGE_STEERING = -2 / 100
P_TIME = 5 / 100

#: ``path_to_loop`` has 40 slots, and ``path_id`` indexes them.
N_REFERENCE_PATHS = 40


def _exp_decreasing(x: Tensor, x0: float, x1: float) -> Tensor:
    """``road_traffic.exponential_decreasing_fcn``, inlined.

    1 at ``x0``, ~0 at ``x1``, 0 outside.  Same shape, so the proximity penalty
    has the same profile as the host it replaces.
    """
    out = (torch.exp(-(x - x0) / (x1 - x0) * 3.5) - math.exp(-3.5)) / (
        1 - math.exp(-3.5)
    )
    return out.clamp(0.0, 1.0)


class LaneletFlow(BaseScenario):
    """N vehicles driving the CPM map's own reference loops.

    The agent's job is the one URB measures: get round your route, in traffic,
    without leaving the lane or hitting anyone.  Reward is progress along the
    route minus time, with the same proximity, deviation and collision terms
    ``road_traffic`` uses and the same coefficients.
    """

    # ------------------------------------------------------------------
    #  construction
    # ------------------------------------------------------------------

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents = int(kwargs.pop("n_agents", 16))
        self.k_peers = int(kwargs.pop("n_nearing_agents_observed", 4))
        n_samples = int(kwargs.pop("flow_n_samples", 512))
        self.n_lookahead = int(kwargs.pop("flow_n_lookahead", 3))
        self.look_stride = int(kwargs.pop("flow_lookahead_stride", 6))
        self.window = int(kwargs.pop("flow_search_window", 12))
        self.do_respawn = bool(kwargs.pop("flow_respawn", True))
        self.reroute_on_lap = bool(kwargs.pop("flow_reroute_on_lap", True))
        self.n_background = int(kwargs.pop("flow_n_background", 0))
        self.bg_speed = float(kwargs.pop("flow_background_speed", 0.6))
        integration = str(kwargs.pop("flow_integration", "rk4"))
        dt = float(kwargs.pop("dt", 0.05))
        self.max_speed = float(kwargs.pop("max_speed", 1.0))

        # road_traffic's geometry, so the vehicle is the same vehicle
        self.agent_length = 0.16
        self.agent_width = 0.08
        self.l_f = self.agent_length / 2
        self.l_r = self.agent_length - self.l_f
        self.max_steering = float(torch.deg2rad(torch.tensor(35.0)))
        self.world_x_dim, self.world_y_dim = 4.5, 4.0

        # The severity layer builds these before calling us when it is stacked
        # on top; standalone, we build them ourselves.
        if not hasattr(self, "struct"):
            self.struct = load_structure().to(device)
            self.routes = self.struct.declared_routes(
                getattr(self, "_route_set", "loops")
            )

        geo = self.struct.route_geometry(self.routes, n_samples=n_samples)
        self.K = n_samples
        self.n_routes = len(self.routes)
        self._point = geo["point"].to(device)  # (P, K, 2)
        self._yaw = geo["yaw"].to(device)  # (P, K)
        self._halfw = geo["half_width"].to(device)  # (P, K)
        self._elem = geo["element"].to(device)  # (P, K)
        self._spacing = geo["spacing"].to(device)  # (P,)
        # flattened, so every lookup is one index_select instead of building a
        # (B, N, K, 2) gather source
        self._point_flat = self._point.reshape(-1, 2)
        self._yaw_flat = self._yaw.reshape(-1)
        self._halfw_flat = self._halfw.reshape(-1)
        self._elem_flat = self._elem.reshape(-1)

        # road_traffic's own agent -> loop table, so the fleet layout is the
        # scenario's rather than one invented here.
        from road_ns.structure import PATH_TO_LOOP

        self._loop_table = torch.tensor(
            [x - 1 for x in PATH_TO_LOOP], device=device, dtype=torch.long
        )

        world = World(
            batch_dim,
            device,
            x_semidim=self.world_x_dim,
            y_semidim=self.world_y_dim,
            dt=dt,
        )
        for i in range(self.n_agents):
            world.add_agent(
                Agent(
                    name=f"agent_{i}",
                    shape=Box(length=self.agent_length, width=self.agent_width),
                    # collide=False exactly as road_traffic has it: collisions
                    # there are reward bookkeeping, not physics, and keeping that
                    # means the harm channel is the only thing acting on motion.
                    collide=False,
                    render_action=False,
                    u_range=[self.max_speed, self.max_steering],
                    u_multiplier=[1, 1],
                    max_speed=self.max_speed,
                    dynamics=KinematicBicycle(
                        world,
                        width=self.agent_width,
                        l_f=self.l_f,
                        l_r=self.l_r,
                        max_steering_angle=self.max_steering,
                        integration=integration,
                    ),
                )
            )

        B, N = batch_dim, self.n_agents
        f = dict(device=device, dtype=torch.float32)
        li = dict(device=device, dtype=torch.long)
        self._route_of = torch.zeros(B, N, **li)
        self._elem_of = torch.zeros(B, N, **li)
        self._prog = torch.zeros(B, N, **li)
        self._prog_prev = torch.zeros(B, N, **li)
        self._lat = torch.zeros(B, N, **f)
        self._head_err = torch.zeros(B, N, **f)
        self._dist = torch.zeros(B, N, N, **f)
        self._peer_idx = torch.zeros(B, N, max(1, min(self.k_peers, N - 1)), **li)
        self._steer_prev = torch.zeros(B, N, **f)
        self._d_steer = torch.zeros(B, N, **f)
        self._pending: Optional[Tensor] = None
        self._collided = torch.zeros(B, N, device=device, dtype=torch.bool)
        self._offroad = torch.zeros(B, N, device=device, dtype=torch.bool)
        self._lap = torch.zeros(B, N, device=device, dtype=torch.bool)
        M = self.n_background
        self._bg_route = torch.zeros(B, M, **li)
        self._bg_prog = torch.zeros(B, M, **f)
        self._device = device

        # The host draws from its OWN generator, never from the global stream.
        #
        # Respawn and re-routing are environment stochasticity, and they must not
        # depend on how many times anything else drew.  Sharing the global RNG
        # makes them depend on the POLICY: two arms that are supposed to be
        # bit-identical (sigma=0, the placebo, trust=0) diverge the moment their
        # policies consume a different number of samples, and the P-7.1 evidence
        # evaporates for a reason that has nothing to do with P-7.1.
        #
        # Seeded ONCE, on the first full reset.  VMAS runs reset and step inside
        # `local_seed(Environment.vmas_random_state)`, and that state is a CLASS
        # attribute shared by every env in the process -- so it drifts as other
        # envs (the evaluation env, the other arm) step.  The one moment it is
        # pinned to this run's seed is `Environment.__init__`, which calls
        # `_seed(seed)` immediately before the first full reset.  Taking the
        # generator's seed there and never again makes the host's whole stream a
        # deterministic function of the run's seed.
        self._gen = torch.Generator(device=device)
        self._seeded = False

        # thresholds, same definitions as road_traffic
        self._near_agent_hi = self.agent_length + self.agent_width
        self._near_agent_lo = (self.agent_length + self.agent_width) / 2
        self._collide_dist = self.agent_width  # centre-to-centre proxy for overlap
        self._lane_half = float(self._halfw.max())
        self._near_b_hi = (2 * self._lane_half - self.agent_width) / 2 * 0.9
        self._deviate_w = self._lane_half

        print(
            f"lanelet_flow    N={self.n_agents} routes={self.n_routes} "
            f"samples={self.K} spacing in "
            f"[{float(self._spacing.min()):.4f}, {float(self._spacing.max()):.4f}] m "
            f"(max travel/step {self.max_speed * dt:.3f} m) "
            f"peers={self.k_peers} respawn={self.do_respawn} "
            f"background={self.n_background}@{self.bg_speed}m/s"
        )
        return world

    # ------------------------------------------------------------------
    #  lifecycle -- fully vectorised, no per-env python
    # ------------------------------------------------------------------

    def _reseed(self) -> None:
        if self._seeded:
            return
        self._gen.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        self._seeded = True

    def _randint(self, hi: int, shape) -> Tensor:
        return torch.randint(0, hi, shape, device=self._device, generator=self._gen)

    def _rand(self, *shape) -> Tensor:
        return torch.rand(*shape, device=self._device, generator=self._gen)

    def _place(self, route: Tensor, slot: Tensor):
        """``(b, n)`` route and sample index -> position, heading."""
        flat = route * self.K + slot
        return self._point_flat[flat], self._yaw_flat[flat]

    def reset_world_at(
        self, env_index: Optional[int] = None, agent_index: Optional[int] = None
    ):
        """One ``randint`` for the whole fleet.

        ``road_traffic`` draws a path and a start point per agent per env and
        REJECTS until the agent is far enough from every already-placed one --
        a python ``while`` loop, 40 of them per env, re-run on every reset.
        Here each agent gets its own arc-length band on its own route, so the
        placement is collision-free by construction and the whole reset is three
        tensor ops.
        """
        B, N = self.world.batch_dim, self.n_agents
        b = B if env_index is None else 1
        sl = slice(None) if env_index is None else slice(env_index, env_index + 1)
        dev = self._device

        if env_index is None:
            self._reseed()
        pid = self._randint(N_REFERENCE_PATHS, (b, N))
        route = self._loop_table[pid]
        band = self.K // N
        slot = (
            torch.arange(N, device=dev).unsqueeze(0) * band
            + self._randint(max(1, band), (b, N))
        ) % self.K

        pos, yaw = self._place(route, slot)
        speed = self._rand(b, N) * self.max_speed
        vel = torch.stack([speed * yaw.cos(), speed * yaw.sin()], dim=-1)

        for i, ag in enumerate(self.world.agents):
            if env_index is None:
                ag.set_pos(pos[:, i], batch_index=None)
                ag.set_rot(yaw[:, i : i + 1], batch_index=None)
                ag.set_vel(vel[:, i], batch_index=None)
            else:
                ag.set_pos(pos[0, i], batch_index=env_index)
                ag.set_rot(yaw[0, i : i + 1], batch_index=env_index)
                ag.set_vel(vel[0, i], batch_index=env_index)

        self._route_of[sl] = route
        self._prog[sl] = slot
        self._prog_prev[sl] = slot
        self._elem_of[sl] = self._elem_flat[route * self.K + slot]
        self._lat[sl] = 0.0
        self._head_err[sl] = 0.0
        self._steer_prev[sl] = 0.0
        self._collided[sl] = False
        self._offroad[sl] = False
        self._lap[sl] = False

        if self.n_background:
            M = self.n_background
            bpid = self._randint(N_REFERENCE_PATHS, (b, M))
            self._bg_route[sl] = self._loop_table[bpid]
            self._bg_prog[sl] = self._rand(b, M) * self.K

        # `Environment._reset` asks for observations immediately, so the
        # read-out has to describe the state we just placed.  A partial reset
        # refreshes the whole fleet: the other worlds are unchanged, so
        # recomputing them costs a step and changes nothing.
        self._steer_prev[sl] = 0.0
        self._d_steer[sl] = 0.0
        self._pending = None
        self._refresh()

    def _respawn(self, mask: Tensor) -> None:
        """Re-place the agents in ``mask`` -- vectorised, episode continues.

        This is ``road_traffic``'s ``is_testing_mode`` behaviour (reset only who
        crashed, not the world) minus the python loop over (env, agent) pairs.
        It is also what keeps the medium populated: terminating the whole env on
        any collision gave episodes ~1 step long at N=40, far too short for
        congestion to build at all.

        A respawned agent draws a NEW route.  That is the honest reading of the
        event -- a vehicle finishing a trip and starting another -- and it is
        also the only source of route turnover in the episode, which the
        estimator needs: with the fleet layout frozen, every agent's basis row
        is constant for the whole episode and there is nothing to identify.
        """
        if not bool(mask.any()):
            return
        B, N = mask.shape
        pid = self._randint(N_REFERENCE_PATHS, (B, N))
        route = torch.where(mask, self._loop_table[pid], self._route_of)
        slot = torch.where(mask, self._randint(self.K, (B, N)), self._prog)
        pos, yaw = self._place(route, slot)
        speed = self._rand(B, N) * self.max_speed
        vel = torch.stack([speed * yaw.cos(), speed * yaw.sin()], dim=-1)
        m = mask.unsqueeze(-1)
        for i, ag in enumerate(self.world.agents):
            ag.state.pos = torch.where(m[:, i], pos[:, i], ag.state.pos)
            ag.state.rot = torch.where(m[:, i], yaw[:, i : i + 1], ag.state.rot)
            ag.state.vel = torch.where(m[:, i], vel[:, i], ag.state.vel)
        self._route_of = route
        self._prog = slot
        self._prog_prev = torch.where(mask, slot, self._prog_prev)
        self._elem_of = self._elem_flat[route * self.K + slot]

    # ------------------------------------------------------------------
    #  background demand -- I.5's Delta_fixed, made real
    # ------------------------------------------------------------------

    def _advance_background(self) -> None:
        """Vehicles nobody controls, driving their route at a fixed pace.

        They exist for three reasons, all structural:

        * I.5 partitions the loading excess into fixed / own / peer, and
          ``Delta_fixed`` is "load from participants no agent controls".  With
          an all-agent fleet that term is zero by construction, the irreducible
          share is 0% at every fleet size, and NS-4.2's prediction -- that the
          coordination gap RISES with controllable share because irreducible
          load disappears while peer load does not -- cannot be tested at all.
          The scripts used to sweep ``n_agents`` and call that a controllable-
          share sweep; it is not, it is a fleet-SIZE sweep at 100% controllable.
        * The medium has to be loaded enough for a capacity loss to matter
          (III.1 Q7).  Background demand raises ``u`` without adding agents, so
          the operating point and the fleet size stop being the same knob.
        * The PACT basis sums strictly over controllable peers, so background
          load is excess the estimator structurally CANNOT explain.  That is
          correct and is what makes the intercept mean something: it is the
          irreducible share, and no method may claim it.

        They are demand, not obstacles: they enter the element load and nothing
        else.  ``road_traffic``'s own agents are ``collide=False``, so a vehicle
        that is not an obstacle is exactly as physical as one that is.
        """
        if not self.n_background:
            return
        step = self.bg_speed * self.world.dt / self._spacing[self._bg_route]
        self._bg_prog = (self._bg_prog + step) % self.K

    def _ns_background_load(self, n_elements: int) -> Optional[Tensor]:
        """``(B, A)`` occupancy contributed by the background, or None."""
        if not self.n_background:
            return None
        here = self._elem_flat[
            self._bg_route * self.K + self._bg_prog.long().clamp(0, self.K - 1)
        ]
        load = torch.zeros(
            here.shape[0], n_elements, device=here.device, dtype=torch.float32
        )
        load.scatter_add_(1, here, torch.ones_like(here, dtype=load.dtype))
        return load

    # ------------------------------------------------------------------
    #  per-step state, computed ONCE for the whole fleet
    # ------------------------------------------------------------------

    def _positions(self) -> Tensor:
        return torch.stack([a.state.pos for a in self.world.agents], dim=1)

    def _cache_kinematics(self) -> None:
        """Stack pose and velocity ONCE per step.

        ``observation`` and ``reward`` are called once PER AGENT, so anything
        fleet-wide done inside them is done N times: the per-agent read-out used
        to re-stack all N positions and all N velocities on every call, which is
        O(N^2) python per step and was 43% of the whole step in the profile.
        """
        self._pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        self._vel = torch.stack([a.state.vel for a in self.world.agents], dim=1)
        self._rot = torch.stack([a.state.rot[:, 0] for a in self.world.agents], dim=1)
        self._cos, self._sin = self._rot.cos(), self._rot.sin()

    def _localise(self, pos: Tensor) -> None:
        """Progress, lateral offset and heading error, for every agent at once.

        The progress index is tracked INCREMENTALLY: a vehicle moves at most
        ``max_speed * dt`` = 0.05 m per step against a sample spacing of
        0.017-0.028 m, so a +-12 sample window around the previous index always
        contains the true nearest point.  That turns an O(K) nearest-neighbour
        search over 512 samples into a gather of 25.
        """
        W = self.window
        offs = torch.arange(-W, W + 1, device=pos.device)
        cand = (self._prog.unsqueeze(-1) + offs) % self.K  # (B,N,2W+1)
        flat = self._route_of.unsqueeze(-1) * self.K + cand
        pts = self._point_flat[flat]  # (B,N,2W+1,2)
        d2 = (pts - pos.unsqueeze(2)).pow(2).sum(-1)
        best = d2.argmin(-1, keepdim=True)
        prog = cand.gather(-1, best).squeeze(-1)  # (B,N)

        # exact lateral offset: project onto the segment [prog, prog+1]
        i0 = self._route_of * self.K + prog
        i1 = self._route_of * self.K + (prog + 1) % self.K
        p0, p1 = self._point_flat[i0], self._point_flat[i1]
        seg = p1 - p0
        seg2 = seg.pow(2).sum(-1).clamp_min(1e-12)
        t = (((pos - p0) * seg).sum(-1) / seg2).clamp(0.0, 1.0)
        rel = pos - (p0 + t.unsqueeze(-1) * seg)
        self._lat = rel.norm(dim=-1)

        yaw = self._yaw_flat[i0]
        rot = self._rot
        self._head_err = torch.atan2(
            torch.sin(rot - yaw), torch.cos(rot - yaw)
        )  # wrapped to [-pi, pi]

        # a lap completed = the index wrapped backwards by more than half a loop
        self._lap = (prog.long() - self._prog.long()) < -(self.K // 2)
        self._prog_prev, self._prog = self._prog, prog
        self._elem_of = self._elem_flat[i0]

    def _pairwise(self, pos: Tensor) -> None:
        B, N = pos.shape[0], pos.shape[1]
        d = torch.cdist(pos, pos)
        eye = torch.eye(N, device=pos.device, dtype=torch.bool)
        self._dist = d.masked_fill(eye, float("inf"))
        k = self._peer_idx.shape[-1]
        self._peer_idx = self._dist.topk(k, dim=-1, largest=False).indices
        self._collided = self._dist.min(dim=-1).values < self._collide_dist
        self._offroad = self._lat > self._halfw_flat[
            self._route_of * self.K + self._prog
        ]

    def _refresh(self) -> None:
        """Localise the fleet and rebuild the read-out.  POST-physics.

        ``vmas.Environment.step`` runs
        ``env_process_action -> world.step -> post_step -> observation/reward``,
        so this belongs in ``post_step``: that is the only point where the state
        the agent is told about is the state it is actually in.

        It used to run in ``process_action``, i.e. before the physics, and three
        things were wrong because of it.  The observation reported post-step
        POSITIONS against a pre-step progress index and lateral offset.  The
        progress reward measured the PREVIOUS step's movement, so the last step
        of every episode was never paid.  And the steering-rate penalty compared
        this step's steering against itself and was therefore identically zero.
        """
        self._cache_kinematics()
        self._localise(self._pos)
        self._pairwise(self._pos)
        self._build_readout()

    # VMAS calls process_action once per agent, in order.  Agent 0 is the hook.
    def process_action(self, agent: Agent) -> None:
        if agent is self.world.agents[0]:
            # Deferred from the previous step so that the collision and
            # off-road penalties were actually paid before the vehicle was
            # moved, and so the read-out described the state that earned them.
            if self._pending is not None:
                self._respawn(self._pending)
                self._pending = None
            self._advance_background()

    def post_step(self) -> None:
        cur = torch.stack([a.action.u[:, 1] for a in self.world.agents], dim=1)
        self._d_steer = (cur - self._steer_prev).abs()
        self._steer_prev = cur
        self._refresh()
        if self.do_respawn:
            mask = self._collided | self._offroad
            if self.reroute_on_lap:
                mask = mask | self._lap
            self._pending = mask if bool(mask.any()) else None

    # ------------------------------------------------------------------
    #  read-out
    # ------------------------------------------------------------------

    def _to_ego(self, vec: Tensor) -> Tensor:
        """Rotate world-frame ``(B, N, ..., 2)`` into each agent's body frame.

        One cos/sin for the fleet, reused for every field.  The per-agent version
        recomputed them on all four of its calls, 4N times a step.
        """
        extra = vec.dim() - 3
        shape = self._cos.shape + (1,) * extra
        c, s = self._cos.reshape(shape), self._sin.reshape(shape)
        x, y = vec[..., 0], vec[..., 1]
        return torch.stack([c * x + s * y, -s * x + c * y], dim=-1)

    def _build_readout(self) -> None:
        """The whole fleet's observation and reward, in ONE pass.

        VMAS asks for these one agent at a time, so the per-agent versions are
        now pure slices of what this computes.  Same numbers, N times less
        python and N times fewer kernel launches.
        """
        pos, vel = self._pos, self._vel
        K = self.K
        route, prog = self._route_of, self._prog
        i0 = route * K + prog
        hw = self._halfw_flat[i0]

        # -- observation ----------------------------------------------------
        own = self._to_ego(vel) / self.max_speed  # (B,N,2)

        steps = torch.arange(
            1, self.n_lookahead + 1, device=pos.device
        ) * self.look_stride
        idx = route.unsqueeze(-1) * K + (prog.unsqueeze(-1) + steps) % K  # (B,N,L)
        ahead = self._point_flat[idx] - pos.unsqueeze(2)  # (B,N,L,2)
        ahead = self._to_ego(ahead).flatten(2) / self._lane_half

        lat = (self._lat / hw).unsqueeze(-1)
        head = (self._head_err / math.pi).unsqueeze(-1)

        # (B,N,N,2): [b,i,j] is peer j seen from i
        rel_all = pos.unsqueeze(1) - pos.unsqueeze(2)
        relv_all = vel.unsqueeze(1) - vel.unsqueeze(2)
        pick = self._peer_idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B,N,k,2)
        rel = torch.gather(rel_all, 2, pick)
        relv = torch.gather(relv_all, 2, pick)
        d = torch.gather(self._dist, 2, self._peer_idx)  # (B,N,k)
        # II.2's masking of distant peers, as road_traffic does it
        seen = (d < self._near_agent_hi * 4).to(pos.dtype).unsqueeze(-1)
        peer = torch.cat(
            [
                (self._to_ego(rel) * seen).flatten(2) / self._lane_half,
                (self._to_ego(relv) * seen).flatten(2) / self.max_speed,
            ],
            dim=-1,
        )
        self._obs = torch.cat([own, ahead, lat, head, peer], dim=-1)  # (B,N,D)

        # -- reward ---------------------------------------------------------
        # progress along the route, wrap-safe, normalised by the most a vehicle
        # could possibly travel in one step -- road_traffic's own normalisation
        d_idx = (prog.long() - self._prog_prev.long() + K // 2) % K - K // 2
        metres = d_idx.to(pos.dtype) * self._spacing[route]
        rew = metres / (self.max_speed * self.world.dt) * R_PROGRESS

        # speed, projected on the route direction
        v = vel.norm(dim=-1)
        v_proj = v * self._head_err.cos()
        rew = rew + torch.where(v_proj > 0, 1.0, 2.0) * v_proj / self.max_speed * R_VEL

        # too close to peers.  The diagonal is +inf, so its term is exactly 0.
        near = _exp_decreasing(self._dist, self._near_agent_lo, self._near_agent_hi)
        rew = rew + near.nan_to_num(0.0).sum(dim=-1) * P_NEAR_AGENTS

        # too close to the lane boundary, and off it
        clearance = (hw - self.agent_width / 2 - self._lat).clamp_min(0.0)
        rew = rew + _exp_decreasing(clearance, 0.0, self._near_b_hi) * P_NEAR_BOUNDARY
        rew = rew + self._offroad.to(pos.dtype) * P_COLLIDE_BOUNDARY

        # deviation from the centre line
        rew = rew + self._lat / self._deviate_w * P_DEVIATE

        # steering rate, against the PREVIOUS step's steering
        rew = rew + (self._d_steer / (2 * self.max_steering)) * P_CHANGE_STEERING

        # collision
        rew = rew + self._collided.to(pos.dtype) * P_COLLIDE_AGENTS

        # time: paid for moving forward, charged for moving backward
        rew = rew + torch.where(v_proj > 0, 1.0, -1.0) * v / self.max_speed * P_TIME
        self._rew = rew  # (B,N)

        # -- info ------------------------------------------------------------
        # Built once for the fleet and sliced per agent, for the same reason the
        # observation is: the dtype casts below allocate, and doing them inside
        # `info` did it N times a step.
        self._info = {
            "flow_lateral": self._lat,
            "flow_offroad": self._offroad.to(pos.dtype),
            "flow_collided": self._collided.to(pos.dtype),
            "flow_speed": v,
            "flow_route": self._route_of.to(pos.dtype),
        }

    def observation(self, agent: Agent):
        return self._obs[:, self.world.agents.index(agent)]

    def reward(self, agent: Agent):
        return self._rew[:, self.world.agents.index(agent)]

    def done(self):
        """Truncation only.

        ``road_traffic`` ends the episode the moment ANY agent touches ANY other
        agent or boundary.  At N=40 with an untrained policy that was measured
        at 94% of env-steps terminating -- episodes about one step long, which
        is no horizon at all for a congestion study, and the agent's return is
        then almost entirely "did I crash" rather than "did I get through the
        traffic".  Collisions are still penalised, and the agents involved are
        respawned; the world keeps running.
        """
        return torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.bool
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        i = self.world.agents.index(agent)
        return {k: v[:, i : i + 1] for k, v in self._info.items()}
