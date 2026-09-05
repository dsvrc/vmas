#  Load the SLC / PACT arithmetic WITHOUT importing the RL stack.
#
#  ``slc_core`` and ``pact_core`` depend on nothing but torch.  That is
#  deliberate: the arithmetic self-check and the Part-C ceiling decomposition
#  must run on a laptop with no vmas, no torchrl and no hydra, so that the
#  environment questions are settled before any GPU time is spent.  But they
#  live inside the ``benchmarl.environments.vmas_slc`` package, and importing
#  them the ordinary way would execute ``benchmarl/__init__.py`` and pull in the
#  whole stack.
#
#  So we register empty namespace packages for the parents and load the two
#  modules straight from their files.  The modules are registered under their
#  real dotted names, so ``pact_core``'s absolute import of ``slc_core``
#  resolves to the same object the environment uses -- there is exactly one copy
#  of the arithmetic and it cannot drift.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG = "benchmarl.environments.vmas_slc"
_SRC = REPO_ROOT / "benchmarl" / "environments" / "vmas_slc"


def _ensure_namespace(name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []  # namespace package
    sys.modules[name] = module


def _load(dotted: str, path: Path):
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {dotted} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


def load_cores():
    """Returns ``(slc_core, pact_core)``, torch-only."""
    real = sys.modules.get(_PKG)
    if real is not None and getattr(real, "__file__", None):
        # the real package is already imported (we are inside a training run);
        # use it rather than shadowing it
        from benchmarl.environments.vmas_slc import pact_core, slc_core  # noqa

        return slc_core, pact_core

    for parent in ("benchmarl", "benchmarl.environments", _PKG):
        _ensure_namespace(parent)
    slc_core = _load(f"{_PKG}.slc_core", _SRC / "slc_core.py")
    pact_core = _load(f"{_PKG}.pact_core", _SRC / "pact_core.py")
    return slc_core, pact_core


# ---------------------------------------------------------------------------
#  A torch-only surrogate of VMAS holonomic dynamics.
#
#  Mirrors ``World._integrate_state`` exactly: drag is applied once per step
#  (substep 0 only), then ``v += (f/m) * sub_dt`` per substep, then
#  ``p += v * sub_dt``.  Terminal speed at unit force is
#  ``u * dt / (m * drag) = 0.4``, which is where ``slc_v_ref`` comes from.
#
#  It exists so the environment questions -- operating scale, the ceiling
#  decomposition, the sigma frontier -- can be answered before touching the
#  server.  It cannot answer the learning questions and does not try to.
# ---------------------------------------------------------------------------

import torch  # noqa: E402


class HolonomicFleet:
    """Vectorised double integrator with VMAS's drag convention."""

    def __init__(
        self,
        batch: int,
        n_agents: int,
        device="cpu",
        dt: float = 0.1,
        drag: float = 0.25,
        substeps: int = 1,
        mass: float = 1.0,
        world_half: float = 1.0,
    ) -> None:
        self.dt, self.drag, self.substeps = dt, drag, substeps
        self.sub_dt = dt / substeps
        self.mass, self.half = mass, world_half
        self.device = device
        self.pos = torch.zeros(batch, n_agents, 2, device=device)
        self.vel = torch.zeros(batch, n_agents, 2, device=device)

    def reset(self, generator=None) -> None:
        self.pos.uniform_(-self.half, self.half, generator=generator)
        self.vel.zero_()

    def step(self, force: torch.Tensor) -> None:
        for s in range(self.substeps):
            if s == 0:
                self.vel = self.vel * (1.0 - self.drag)
            self.vel = self.vel + (force / self.mass) * self.sub_dt
            self.pos = (self.pos + self.vel * self.sub_dt).clamp(-self.half, self.half)


def waypoint_servo(
    pos: torch.Tensor,
    vel: torch.Tensor,
    goal: torch.Tensor,
    gain: float = 4.0,
    speed: float = 0.4,
    u_range: float = 1.0,
) -> torch.Tensor:
    """A saturating velocity servo -- deliberately a *strong* stand-in for a
    blind policy, since high-gain velocity feedback is the controller class most
    able to reject an actuator disturbance.  Results are therefore optimistic
    for blind, i.e. conservative for the design.
    """
    delta = goal - pos
    dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    desired = delta / dist * torch.clamp(dist * 3.0, max=1.0) * speed
    return (gain * (desired - vel)).clamp(-u_range, u_range)


def surrogate_rollout(
    slc_core,
    params,
    n_agents: int,
    steps: int = 1200,
    batch: int = 8,
    seed: int = 0,
    compensator=None,
    waypoint_every: int = 40,
    device="cpu",
):
    """Run the SLC loop on the surrogate fleet.

    Uses the *same* ``slc_core`` functions the scenario does, so the numbers it
    reports are the environment's arithmetic, not a re-derivation.  Returns a
    dict of trajectories, all ``(T, B, ...)``.
    """
    op = slc_core.build_operator(params, n_agents, torch.device(device))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    fleet = HolonomicFleet(batch, n_agents, device=device)
    fleet.pos.uniform_(-1.0, 1.0, generator=gen)
    goal = torch.empty_like(fleet.pos).uniform_(-1.0, 1.0, generator=gen)

    phi_prev = torch.full((batch, n_agents), params.phi_floor, device=device)
    u_prev = torch.zeros(batch, n_agents, device=device)
    g_prev = torch.ones(batch, n_agents, device=device)
    step = torch.zeros(batch, dtype=torch.long, device=device)
    phase = torch.arange(batch, dtype=torch.float32, device=device) / batch
    shift = torch.zeros(batch, dtype=torch.long, device=device)

    rec = {k: [] for k in ("phi", "u", "g", "g_chan", "c", "speed", "u_hat", "delta",
                           "trust", "clip", "sat", "peer_abs", "ff_abs", "goal_dist",
                           "reached")}

    for t in range(steps):
        if t and t % waypoint_every == 0:
            goal = torch.empty_like(fleet.pos).uniform_(-1.0, 1.0, generator=gen)

        # The servo's INTENDED command, before any compensation.  Needed up
        # front because the no-loop arm reads Phi from it.
        cmd_intended = waypoint_servo(fleet.pos, fleet.vel, goal)

        if params.phi_reads_executed:
            # LOOP-COUPLED: Phi reads the executed motion, so compensating feeds
            # the medium it compensates against.
            phi = slc_core.exertion(fleet.vel, params)
        else:
            # A.6's contrast: one step of the SAME integrator on the intended
            # command.  Same velocity scale as the loop branch, so the contrast
            # isolates the loop rather than a level shift in exertion.
            phi = slc_core.exertion(
                fleet.vel * (1.0 - fleet.drag)
                + cmd_intended / fleet.mass * fleet.dt,
                params,
            )

        a = slc_core.driver_A(step, phase, params.driver_period)
        a_eff = slc_core.effective_driver(a, shift, params)
        g = slc_core.dial_g(a_eff, params, op)
        load = slc_core.channel_load(phi, op)
        u, _ = slc_core.loading(load, g, op)
        c = slc_core.harm_coefficient(u, params)
        g_ag = (
            g.unsqueeze(1)
            .masked_fill(~op.D.unsqueeze(0), float("inf"))
            .min(-1)
            .values
        )

        cmd = cmd_intended
        delta = torch.zeros_like(cmd)
        u_hat = u.clone()
        trust = torch.zeros_like(u)
        clip = torch.zeros_like(u)
        peer_abs = torch.zeros_like(u)
        ff_abs = torch.zeros_like(u)
        if compensator is not None:
            out = compensator.step(
                u_prev=u_prev,
                phi_bcast=phi_prev,
                phi_own_now=phi,
                g_prev=g_prev,
                g_now=g_ag,
                alive=torch.full((batch, n_agents), t > 0, device=device),
            )
            delta, cl = compensator.compensate(cmd, out["c_hat"])
            u_hat, trust = out["u_hat"], out["applied_trust"]
            clip = cl.to(u.dtype)
            peer_abs, ff_abs = out["peer_abs"], out["ff_abs"]
        raw = cmd + delta
        cmd = raw.clamp(-1.0, 1.0)
        sat = (raw != cmd).any(-1).to(u.dtype)

        fleet.step(slc_core.apply_harm(cmd, c))

        rec["phi"].append(phi)
        rec["u"].append(u)
        rec["g"].append(g_ag)
        rec["g_chan"].append(g)
        rec["c"].append(c)
        rec["speed"].append(fleet.vel.norm(dim=-1))
        rec["u_hat"].append(u_hat)
        rec["delta"].append(delta.abs().mean(-1))
        rec["trust"].append(trust)
        rec["clip"].append(clip)
        rec["sat"].append(sat)
        rec["peer_abs"].append(peer_abs)
        rec["ff_abs"].append(ff_abs)
        # The task metric.  Distance to the current waypoint, NOT speed: speed
        # is exactly what raises Phi, so scoring on it would reward the
        # compensator for the very thing that congests the medium.
        rec["goal_dist"].append((fleet.pos - goal).norm(dim=-1))
        rec["reached"].append(
            ((fleet.pos - goal).norm(dim=-1) < 0.15).to(fleet.vel.dtype)
        )

        phi_prev, u_prev, g_prev = phi, u, g_ag
        step = step + 1

    out = {k: torch.stack(v) for k, v in rec.items()}
    out["operator"] = op
    return out
