#  PACT — Peer-Action Compensation with a Trained gain, for Navigation-PCW.
#
#  Everything here lives on the environment side of the boundary.  The host RL
#  algorithm is untouched: it sees a slightly wider action space and a slightly
#  wider observation, and that is all.
#
#  ---------------------------------------------------------------------------
#  Per agent i, every step
#  ---------------------------------------------------------------------------
#
#    obs_i --> host policy (UNCHANGED) --> (a_i, w_i)
#                                            |
#                                            +--> beta_i = beta_max * f(w_i)   [the ONE learned scalar]
#                                            v
#    peers' one-step-delayed scalars m_j  --> x2_i = leak_rho( mean_{j!=i} m_j )   [EXACT arithmetic]
#                                            v
#                       u_i = R(-beta_i * x2_i) a_i                [the certified channel inverse]
#                                            v
#                       env.step(u) ; recompute x2_i from executed u  (cache for t+1)
#
#    observation augmentation:  o_i (+) [ x2_i , beta_i , <|x2_i|> ]   (appended *after* the
#                               host's own observation, in native units)
#
#  ---------------------------------------------------------------------------
#  Why this cannot crater below the blind baseline
#  ---------------------------------------------------------------------------
#  With ``beta == 0`` the compensation is ``R(0) a == a`` bit for bit, i.e. the
#  blind policy.  There is no estimator anywhere in the control path that could
#  be wrong: the waveform ``x2`` is *computed* from shared messages by the same
#  recursion the environment runs, and the only learned quantity is a bounded
#  scalar that multiplies it.  An evaluation below blind therefore indicates a
#  wiring bug, not a failure of the method.
#
#  ---------------------------------------------------------------------------
#  Decentralisation
#  ---------------------------------------------------------------------------
#  These transforms compute every agent's quantities in one batched call, which
#  is an implementation convenience, not extra information.  The inputs agent i
#  actually consumes are: its own observation, its own action, and the scalars
#  ``m_j`` its teammates broadcast one step earlier -- O(N) scalars per step.
#  ``CtdePayloadTransform`` is the sole exception and is training-only: it feeds
#  the *critic*, never the actor.

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from tensordict import TensorDictBase
from torch import Tensor
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import Transform

from benchmarl.environments.vmas_ns.pcw_core import (
    angular_impulse,
    beta_from_w,
    beta_init_value,
    channel_inverse,
    leak_step,
    PcwParams,
    peer_mean,
)

#: Extra observation features PACT appends, in order.
PACT_OBS_FEATURES = ("x2", "beta", "abs_x2_mean")

#: Diagnostic entries PACT publishes under ``(group, "info")``.
PACT_INFO_KEYS = ("pact_x2", "pact_beta", "pact_theta_hat", "pact_sat")

#: Entries the Phase-1 probe / oracle arm reads from the environment.
_THETA_NEXT_KEY = "ns_theta_next"


def _reset_mask(
    tensordict: Optional[TensorDictBase], batch_shape: torch.Size, device
) -> Tensor:
    """Extract a ``(B,)`` boolean "this world is resetting" mask.

    Vectorised wrappers can reset a subset of worlds, and a wrapper whose state
    survives such a partial reset is the single most common way to get this kind
    of pipeline silently wrong.  When no ``_reset`` entry is present at all the
    call is a full reset, so everything resets.
    """
    full = torch.ones(batch_shape[0], dtype=torch.bool, device=device)
    if tensordict is None:
        return full
    mask = None
    for key in tensordict.keys(include_nested=True, leaves_only=True):
        leaf = key[-1] if isinstance(key, tuple) else key
        if leaf != "_reset":
            continue
        value = tensordict.get(key)
        value = value.reshape(value.shape[0], -1).any(dim=-1)
        mask = value if mask is None else (mask | value)
    if mask is None:
        return full
    return mask.to(device)


class _PcwTransformBase(Transform):
    """Shared batch-shaped state handling for the PCW/PACT env-side wrappers."""

    def __init__(
        self,
        group: str,
        n_agents: int,
        params: PcwParams,
        action_low: Tensor,
        action_high: Tensor,
        in_keys_inv: Sequence = (),
        out_keys_inv: Sequence = (),
    ):
        super().__init__(
            in_keys=[],
            out_keys=[],
            in_keys_inv=list(in_keys_inv),
            out_keys_inv=list(out_keys_inv),
        )
        self.group = group
        self.n_agents = n_agents
        self.params = params
        self._action_low = action_low
        self._action_high = action_high
        self._state_shape: Optional[torch.Size] = None
        self._sat_frac_running = 0.0
        self._sat_frac_count = 0

    # -- state -------------------------------------------------------------

    def _zeros(self, ref: Tensor) -> Tensor:
        return torch.zeros(self._state_shape, dtype=torch.float32, device=ref.device)

    def _ensure_state(self, ref: Tensor) -> bool:
        """(Re)allocate per-world state to match ``ref``.  Returns True if fresh."""
        shape = ref.shape[:-1]
        if (
            self._state_shape is not None
            and shape == self._state_shape
            and self._device_ok(ref)
        ):
            return False
        self._state_shape = shape
        self._allocate(ref)
        return True

    def _device_ok(self, ref: Tensor) -> bool:
        anchor = self._anchor_tensor()
        return anchor is None or anchor.device == ref.device

    def _anchor_tensor(self) -> Optional[Tensor]:
        raise NotImplementedError

    def _allocate(self, ref: Tensor) -> None:
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------

    def _clamp_action(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        clamped = torch.max(torch.min(u, self._action_high.to(u)), self._action_low.to(u))
        saturated = (clamped != u).any(dim=-1).to(torch.float32)
        return clamped, saturated

    def _note_saturation(self, saturated: Tensor) -> None:
        self._sat_frac_running += float(saturated.mean())
        self._sat_frac_count += 1

    @property
    def sat_frac(self) -> float:
        """Fraction of agent-steps whose compensated command hit the action box."""
        if self._sat_frac_count == 0:
            return 0.0
        return self._sat_frac_running / self._sat_frac_count

    def reset_sat_stats(self) -> None:
        self._sat_frac_running = 0.0
        self._sat_frac_count = 0

    def _read_info(self, tensordict: TensorDictBase, key: str) -> Optional[Tensor]:
        full_key = (self.group, "info", key)
        if full_key not in tensordict.keys(include_nested=True, leaves_only=True):
            return None
        value = tensordict.get(full_key)
        if value.shape[-1] == 1 and value.dim() > len(self._state_shape or ()):
            value = value.squeeze(-1)
        return value.to(torch.float32)

    @staticmethod
    def _write_info(tensordict: TensorDictBase, group: str, key: str, value: Tensor):
        """Write a diagnostic under ``(group, "info")`` in VMAS's ``(..., N, 1)`` layout."""
        if (group, "info") not in tensordict.keys(include_nested=True):
            return
        tensordict.set((group, "info", key), value.unsqueeze(-1).to(torch.float32))


class PactTransform(_PcwTransformBase):
    """Phase-2 PACT: computed waveform, learned scalar gain.

    Args:
        group: agent group name (``"agents"`` for VMAS).
        n_agents: number of agents in the group.
        params: the environment's PCW constants.  ``rho`` and ``gain`` are
            structural constants that PACT is entitled to know; ``severity`` and
            the driver are **not** used here (that is exactly what beta must learn).
        beta_max: upper bound of the learned gain.  Conventionally ``1.3 * peak c``.
        beta_mode: ``"affine"`` (default, for tanh-squashed hosts) or ``"sigmoid"``.
        beta_ema: light smoothing on the gain; ``0.0`` disables it.
        oracle: if True the compensation uses the environment's *true* deflection
            instead of ``beta * x2``.  This is the O1 compensation-ceiling arm,
            not a method -- it consumes privileged information at execution time.
        obs_features: append ``[x2, beta, <|x2|>]`` to the observation.
        action_low / action_high: bounds of a physical action dim, used to clip
            the compensated command exactly as the environment would.
    """

    def __init__(
        self,
        group: str,
        n_agents: int,
        params: PcwParams,
        beta_max: float,
        action_low: Tensor,
        action_high: Tensor,
        beta_mode: str = "affine",
        beta_ema: float = 0.3,
        oracle: bool = False,
        obs_features: bool = True,
    ):
        super().__init__(
            group=group,
            n_agents=n_agents,
            params=params,
            action_low=action_low,
            action_high=action_high,
            in_keys_inv=[(group, "action")],
            out_keys_inv=[(group, "action")],
        )
        self.beta_max = float(beta_max)
        self.beta_mode = beta_mode
        self.beta_ema = float(beta_ema)
        self.oracle = bool(oracle)
        self.obs_features = bool(obs_features)
        self.beta_init = beta_init_value(self.beta_max, self.beta_mode)
        self.n_extra_obs = len(PACT_OBS_FEATURES) if self.obs_features else 0

        self._x2: Optional[Tensor] = None
        self._beta: Optional[Tensor] = None
        self._theta_true: Optional[Tensor] = None
        self._abs_sum: Optional[Tensor] = None
        self._abs_count: Optional[Tensor] = None
        self._u_exec: Optional[Tensor] = None
        self._theta_hat: Optional[Tensor] = None
        self._sat: Optional[Tensor] = None

    # -- state -------------------------------------------------------------

    def _anchor_tensor(self):
        return self._x2

    def _allocate(self, ref: Tensor) -> None:
        self._x2 = self._zeros(ref)
        self._beta = torch.full_like(self._x2, self.beta_init)
        self._theta_true = self._zeros(ref)
        self._abs_sum = self._zeros(ref)
        self._abs_count = self._zeros(ref)
        self._theta_hat = self._zeros(ref)
        self._sat = self._zeros(ref)
        self._u_exec = torch.zeros(
            (*self._state_shape, 2), dtype=torch.float32, device=ref.device
        )

    def _abs_mean(self) -> Tensor:
        return self._abs_sum / self._abs_count.clamp_min(1.0)

    # -- specs -------------------------------------------------------------

    def transform_action_spec(self, action_spec: Composite) -> Composite:
        action_spec = action_spec.clone()
        key = (self.group, "action")
        spec = action_spec[key]
        low = spec.space.low
        high = spec.space.high
        # The extra control dim gets the same bounds as a physical dim so that
        # the host's action squashing treats it uniformly.
        new_low = torch.cat((low, low[..., :1]), dim=-1)
        new_high = torch.cat((high, high[..., :1]), dim=-1)
        action_spec[key] = Bounded(
            low=new_low,
            high=new_high,
            shape=torch.Size((*spec.shape[:-1], spec.shape[-1] + 1)),
            device=spec.device,
            dtype=spec.dtype,
        )
        return action_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        # Overridden explicitly rather than relying on the base class dispatching
        # into transform_action_spec, so this works across torchrl versions.
        input_spec = input_spec.clone()
        if "full_action_spec" in input_spec.keys():
            input_spec["full_action_spec"] = self.transform_action_spec(
                input_spec["full_action_spec"]
            )
        else:  # older layouts hand the action spec directly
            input_spec = self.transform_action_spec(input_spec)
        return input_spec

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec = observation_spec.clone()
        if self.n_extra_obs:
            key = (self.group, "observation")
            spec = observation_spec[key]
            observation_spec[key] = Unbounded(
                shape=torch.Size((*spec.shape[:-1], spec.shape[-1] + self.n_extra_obs)),
                device=spec.device,
                dtype=spec.dtype,
            )
        info_key = (self.group, "info")
        if info_key in observation_spec.keys(include_nested=True):
            info_spec = observation_spec[info_key]
            for name in PACT_INFO_KEYS:
                info_spec[name] = Unbounded(
                    shape=torch.Size((*info_spec.shape, 1)),
                    device=info_spec.device,
                    dtype=torch.float32,
                )
        return observation_spec

    # -- the mechanism -----------------------------------------------------

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        key = (self.group, "action")
        action = tensordict.get(key)
        if action.shape[-1] < 3:
            raise ValueError(
                f"PACT expects a {action.shape[-1] + 1}-wide action (2 thrust dims + the "
                f"gain dim w) but got {action.shape[-1]}. This happens when a policy built "
                "against the blind spec is run on a pact_enabled env -- rebuild the policy, "
                "or evaluate the checkpoint with the task config it was trained on."
            )
        self._ensure_state(action)

        a, w = action[..., :2], action[..., 2]
        beta = beta_from_w(w, self.beta_max, self.beta_mode)
        if self.beta_ema > 0.0:
            beta = self.beta_ema * self._beta + (1.0 - self.beta_ema) * beta
        self._beta = beta.detach()

        # The ONE place privileged information could enter the control path.
        # It is gated behind `oracle`, which is the ceiling arm, never the method.
        theta_hat = self._theta_true if self.oracle else beta * self._x2

        u, saturated = self._clamp_action(channel_inverse(a, theta_hat))
        self._note_saturation(saturated)
        self._u_exec = u.detach()
        self._theta_hat = theta_hat.detach()
        self._sat = saturated

        tensordict.set(key, u)
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        obs_key = (self.group, "observation")
        obs = tensordict.get(obs_key)

        # ---- advance the waveform exactly as the medium does ----
        # Positions are read from the observation the policy just acted on, i.e.
        # the positions the commands were issued from -- the same tensor the
        # scenario reads inside process_action.  Messages are one step delayed by
        # construction: the x2 used to compensate step t was finalised at t-1.
        #
        # This relies on VMAS navigation putting [pos_x, pos_y] first in the
        # observation.  That assumption is not asserted here because it does not
        # need to be: if it were wrong the per-step cosine gate would read well
        # below 1.0 and abort the run.
        pos = obs[..., 0:2]
        message = angular_impulse(pos, self._u_exec)
        phi = peer_mean(message)
        self._x2 = leak_step(
            self._x2, phi, rho=self.params.rho, gain=self.params.gain
        ).detach()

        self._abs_sum = self._abs_sum + self._x2.abs()
        self._abs_count = self._abs_count + 1.0

        # ---- cache the privileged deflection for the oracle arm ----
        theta_next = self._read_info(next_tensordict, _THETA_NEXT_KEY)
        if theta_next is not None:
            self._theta_true = theta_next.detach()

        self._augment(next_tensordict)
        return next_tensordict

    def _reset(
        self, tensordict: Optional[TensorDictBase], tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        obs = tensordict_reset.get((self.group, "observation"))
        fresh = self._ensure_state(obs)
        if not fresh:
            mask = _reset_mask(tensordict, self._state_shape, obs.device)
            self._x2[mask] = 0.0
            self._beta[mask] = self.beta_init
            self._theta_true[mask] = 0.0
            self._abs_sum[mask] = 0.0
            self._abs_count[mask] = 0.0
            self._theta_hat[mask] = 0.0
            self._sat[mask] = 0.0
            self._u_exec[mask] = 0.0
        self._augment(tensordict_reset)
        return tensordict_reset

    def _augment(self, tensordict: TensorDictBase) -> None:
        obs_key = (self.group, "observation")
        if self.n_extra_obs:
            obs = tensordict.get(obs_key)
            features = torch.stack(
                (self._x2, self._beta, self._abs_mean()), dim=-1
            ).to(obs.dtype)
            tensordict.set(obs_key, torch.cat((obs, features), dim=-1))
        self._write_info(tensordict, self.group, "pact_x2", self._x2)
        self._write_info(tensordict, self.group, "pact_beta", self._beta)
        self._write_info(tensordict, self.group, "pact_theta_hat", self._theta_hat)
        self._write_info(tensordict, self.group, "pact_sat", self._sat)


class Phase1ProbeTransform(_PcwTransformBase):
    """Phase-1 probe: scripted, privileged, no learning.

    Intercepts the action at the environment boundary, rewrites it with the
    certified compensation law driven by the environment's *true* deflection at
    a hand-set gain, and steps.  It changes no spec, so a policy trained on the
    undisturbed task can be rolled through it unmodified -- which is what makes
    the sigma* sweep cost minutes instead of a training run.

    At ``beta == 0`` (or with the driver off) the law is ``R(0) a == a`` bit for
    bit, so the transparency check passes by construction rather than by luck.
    """

    def __init__(
        self,
        group: str,
        n_agents: int,
        params: PcwParams,
        beta: float,
        action_low: Tensor,
        action_high: Tensor,
    ):
        super().__init__(
            group=group,
            n_agents=n_agents,
            params=params,
            action_low=action_low,
            action_high=action_high,
            in_keys_inv=[(group, "action")],
            out_keys_inv=[(group, "action")],
        )
        self.beta = float(beta)
        self._theta: Optional[Tensor] = None

    def _anchor_tensor(self):
        return self._theta

    def _allocate(self, ref: Tensor) -> None:
        self._theta = self._zeros(ref)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        key = (self.group, "action")
        action = tensordict.get(key)
        self._ensure_state(action)
        u, saturated = self._clamp_action(
            channel_inverse(action[..., :2], self.beta * self._theta)
        )
        self._note_saturation(saturated)
        tensordict.set(key, u)
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        theta_next = self._read_info(next_tensordict, _THETA_NEXT_KEY)
        if theta_next is None:
            raise KeyError(
                f"Phase-1 probe could not find (group, 'info', '{_THETA_NEXT_KEY}'). "
                "The privileged signal must be published by the scenario in native units."
            )
        self._theta = theta_next.detach()
        return next_tensordict

    def _reset(
        self, tensordict: Optional[TensorDictBase], tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        obs = tensordict_reset.get((self.group, "observation"))
        fresh = self._ensure_state(obs)
        if not fresh:
            # Probe state must die exactly when the episode does; a stale
            # deflection carried across a reset silently corrupts the sweep.
            self._theta[_reset_mask(tensordict, self._state_shape, obs.device)] = 0.0
        return tensordict_reset


class CtdePayloadTransform(Transform):
    """Hands the *centralised critic* the true exogenous driver.

    Training-only and critic-only: this writes a separate key that the actor's
    input spec never contains, so execution stays fully decentralised.  It is
    standard CTDE -- the critic gets privileged state, the policy does not.
    """

    #: features written, in order
    FEATURES = ("A", "c")

    def __init__(self, group: str):
        super().__init__(in_keys=[], out_keys=[])
        self.group = group
        self.key = (group, "ctde_state")

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec = observation_spec.clone()
        obs_spec = observation_spec[(self.group, "observation")]
        observation_spec[self.key] = Unbounded(
            shape=torch.Size((*obs_spec.shape[:-1], len(self.FEATURES))),
            device=obs_spec.device,
            dtype=torch.float32,
        )
        return observation_spec

    def payload_spec(self, n_agents: int, device) -> Composite:
        """The spec a critic should add to its input to consume this payload."""
        return Composite(
            {
                self.group: Composite(
                    {
                        "ctde_state": Unbounded(
                            shape=torch.Size((n_agents, len(self.FEATURES))),
                            device=device,
                            dtype=torch.float32,
                        )
                    },
                    shape=(n_agents,),
                )
            }
        )

    def _payload(self, tensordict: TensorDictBase) -> Tensor:
        obs = tensordict.get((self.group, "observation"))
        parts = []
        for name in self.FEATURES:
            full_key = (self.group, "info", f"ns_{name}")
            if full_key in tensordict.keys(include_nested=True, leaves_only=True):
                value = tensordict.get(full_key).to(torch.float32)
                if value.shape[-1] != 1:
                    value = value[..., :1]
            else:
                value = torch.zeros(
                    (*obs.shape[:-1], 1), dtype=torch.float32, device=obs.device
                )
            parts.append(value)
        return torch.cat(parts, dim=-1)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        next_tensordict.set(self.key, self._payload(next_tensordict))
        return next_tensordict

    def _reset(
        self, tensordict: Optional[TensorDictBase], tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset.set(self.key, self._payload(tensordict_reset))
        return tensordict_reset


def build_pact_transforms(
    *,
    group: str,
    n_agents: int,
    params: PcwParams,
    action_low: Tensor,
    action_high: Tensor,
    beta_max: float,
    beta_mode: str,
    beta_ema: float,
    oracle: bool,
    obs_features: bool,
) -> List[Transform]:
    """Convenience factory used by the task class."""
    return [
        PactTransform(
            group=group,
            n_agents=n_agents,
            params=params,
            beta_max=beta_max,
            action_low=action_low,
            action_high=action_high,
            beta_mode=beta_mode,
            beta_ema=beta_ema,
            oracle=oracle,
            obs_features=obs_features,
        )
    ]
