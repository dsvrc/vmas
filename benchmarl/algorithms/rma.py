#  RMA / UP-OSI -- an adaptation module distilled from a privileged teacher.
#
#      RMA: Kumar, Fu, Pathak, Malik, "RMA: Rapid Motor Adaptation for Legged
#      Robots", RSS 2021.  Code: antonilo/rl_locomotion.
#      UP-OSI: Yu, Liu, Turk, "Preparing for the Unknown: Learning a Universal
#      Policy with Online System Identification", RSS 2017.  Code:
#      VincentYu68/policy_transfer.
#
#  BASELINES.md B8 asks for exactly this, in-house: "a teacher policy
#  conditioned on the true beta*(t)*A(t) -- available inside the environment --
#  then a history encoder distilled to predict it".  It is the closest
#  METHODOLOGICAL neighbour of PACT: it also conditions on an identified
#  quantity, but the identifier is a learned history encoder rather than an
#  estimator on a declared basis.
#
#  Phase 1 (frames < phase1_frames), RMA's "base policy" stage:
#      z   = mu(e)                       e = the privileged context
#      a   = pi(o_public, z)             trained by PPO, mu jointly with pi
#      phi is trained, off to one side, to predict z from the history window.
#
#  Phase 2 (frames >= phase1_frames), RMA's "adaptation module" stage:
#      zhat = phi(history of (o, a_prev))
#      a    = pi(o_public, zhat.detach())        pi and mu FROZEN
#      phi  keeps being trained on || zhat - z ||^2, on data collected with
#      zhat in the loop -- which is what RMA does, and what makes the module
#      learn to cover its own induced state distribution.
#
#  The privileged context here is the driver A(t).  beta*(t) = sigma * L * A(t)
#  * send_m with sigma, L and send fixed before training, so beta* is A(t)
#  times a constant vector: a teacher conditioned on A(t) is a teacher
#  conditioned on beta*(t), and mu's first linear layer absorbs the constant.
#  The environment publishes A(t) with `task.ns_observe_driver=true`, and this
#  algorithm removes it from the student's inputs by construction.
#
#  See `baselines/docs/rma_osi.md`.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, List, Optional, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MultiAgentMLP,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms._history import with_observation_history
from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.mappo import Mappo, MappoConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models.common import ModelConfig


HISTORY_KEY = "rma_history"
POLICY_INPUT_KEY = "rma_policy_input"


class RmaEncoder(nn.Module):
    """RMA's environment-factor encoder ``mu`` and adaptation module ``phi``.

    ``mu`` maps the privileged context to the latent the policy consumes.
    ``phi`` maps a window of the agent's own recent observations -- which carry
    its own previous action when ``task.ns_observe_prev_action=true``, so the
    window is RMA's (state, action) history -- to a prediction of that latent.
    The context columns are DROPPED from everything ``phi`` sees and from the
    policy's own input, so the student is never handed the quantity it exists
    to infer.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        context_slice: slice,
        history_len: int,
        latent_dim: int,
        hidden: int,
        share_params: bool,
        predict_latent: bool,
        device,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.context_slice = context_slice
        self.history_len = history_len
        self.predict_latent = bool(predict_latent)
        self.context_size = context_slice.stop - context_slice.start
        self.public_dim = obs_dim - self.context_size
        self.latent_dim = int(latent_dim) if self.predict_latent else self.context_size
        self.phase = 1

        if self.predict_latent:
            # RMA's environment factor encoder: a small MLP on e.
            self.mu = MultiAgentMLP(
                n_agent_inputs=self.context_size,
                n_agent_outputs=self.latent_dim,
                n_agents=n_agents,
                centralised=False,
                share_params=share_params,
                device=device,
                depth=2,
                num_cells=hidden,
                activation_class=nn.Tanh,
            )
        else:
            #  UP-OSI: the universal policy is conditioned on the model
            #  parameters themselves and the identifier regresses onto them, so
            #  there is no latent and mu is the identity.
            self.mu = None

        # RMA's adaptation module: a feed-forward net over the history window.
        self.phi = MultiAgentMLP(
            n_agent_inputs=self.public_dim * history_len,
            n_agent_outputs=self.latent_dim,
            n_agents=n_agents,
            centralised=False,
            share_params=share_params,
            device=device,
            depth=3,
            num_cells=hidden,
            activation_class=nn.ReLU,
        )

    # -- slicing --------------------------------------------------------

    def _split(self, observation: torch.Tensor):
        cs = self.context_slice
        context = observation[..., cs]
        public = torch.cat(
            [observation[..., : cs.start], observation[..., cs.stop :]], dim=-1
        )
        return public, context

    def _history_public(self, history: torch.Tensor) -> torch.Tensor:
        """Drop the context columns from every frame of the window."""
        shape = history.shape[:-1]
        frames = history.reshape(*shape, self.history_len, self.obs_dim)
        public, _ = self._split(frames)
        return public.reshape(*shape, self.history_len * self.public_dim)

    # -- the two latents ------------------------------------------------

    def latents(self, observation: torch.Tensor, history: torch.Tensor):
        _, context = self._split(observation)
        z = self.mu(context) if self.mu is not None else context
        zhat = self.phi(self._history_public(history))
        return z, zhat

    def forward(self, observation: torch.Tensor, history: torch.Tensor):
        public, context = self._split(observation)
        if self.phase == 1:
            latent = self.mu(context) if self.mu is not None else context
        else:
            #  Detached: phi is trained by the distillation loss only, never by
            #  the policy gradient.  RMA phase 2 backpropagates nothing from the
            #  return into the adaptation module.
            latent = self.phi(self._history_public(history)).detach()
        return torch.cat([public, latent], dim=-1)


class _RmaFreezeCallback(Callback):
    """Holds the base policy and ``mu`` still once phase 2 starts."""

    def __init__(self, losses: Dict[str, "RmaLoss"]):
        super().__init__()
        self._losses = losses

    def on_train_end(self, training_td: TensorDictBase, group: str):
        loss = self._losses.get(group)
        if loss is not None:
            loss.restore_frozen()


class RmaLoss(ClipPPOLoss):
    """MAPPO's loss, plus the distillation loss, plus the phase-2 freeze.

    The freeze exists for the same reason HAPPO's does: the parameters of the
    base policy carry no gradient in phase 2 -- the policy loss is computed but
    every path into ``pi`` and ``mu`` is through a detached latent -- yet Adam
    keeps stepping them from leftover momentum. RMA phase 2 trains ONLY the
    adaptation module, so the base policy is written back after every step.
    """

    def __init__(self, *args, encoder: RmaEncoder, adapt_coef: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.adapt_coef = float(adapt_coef)
        self._frozen: Optional[Dict] = None
        self._phase_snapshot_taken = False

    # -- parameter partition --------------------------------------------

    def _leaf_items(self):
        return list(self.actor_network_params.items(True, True))

    def _is_phi(self, key, value) -> bool:
        #  torchrl's `convert_to_functional` calls `TensorDict.from_module`,
        #  which holds the SAME Parameter objects the module does, so identity
        #  is an exact partition.  The key-path fallback covers a future
        #  torchrl that clones them.
        if id(value) in self._phi_ids:
            return True
        flat = key if isinstance(key, str) else ".".join(str(k) for k in key)
        return ".phi." in f".{flat}."

    @property
    def _phi_ids(self):
        if self.__dict__.get("_phi_id_cache") is None:
            self.__dict__["_phi_id_cache"] = {
                id(p) for p in self.encoder.phi.parameters()
            }
        return self.__dict__["_phi_id_cache"]

    def phi_params(self) -> List[torch.Tensor]:
        return [v for k, v in self._leaf_items() if self._is_phi(k, v)]

    def base_params(self) -> List[torch.Tensor]:
        return [v for k, v in self._leaf_items() if not self._is_phi(k, v)]

    # -- the phase-2 freeze ---------------------------------------------

    def _snapshot_base(self) -> None:
        self._frozen = {
            k: v.detach().clone()
            for k, v in self._leaf_items()
            if not self._is_phi(k, v)
        }

    def restore_frozen(self) -> None:
        if self._frozen is None:
            return
        with torch.no_grad():
            for k, v in self._leaf_items():
                stored = self._frozen.get(k)
                if stored is not None and stored.shape == v.shape:
                    v.data.copy_(stored)

    # ------------------------------------------------------------------

    # -- losses ----------------------------------------------------------

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.encoder.phase == 2 and not self._phase_snapshot_taken:
            self._snapshot_base()
            self._phase_snapshot_taken = True
        if self.encoder.phase == 2:
            self.restore_frozen()

        td_out = super().forward(tensordict)

        #  RMA phase 2 / UP-OSI's OSI training: a plain regression of the
        #  history encoder onto the privileged encoding, on the data that was
        #  just collected.  z is detached -- phi chases mu, never the reverse.
        group = self.tensor_keys.action[0]
        observation = tensordict.get((group, "observation"))
        history = tensordict.get((group, HISTORY_KEY))
        z, zhat = self.encoder.latents(observation, history)
        loss_adapt = torch.nn.functional.mse_loss(zhat, z.detach())
        td_out.set("loss_adapt", self.adapt_coef * loss_adapt)
        td_out.set(
            "rma_phase",
            torch.as_tensor(
                float(self.encoder.phase), device=loss_adapt.device, dtype=torch.float32
            ),
        )
        td_out.set("rma_adapt_mse", loss_adapt.detach())
        return td_out


class Rma(Mappo):
    """An RMA / UP-OSI-style adaptation module on BenchMARL's MAPPO host.

    Args:
        context_start, context_size (int): where the privileged context lives in
            the observation. ``-1`` / ``1`` points at the driver appended by
            ``task.ns_observe_driver=true``.
        history_len (int): how many past observations the adaptation module
            sees. RMA uses the last 50 state-action pairs.
        latent_dim (int): width of RMA's latent ``z``.
        encoder_hidden (int): width of ``mu`` and ``phi``.
        predict_latent (bool): ``True`` is RMA -- the policy consumes a learned
            latent ``z = mu(e)`` and the student regresses onto ``z``.
            ``False`` is UP-OSI -- the policy consumes the context itself and
            the student regresses onto it.
        phase1_frames (int): frames of privileged (teacher) training before the
            adaptation module takes over the policy's input.
        adapt_coef (float): multiplier on the distillation loss. It has its own
            optimiser, so this only rescales the effective learning rate.
    """

    def __init__(
        self,
        context_start: int,
        context_size: int,
        history_len: int,
        latent_dim: int,
        encoder_hidden: int,
        predict_latent: bool,
        phase1_frames: int,
        adapt_coef: float,
        **kwargs,
    ):
        self.context_start = int(context_start)
        self.context_size = int(context_size)
        self.history_len = int(history_len)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden = int(encoder_hidden)
        self.predict_latent = bool(predict_latent)
        self.phase1_frames = int(phase1_frames)
        self.adapt_coef = float(adapt_coef)
        super().__init__(**kwargs)

        if self.has_rnn:
            raise NotImplementedError(
                "RMA here does not support recurrent models: the adaptation "
                "module already IS the history encoder, and stacking a "
                "recurrent policy on top would make 'what does the student "
                "know' unanswerable."
            )
        self._encoders: Dict[str, RmaEncoder] = {}
        self._rma_losses: Dict[str, RmaLoss] = {}
        callback = _RmaFreezeCallback(self._rma_losses)
        callback.experiment = self.experiment
        self.experiment.callbacks.append(callback)

    # ------------------------------------------------------------------

    def _obs_dim(self, group: str) -> int:
        return int(self.observation_spec[group, "observation"].shape[-1])

    def _context_slice(self, group: str) -> slice:
        obs_dim = self._obs_dim(group)
        start = self.context_start
        if start < 0:
            start += obs_dim
        stop = start + self.context_size
        if not (0 <= start < stop <= obs_dim):
            raise ValueError(
                f"RMA context slice [{start}, {stop}) does not fit an "
                f"observation of width {obs_dim} for group {group!r}. The "
                "teacher is conditioned on a privileged context that must be "
                "in the observation: set `task.ns_observe_driver=true`."
            )
        return slice(start, stop)

    def process_env_fun(self, env_fun):
        """Stack the last ``history_len`` observations into a separate key.

        RMA's adaptation module is a feed-forward network over a fixed window
        of the agent's recent history, so the window has to exist in the data.
        The observation itself is left untouched -- a copy is made under a new
        key and stacked in place, which is torchrl's own recommended pattern
        for frame stacking a key you also want to keep. That is what keeps the
        critic, and every diagnostic that reads the observation, exactly as it
        is for every other arm.
        """
        return with_observation_history(
            env_fun,
            groups=list(self.group_map.keys()),
            history_len=self.history_len,
            history_key=HISTORY_KEY,
        )

    # ------------------------------------------------------------------

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        obs_dim = self._obs_dim(group)
        context_slice = self._context_slice(group)

        encoder = RmaEncoder(
            n_agents=n_agents,
            obs_dim=obs_dim,
            context_slice=context_slice,
            history_len=self.history_len,
            latent_dim=self.latent_dim,
            hidden=self.encoder_hidden,
            share_params=self.experiment_config.share_policy_params,
            predict_latent=self.predict_latent,
            device=self.device,
        ).to(self.device)
        self._encoders[group] = encoder

        encoder_module = TensorDictModule(
            encoder,
            in_keys=[(group, "observation"), (group, HISTORY_KEY)],
            out_keys=[(group, POLICY_INPUT_KEY)],
        )

        policy_in_width = encoder.public_dim + encoder.latent_dim
        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = Composite(
            {
                group: Composite(
                    {
                        POLICY_INPUT_KEY: Unbounded(
                            shape=(n_agents, policy_in_width), device=self.device
                        )
                    },
                    shape=(n_agents,),
                )
            }
        )
        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)}, shape=(n_agents,)
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            return ProbabilisticActor(
                module=TensorDictSequential(
                    encoder_module, actor_module, extractor_module
                ),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )
        return ProbabilisticActor(
            module=TensorDictSequential(encoder_module, actor_module),
            spec=self.action_spec[group, "action"],
            in_keys=[(group, "logits")],
            out_keys=[(group, "action")],
            distribution_class=Categorical,
            return_log_prob=True,
            log_prob_key=(group, "log_prob"),
        )

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        encoder = self._encoders[group]
        print(
            f"RMA {group}: {'RMA (latent z = mu(e))' if self.predict_latent else 'UP-OSI (identify e directly)'}, "
            f"context = observation[..., {encoder.context_slice.start}:"
            f"{encoder.context_slice.stop}], public obs width "
            f"{encoder.public_dim}, latent {encoder.latent_dim}, history "
            f"{self.history_len} frames, phase 1 for {self.phase1_frames} frames"
        )
        loss_module = RmaLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coef,
            critic_coeff=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            encoder=encoder,
            adapt_coef=self.adapt_coef,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        n_phi_expected = len(list(encoder.phi.parameters()))
        n_phi_found = len(loss_module.phi_params())
        if n_phi_found != n_phi_expected:
            raise RuntimeError(
                "RMA could not separate the adaptation module's parameters "
                f"from the base policy's: expected {n_phi_expected} leaves "
                f"matching '.phi.', found {n_phi_found}. The distillation loss "
                "and the policy loss would then share an optimiser, which "
                "would train phi with the policy gradient."
            )
        self._rma_losses[group] = loss_module
        return loss_module, False

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return {
            "loss_objective": loss.base_params(),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
            "loss_adapt": loss.phi_params(),
        }

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        #  The phase switch, once per collection iteration.  total_frames is
        #  already updated for the batch that is about to be trained on.
        phase = 2 if self.experiment.total_frames >= self.phase1_frames else 1
        encoder = self._encoders[group]
        if encoder.phase != phase:
            print(
                f"[rma] {group}: entering phase {phase} at "
                f"{self.experiment.total_frames} frames -- the policy now reads "
                f"{'the adaptation module' if phase == 2 else 'the privileged encoder'}"
                + (", and the base policy is frozen" if phase == 2 else "")
            )
            encoder.phase = phase
        return super().process_batch(group, batch)

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        #  loss_adapt has its own optimiser; keep it out of loss_objective so
        #  the distillation gradient never reaches the policy.
        if "loss_entropy" in loss_vals.keys():
            loss_vals.set(
                "loss_objective",
                loss_vals["loss_objective"] + loss_vals["loss_entropy"],
            )
            del loss_vals["loss_entropy"]
        return loss_vals


@dataclass
class RmaConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Rma`."""

    context_start: int = MISSING
    context_size: int = MISSING
    history_len: int = MISSING
    latent_dim: int = MISSING
    encoder_hidden: int = MISSING
    predict_latent: bool = MISSING
    phase1_frames: int = MISSING
    adapt_coef: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Rma
