#  LIAM -- Local Information Agent Modelling.
#
#      Papoudakis, Christianos, Albrecht, "Agent Modelling under Partial
#      Observability for Deep Reinforcement Learning", NeurIPS 2021.
#      Reference code: uoe-agents/LIAM -- `lb_foraging/models.py` (Encoder,
#      Decoder, PolicyNet) and `lb_foraging/agent.py` (class A2C).
#
#  BASELINES.md B5: the opponent/teammate-modelling line.  The prediction it is
#  there to test is that agent modelling models WHO the peers are -- their
#  policies -- which at sigma=0 is stationary, and does not model HOW MUCH THEY
#  MATTER, which is what drifts.
#
#  Three parts, and they are exactly the three in the reference code:
#
#    encoder   LSTM over the controlled agent's own (observation, last action)
#              stream -> an embedding z.
#    decoder   from z alone, reconstruct the MODELLED agents' observations and
#              actions.  Training-time only: it is a target, never an input, so
#              execution stays decentralised.  This is what makes z a model of
#              the other agents rather than a generic recurrent feature.
#    policy    acts on [own observation, z.DETACHED].
#
#  Two optimisers, as in the reference: the RL loss never reaches the encoder,
#  and the reconstruction loss never reaches the policy.  That separation is
#  the method -- without it the embedding is just a bigger network.
#
#  See `baselines/docs/liam.md` for the checklist and the four adaptations
#  (PPO in place of A2C, a truncated window in place of the whole episode, a
#  Gaussian action head in place of the softmax one, and the critic).

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, List, Tuple, Type

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

from benchmarl.algorithms._baseline_math import others_view
from benchmarl.algorithms._history import with_observation_history
from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.ippo import Ippo, IppoConfig
from benchmarl.models.common import ModelConfig


HISTORY_KEY = "liam_history"
POLICY_INPUT_KEY = "liam_policy_input"


class LiamEncoder(nn.Module):
    """``models.Encoder``: LSTM -> fc1 (ReLU) -> embedding.

    The reference runs the LSTM over the whole episode, carrying its hidden
    state.  Here it runs over the last ``history_len`` frames of the agent's own
    observation stream, which the environment stacks; with
    ``task.ns_observe_prev_action=true`` each frame is ``(o_t, a_{t-1})``, which
    is the reference's LSTM input verbatim.
    """

    def __init__(
        self,
        n_agents: int,
        frame_dim: int,
        history_len: int,
        hidden_dim: int,
        embedding_dim: int,
        share_params: bool,
        device,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.frame_dim = frame_dim
        self.history_len = history_len
        self.embedding_dim = embedding_dim
        self.share_params = share_params
        n_nets = 1 if share_params else n_agents
        self.lstm = nn.ModuleList(
            [
                nn.LSTM(frame_dim, hidden_dim, batch_first=True, device=device)
                for _ in range(n_nets)
            ]
        )
        self.fc1 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, device=device) for _ in range(n_nets)]
        )
        self.embedding = nn.ModuleList(
            [
                nn.Linear(hidden_dim, embedding_dim, device=device)
                for _ in range(n_nets)
            ]
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """``history``: ``(*batch, n_agents, history_len * frame_dim)``."""
        lead = history.shape[:-2]
        frames = history.reshape(
            -1, self.n_agents, self.history_len, self.frame_dim
        )
        outs = []
        for agent in range(self.n_agents):
            net = 0 if self.share_params else agent
            seq = frames[:, agent]  # (B, T, F)
            h, _ = self.lstm[net](seq)
            h = torch.relu(self.fc1[net](h[:, -1]))
            outs.append(self.embedding[net](h))
        out = torch.stack(outs, dim=1)  # (B, n_agents, embedding_dim)
        return out.reshape(*lead, self.n_agents, self.embedding_dim)


class LiamDecoder(nn.Module):
    """``models.Decoder``: two separate heads off the same embedding.

    Head one reconstructs the modelled agents' observations, head two their
    actions.  The reference's action head is a softmax over a discrete action
    set scored by cross-entropy; for a continuous action space the analogue is
    the same Gaussian/squared-error head the observation branch already uses,
    which is what this does.
    """

    def __init__(
        self,
        n_agents: int,
        embedding_dim: int,
        hidden_dim: int,
        others_obs_dim: int,
        others_act_dim: int,
        continuous: bool,
        share_params: bool,
        device,
    ):
        super().__init__()
        self.continuous = continuous
        self.others_act_dim = others_act_dim
        self.obs_head = MultiAgentMLP(
            n_agent_inputs=embedding_dim,
            n_agent_outputs=others_obs_dim,
            n_agents=n_agents,
            centralised=False,
            share_params=share_params,
            device=device,
            depth=2,
            num_cells=hidden_dim,
            activation_class=nn.ReLU,
        )
        self.act_head = MultiAgentMLP(
            n_agent_inputs=embedding_dim,
            n_agent_outputs=others_act_dim,
            n_agents=n_agents,
            centralised=False,
            share_params=share_params,
            device=device,
            depth=2,
            num_cells=hidden_dim,
            activation_class=nn.ReLU,
        )

    def forward(self, embedding: torch.Tensor):
        return self.obs_head(embedding), self.act_head(embedding)


class LiamPolicyInput(nn.Module):
    """``torch.cat((obs, embedding.detach()), dim=-1)``, the reference's
    ``evaluate``: the RL gradient never reaches the encoder."""

    def __init__(self, encoder: LiamEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, observation: torch.Tensor, history: torch.Tensor):
        embedding = self.encoder(history)
        return torch.cat([observation, embedding.detach()], dim=-1)


class LiamLoss(ClipPPOLoss):
    """IPPO's loss plus LIAM's reconstruction loss, kept strictly apart."""

    def __init__(
        self,
        *args,
        group: str,
        n_agents: int,
        encoder: LiamEncoder,
        decoder: LiamDecoder,
        recon_coef: float,
        mask_done: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group = group
        self.n_agents = n_agents
        self.encoder = encoder
        self.decoder = decoder
        self.recon_coef = float(recon_coef)
        self.mask_done = bool(mask_done)

    def reconstruction_loss(self, tensordict: TensorDictBase):
        history = tensordict.get((self.group, HISTORY_KEY))
        embedding = self.encoder(history)
        pred_obs, pred_act = self.decoder(embedding)

        target_obs = others_view(
            tensordict.get((self.group, "observation")), self.n_agents
        ).detach()
        action = tensordict.get((self.group, "action"))
        if not self.decoder.continuous:
            action = torch.nn.functional.one_hot(
                action.squeeze(-1).long(), num_classes=self.decoder.others_act_dim
                // max(self.n_agents - 1, 1),
            ).to(pred_act.dtype)
        target_act = others_view(action, self.n_agents).detach()

        #  `eval_decoding`: 0.5 * squared error, summed over features.
        rec_obs = 0.5 * ((target_obs - pred_obs) ** 2).sum(-1)
        if self.decoder.continuous:
            rec_act = 0.5 * ((target_act - pred_act) ** 2).sum(-1)
        else:
            probs = torch.softmax(
                pred_act.reshape(*pred_act.shape[:-1], self.n_agents - 1, -1), dim=-1
            )
            onehot = target_act.reshape(*target_act.shape[:-1], self.n_agents - 1, -1)
            rec_act = -torch.log((probs * onehot).sum(-1) + 1e-20).sum(-1)

        loss = rec_obs + rec_act
        if self.mask_done:
            #  `loss2 = ((1 - dones_batch.float()) * (recon1 + recon2)).mean()`
            done = tensordict.get(("next", self.group, "done"), None)
            if done is not None:
                loss = loss * (1.0 - done.squeeze(-1).to(loss.dtype))
        return loss.mean(), rec_obs.detach().mean(), rec_act.detach().mean()

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_out = super().forward(tensordict)
        loss, rec_obs, rec_act = self.reconstruction_loss(tensordict)
        td_out.set("loss_liam", self.recon_coef * loss)
        td_out.set("liam_recon_obs", rec_obs)
        td_out.set("liam_recon_act", rec_act)
        return td_out


class Liam(Ippo):
    """LIAM on BenchMARL's IPPO host.

    Args:
        history_len (int): how many frames of the agent's own stream the LSTM
            encoder reads. The reference runs over the whole episode; this is
            the truncation the environment-side window imposes.
        hidden_dim (int): the encoder's and decoder's hidden width
            (``hidden_dim1`` in the reference).
        embedding_dim (int): width of the embedding the policy consumes.
        recon_coef (float): multiplier on the reconstruction loss. It has its
            own optimiser, so this only rescales its learning rate; the
            reference uses a separate learning rate (``lr2``) for the same
            purpose.
        mask_done (bool): mask the reconstruction loss on terminal transitions,
            as the reference's ``(1 - dones_batch)`` does.
    """

    def __init__(
        self,
        history_len: int,
        hidden_dim: int,
        embedding_dim: int,
        recon_coef: float,
        mask_done: bool,
        **kwargs,
    ):
        self.history_len = int(history_len)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.recon_coef = float(recon_coef)
        self.mask_done = bool(mask_done)
        super().__init__(**kwargs)

        if self.has_rnn:
            raise NotImplementedError(
                "LIAM here does not support a recurrent POLICY model: the "
                "encoder is already the recurrent part, and the separation "
                "between what the RL gradient trains and what the "
                "reconstruction loss trains is the method."
            )
        self._encoders: Dict[str, LiamEncoder] = {}
        self._decoders: Dict[str, LiamDecoder] = {}

    def process_env_fun(self, env_fun):
        return with_observation_history(
            env_fun,
            groups=list(self.group_map.keys()),
            history_len=self.history_len,
            history_key=HISTORY_KEY,
        )

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        obs_dim = int(self.observation_spec[group, "observation"].shape[-1])
        action_dim = int(self.action_spec[group, "action"].shape[-1]) if continuous else 1
        n_actions = (
            None if continuous else int(self.action_spec[group, "action"].space.n)
        )

        encoder = LiamEncoder(
            n_agents=n_agents,
            frame_dim=obs_dim,
            history_len=self.history_len,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
        ).to(self.device)
        others_act_dim = (n_agents - 1) * (action_dim if continuous else n_actions)
        decoder = LiamDecoder(
            n_agents=n_agents,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            others_obs_dim=(n_agents - 1) * obs_dim,
            others_act_dim=others_act_dim,
            continuous=continuous,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
        ).to(self.device)
        self._encoders[group] = encoder
        self._decoders[group] = decoder

        encoder_module = TensorDictModule(
            LiamPolicyInput(encoder),
            in_keys=[(group, "observation"), (group, HISTORY_KEY)],
            out_keys=[(group, POLICY_INPUT_KEY)],
        )

        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                n_actions,
            ]
        actor_input_spec = Composite(
            {
                group: Composite(
                    {
                        POLICY_INPUT_KEY: Unbounded(
                            shape=(n_agents, obs_dim + self.embedding_dim),
                            device=self.device,
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
        n_agents = len(self.group_map[group])
        print(
            f"LIAM {group}: encoder LSTM over {self.history_len} frames -> "
            f"embedding {self.embedding_dim}; decoder reconstructs "
            f"{n_agents - 1} peers' observations and actions; the embedding is "
            "DETACHED before the policy and the RL loss never reaches the "
            "encoder"
        )
        loss_module = LiamLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coef,
            critic_coeff=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            group=group,
            n_agents=n_agents,
            encoder=self._encoders[group],
            decoder=self._decoders[group],
            recon_coef=self.recon_coef,
            mask_done=self.mask_done,
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
        return loss_module, False

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        #  Three optimisers, mirroring the reference's two plus BenchMARL's
        #  separate critic:
        #    loss_objective -> the policy (the encoder is inside the actor
        #                      network but its output is detached, so it gets
        #                      no gradient from here)
        #    loss_critic    -> the value function
        #    loss_liam      -> encoder + decoder, from the reconstruction loss
        #                      alone (`optimizer2` in the reference)
        encoder = self._encoders[group]
        decoder = self._decoders[group]
        liam_params = list(encoder.parameters()) + list(decoder.parameters())
        liam_ids = {id(p) for p in liam_params}
        all_actor = list(loss.actor_network_params.flatten_keys().values())
        actor_params = [p for p in all_actor if id(p) not in liam_ids]
        #  torchrl's `convert_to_functional` calls `TensorDict.from_module`,
        #  which keeps the SAME Parameter objects, so the encoder's leaves must
        #  appear in the actor's functional parameters.  If they stop doing so,
        #  the encoder would be trained twice -- once by the reconstruction
        #  loss on the module's own parameters and once by the policy gradient
        #  on a copy -- and the separation the method rests on would be gone.
        n_encoder = len(list(encoder.parameters()))
        if len(all_actor) - len(actor_params) != n_encoder:
            raise RuntimeError(
                "LIAM could not separate the encoder's parameters from the "
                f"policy's: {len(all_actor) - len(actor_params)} of the actor's "
                f"{len(all_actor)} leaves matched the encoder, expected "
                f"{n_encoder}. The RL loss would then train the encoder, which "
                "the reference explicitly forbids (`embeddings.detach()`)."
            )
        return {
            "loss_objective": actor_params,
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
            "loss_liam": liam_params,
        }


@dataclass
class LiamConfig(IppoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Liam`."""

    history_len: int = MISSING
    hidden_dim: int = MISSING
    embedding_dim: int = MISSING
    recon_coef: float = MISSING
    mask_done: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Liam
