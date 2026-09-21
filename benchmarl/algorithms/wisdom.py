#  WISDOM -- wavelet predictive representations for non-stationary RL.
#
#      Wang, Li, He, Li, Bennis, Islam, Wang, "Wavelet Predictive
#      Representations for Non-Stationary Reinforcement Learning", arXiv
#      2510.04507.  Code: MinWangcs/WISDOM (a fork of CEMRL, which is a fork of
#      PEARL's rlkit -- the released tree still carries
#      `__pycache__/cemrl_algorithm.cpython-36.pyc`).
#
#  Three objects, on top of SAC:
#
#    1. A CONTEXT ENCODER.  q(z | c) over the last `time_steps` transitions
#       (o, a, r, o'), built as PEARL's permutation-invariant product of
#       Gaussians.  z is the "task representation" -- what MDP am I in.
#
#    2. A LEARNABLE WAVELET over z.  `Y_Network`: a causal a-trous
#       decomposition with LEARNED low-pass h0 and high-pass h1 filters, run to
#       `depth` levels at doubling dilation.  Its output `y` is a learned
#       weighted sum of every detail band, the final approximation band and the
#       input; its final approximation coefficient `res_lo` is the object the
#       TD operator acts on.
#
#    3. A WAVELET TD OPERATOR.  `res_lo(z_t) <- z_t + gamma res_lo(z_{t+1})`,
#       bootstrapped off a polyak-averaged target copy of the wavelet network.
#       Its fixed point is `sum_k gamma^k z_{t+k}` -- the discounted FUTURE of
#       the task representation.  That is what makes the representation
#       predictive rather than a filter, and it is the paper's contribution
#       ("a wavelet temporal difference update operator ... with theoretical
#       convergence guarantees").
#
#  The policy and the critic then see `[o, y(z)]`.
#
#  Why it belongs here.  It is the closest published neighbour of PACT among
#  the methods that LEARN the non-stationarity instead of being told it: the
#  encoder infers a latent from a window of its own transitions (as RMA's
#  adaptation module does, B8) and then PREDICTS that latent forward (which
#  nothing else in the table does).  If the disturbance on this instance is
#  predictable from a window of proprioception, this is the arm that should
#  find it.
#
#  REQUIRES both of:
#      task.ns_observe_prev_action=true    so a window of observations carries a
#      task.ns_observe_prev_reward=true    window of (o, a, r) -- the context
#                                          the encoder is defined on.
#
#  See `baselines/docs/wisdom.md`, which lists every clause and, in particular,
#  the one place the released code and the paper disagree: the released
#  `ReconstructionTrainer` trains the encoder by the KL-to-prior term ALONE,
#  with no decoder anywhere in the tree, which would drive z to the prior and
#  leave nothing for the wavelet to act on.  `encoder_loss` selects between
#  reproducing that exactly and restoring CEMRL's decoder.

from __future__ import annotations

import math
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, List, Optional, Tuple, Type

import torch
from tensordict import TensorDictBase, TensorDictParams
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MultiAgentMLP,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import LossModule, SACLoss, ValueEstimators

from benchmarl.algorithms._baseline_math import (
    gaussian_kl_to_standard_normal,
    product_of_gaussians,
    wavelet_forward_fading,
    wavelet_td_target,
)
from benchmarl.algorithms._history import with_observation_history
from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.isac import Isac, IsacConfig
from benchmarl.models.common import ModelConfig


CTX_KEY = "wisdom_context"
INPUT_KEY = "wisdom_input"


# ===========================================================================
#  the learnable wavelet
# ===========================================================================


class YNetwork(nn.Module):
    """WISDOM's ``Y_Network``, verbatim, minus the rlkit plumbing.

    ``d_model`` channels, a learned low-pass and high-pass filter of
    ``kernel_size`` taps each, ``depth`` decomposition levels, and a learned
    mixing vector ``w`` of length ``depth + 2`` (one weight per detail band,
    one for the final approximation band, one for the input itself).

    NOTE THE SHAPE CONVENTION, because it is not the obvious one.  The
    reference feeds ``task_z.view(batch, 1, latent_dim)`` with
    ``wavelet_params.dimension = 1``, i.e. ONE channel and a "sequence" whose
    length is the LATENT DIMENSION.  The decomposition is therefore across the
    coordinates of z, and the TEMPORAL structure enters through the TD operator
    (which relates ``res_lo(z_t)`` to ``res_lo(z_{t+1})``), not through this
    convolution.  Reproduced as released.
    """

    def __init__(self, d_model: int, kernel_size: int, depth: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)
        self.kernel_size = int(kernel_size)
        self.depth = int(depth)
        self.m = self.depth + 1
        scale = math.sqrt(2.0 / (self.kernel_size * 2))
        self.h0 = nn.Parameter(
            torch.empty(self.d_model, 1, self.kernel_size).uniform_(-1.0, 1.0) * scale
        )
        self.h1 = nn.Parameter(
            torch.empty(self.d_model, 1, self.kernel_size).uniform_(-1.0, 1.0) * scale
        )
        self.w = nn.Parameter(
            torch.empty(self.d_model, self.m + 1).uniform_(-1.0, 1.0)
            * math.sqrt(2.0 / (2 * self.m + 2))
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        y, res_lo = wavelet_forward_fading(
            x, self.h0, self.h1, self.w, self.depth, self.kernel_size
        )
        return self.dropout(self.activation(y)), res_lo


# ===========================================================================
#  the context encoder, and the policy's input
# ===========================================================================


class RunningNormalizer(nn.Module):
    """Welford statistics over the context, ``use_data_normalization=True``.

    The reference normalises the encoder's input with the replay buffer's own
    running mean and standard deviation, and it has to: the context carries a
    REWARD channel alongside observation and action channels, and on this
    instance a per-step reward can be two orders of magnitude larger than a
    normalised position.  Updated only from the training path, so the statistic
    does not depend on how many parallel workers collected.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.ones(dim))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1])
        n = float(flat.shape[0])
        if n < 2:
            return
        batch_mean = flat.mean(0)
        batch_m2 = flat.var(0, unbiased=False) * n
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * (n / total)
        self.m2 += batch_m2 + delta.pow(2) * (self.count * n / total)
        self.count += n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if float(self.count) < 2.0:
            return x
        std = (self.m2 / self.count).clamp_min(1e-8).sqrt()
        return (x - self.mean) / std


class WisdomEncoder(nn.Module):
    """``q(z|c)``, the wavelet over ``z``, and the policy input ``[o, y(z)]``."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        history_len: int,
        latent_dim: int,
        hidden: int,
        share_params: bool,
        wavelet_depth: int,
        wavelet_kernel: int,
        wavelet_dropout: float,
        device,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.latent_dim = latent_dim

        self.normalizer = RunningNormalizer(obs_dim)
        #  One encoder applied to EVERY frame of the window, then a product of
        #  Gaussians over frames.  That is PEARL's permutation-invariant
        #  posterior and it is what the reference builds with `MlpEncoder` plus
        #  `_product_of_gaussians`.
        self.encoder = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * latent_dim,
            n_agents=n_agents,
            centralised=False,
            share_params=share_params,
            device=device,
            depth=3,
            num_cells=hidden,
            activation_class=nn.ReLU,
        )
        self.z_model = YNetwork(
            d_model=1,
            kernel_size=wavelet_kernel,
            depth=wavelet_depth,
            dropout=wavelet_dropout,
        ).to(device)
        self.target_z_model = YNetwork(
            d_model=1,
            kernel_size=wavelet_kernel,
            depth=wavelet_depth,
            dropout=wavelet_dropout,
        ).to(device)
        self.target_z_model.load_state_dict(self.z_model.state_dict())
        for param in self.target_z_model.parameters():
            param.requires_grad = False

    # -- the posterior --------------------------------------------------

    def frames(self, history: torch.Tensor) -> torch.Tensor:
        """``(..., history_len * obs_dim)`` -> ``(..., history_len, obs_dim)``."""
        lead = history.shape[:-1]
        return history.reshape(*lead, self.history_len, self.obs_dim)

    def posterior(self, history: torch.Tensor):
        #  (*batch, n_agents, history_len, obs_dim)
        frames = self.normalizer(self.frames(history))
        #  MultiAgentMLP wants the AGENT axis at -2, so the window is swapped
        #  in front of it and folded into the batch: (*batch, L, n_agents, D).
        swapped = frames.transpose(-3, -2)
        flat = swapped.reshape(-1, self.n_agents, self.obs_dim)
        params = self.encoder(flat).reshape(*swapped.shape[:-1], -1)
        #  back to (*batch, n_agents, history_len, 2 * latent_dim)
        params = params.transpose(-3, -2)
        mu = params[..., : self.latent_dim]
        sigma_squared = torch.nn.functional.softplus(params[..., self.latent_dim :])
        #  the product is over the CONTEXT axis, which is now at -2
        return product_of_gaussians(mu, sigma_squared)

    def sample_z(self, history: torch.Tensor):
        mu, var = self.posterior(history)
        z = mu + var.sqrt() * torch.randn_like(mu)
        return z, mu, var

    # -- the wavelet ----------------------------------------------------

    def wavelet(self, z: torch.Tensor, target: bool = False):
        """``(y, res_lo)`` for a latent, in the reference's shape convention."""
        lead = z.shape[:-1]
        signal = z.reshape(-1, 1, self.latent_dim)
        model = self.target_z_model if target else self.z_model
        y, res_lo = model(signal)
        return y.reshape(*lead, self.latent_dim), res_lo.reshape(
            *lead, self.latent_dim
        )

    def soft_update_target(self, tau: float) -> None:
        with torch.no_grad():
            for target, live in zip(
                self.target_z_model.parameters(), self.z_model.parameters()
            ):
                target.data.mul_(1.0 - tau).add_(live.data, alpha=tau)

    # -- what the policy and the critic read ----------------------------

    def forward(
        self, observation: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        #  DETACHED, as the reference's `new_task_z.detach()` is: neither the
        #  encoder nor the wavelet is trained by the SAC gradient. They have
        #  their own objectives, and mixing them is what PEARL does, not this.
        with torch.no_grad():
            z, _, _ = self.sample_z(history)
            y, _ = self.wavelet(z)
        return torch.cat([observation, y], dim=-1)


class WisdomDecoder(nn.Module):
    """CEMRL's decoder: predict ``(o', r)`` from ``(o, a, z)``.

    NOT in the released WISDOM tree -- see the module docstring and
    baselines/docs/wisdom.md.  Selected by ``encoder_loss=reconstruction``,
    which is the default, because without it the encoder's only objective is
    the KL to the prior and ``z`` collapses.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden: int,
        share_params: bool,
        device,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = MultiAgentMLP(
            n_agent_inputs=obs_dim + action_dim + latent_dim,
            n_agent_outputs=obs_dim + 1,
            n_agents=n_agents,
            centralised=False,
            share_params=share_params,
            device=device,
            depth=3,
            num_cells=hidden,
            activation_class=nn.ReLU,
        )

    def forward(self, observation, action, z):
        out = self.net(torch.cat([observation, action, z], dim=-1))
        return out[..., : self.obs_dim], out[..., self.obs_dim :]


# ===========================================================================
#  the loss
# ===========================================================================


class WisdomLoss(SACLoss):
    """SAC, plus the encoder's objective and the wavelet's.

    Three gradients, three optimisers, no overlap:

      * ``loss_actor`` / ``loss_qvalue`` / ``loss_alpha`` -- stock SAC, on an
        input that contains ``y(z)`` as a CONSTANT.
      * ``loss_wisdom_enc`` -- the encoder (and, by default, the decoder):
        reconstruction of ``(o', r)`` plus ``kl_coef`` times the KL to the
        unit prior.
      * ``loss_wisdom_z`` -- the wavelet: ``MSE(y, z)`` plus
        ``td_coef * RMSE(res_lo, z + gamma res_lo_target(z'))``, which is the
        published pair (``pred_loss + td_loss_coefficient * td_loss``).
    """

    #  Redeclared so torchrl's convert_to_functional does not warn: it checks
    #  the SUBCLASS's own __annotations__.  Same list torchrl's own losses
    #  carry.
    actor_network: TensorDictModule
    qvalue_network: TensorDictModule
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams

    def __init__(
        self,
        *args,
        group: str,
        encoder: WisdomEncoder,
        decoder: Optional[WisdomDecoder],
        kl_coef: float,
        td_coef: float,
        wavelet_gamma: float,
        wavelet_tau: float,
        normalize_context: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group = group
        self.encoder = encoder
        self.decoder = decoder
        self.kl_coef = float(kl_coef)
        self.td_coef = float(td_coef)
        self.wavelet_gamma = float(wavelet_gamma)
        self.wavelet_tau = float(wavelet_tau)
        self.normalize_context = bool(normalize_context)

    # ------------------------------------------------------------------

    def _augment(self, tensordict: TensorDictBase) -> None:
        """Write ``[o, y(z)]`` for the current AND the next step.

        Recomputed here rather than read back off the buffer.  The reference
        stores ``task_indicator`` at collection time and trains on that stale
        value, which leaves its SAC update conditioned on an encoder several
        thousand gradient steps old while its wavelet is current; recomputing
        removes the inconsistency and costs one forward pass.  Stated in
        baselines/docs/wisdom.md.
        """
        group = self.group
        for prefix in ((), ("next",)):
            obs_key = prefix + (group, "observation")
            ctx_key = prefix + (group, CTX_KEY)
            if ctx_key not in tensordict.keys(True, True):
                continue
            tensordict.set(
                prefix + (group, INPUT_KEY),
                self.encoder(tensordict.get(obs_key), tensordict.get(ctx_key)),
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        if self.normalize_context:
            self.encoder.normalizer.update(
                self.encoder.frames(tensordict.get((self.group, CTX_KEY)))
            )
        self._augment(tensordict)

        td_out = super().forward(tensordict)
        td_out.update(self._representation_losses(tensordict))
        #  `ptu.soft_update_from_to(self.z_model, self.target_z_model, tau)`,
        #  in the same place the reference does it: once per training step,
        #  after the TD loss has been formed.
        self.encoder.soft_update_target(self.wavelet_tau)
        return td_out

    def _representation_losses(self, tensordict: TensorDictBase) -> Dict:
        group = self.group
        history = tensordict.get((group, CTX_KEY))
        next_history = tensordict.get(("next", group, CTX_KEY))

        # ---- the encoder -------------------------------------------------
        z, mu, var = self.encoder.sample_z(history)
        kl = gaussian_kl_to_standard_normal(mu, var).mean()
        if self.decoder is not None:
            observation = tensordict.get((group, "observation"))
            action = tensordict.get(self.tensor_keys.action)
            next_observation = tensordict.get(("next", group, "observation"))
            reward = tensordict.get(("next", group, "reward"))
            obs_hat, reward_hat = self.decoder(observation, action, z)
            recon = torch.nn.functional.mse_loss(
                obs_hat, next_observation
            ) + torch.nn.functional.mse_loss(reward_hat, reward)
        else:
            recon = torch.zeros((), device=kl.device, dtype=kl.dtype)
        loss_enc = recon + self.kl_coef * kl

        # ---- the wavelet --------------------------------------------------
        #  z is DETACHED into the wavelet: the reference feeds `task_z` from
        #  the replay buffer, which carries no gradient to the encoder, so the
        #  prediction and TD objectives shape the FILTERS and nothing else.
        z_now = z.detach()
        with torch.no_grad():
            z_next, _, _ = self.encoder.sample_z(next_history)
            _, next_res_lo = self.encoder.wavelet(z_next, target=True)
            target_res_lo = wavelet_td_target(
                z_now, next_res_lo, self.wavelet_gamma
            )
        y, res_lo = self.encoder.wavelet(z_now)
        pred_loss = torch.nn.functional.mse_loss(y, z_now)
        td_loss = torch.sqrt(((res_lo - target_res_lo) ** 2).mean())
        loss_z = pred_loss + self.td_coef * td_loss

        return {
            "loss_wisdom_enc": loss_enc,
            "loss_wisdom_z": loss_z,
            "wisdom_kl": kl.detach(),
            "wisdom_recon": recon.detach(),
            "wisdom_pred": pred_loss.detach(),
            "wisdom_td": td_loss.detach(),
            "wisdom_z_std": z_now.std().detach(),
        }


# ===========================================================================


class Wisdom(Isac):
    """WISDOM on BenchMARL's independent-SAC host.

    Args:
        time_steps (int): the context window, WISDOM's ``time_steps`` (30).
        latent_dim (int): ``latent_size``, the width of ``z`` (5).
        encoder_hidden (int): width of the context encoder.
        wavelet_depth (int): decomposition levels, ``wavelet_params.depth``.
        wavelet_kernel (int): filter taps, ``wavelet_params.filter_size``.
        wavelet_dropout (float): the reference's ``dropout=0.2``.
        wavelet_gamma (float): the discount inside the wavelet TD operator,
            ``ReconstructionTrainer.gamma``.
        wavelet_tau (float): polyak rate for the target wavelet,
            ``soft_target_tau``.
        td_coef (float): ``td_loss_coefficient``.
        kl_coef (float): weight on the KL to the unit prior. The released code
            hardcodes ``0.1``; its own config says ``alpha_kl_z = 1e-3``.
        encoder_loss (str): ``"reconstruction"`` adds CEMRL's decoder, which is
            what gives the encoder something to be informative about.
            ``"kl_only"`` reproduces the released tree exactly -- and collapses
            ``z``. See baselines/docs/wisdom.md.
        decoder_hidden (int): width of that decoder.
        normalize_context (bool): the reference's ``use_data_normalization``.

    All other arguments are :class:`~benchmarl.algorithms.Isac`'s.
    """

    def __init__(
        self,
        time_steps: int,
        latent_dim: int,
        encoder_hidden: int,
        wavelet_depth: int,
        wavelet_kernel: int,
        wavelet_dropout: float,
        wavelet_gamma: float,
        wavelet_tau: float,
        td_coef: float,
        kl_coef: float,
        encoder_loss: str,
        decoder_hidden: int,
        normalize_context: bool,
        **kwargs,
    ):
        self.time_steps = int(time_steps)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden = int(encoder_hidden)
        self.wavelet_depth = int(wavelet_depth)
        self.wavelet_kernel = int(wavelet_kernel)
        self.wavelet_dropout = float(wavelet_dropout)
        self.wavelet_gamma = float(wavelet_gamma)
        self.wavelet_tau = float(wavelet_tau)
        self.td_coef = float(td_coef)
        self.kl_coef = float(kl_coef)
        self.encoder_loss = str(encoder_loss)
        self.decoder_hidden = int(decoder_hidden)
        self.normalize_context = bool(normalize_context)
        super().__init__(**kwargs)

        if self.encoder_loss not in ("reconstruction", "kl_only"):
            raise ValueError(
                "encoder_loss must be 'reconstruction' (CEMRL's decoder, the "
                "default) or 'kl_only' (the released WISDOM tree, which "
                f"collapses z); got {encoder_loss!r}"
            )
        if self.has_rnn:
            raise NotImplementedError(
                "WISDOM here does not support recurrent models: the context "
                "encoder already IS the history encoder, and a recurrent "
                "policy on top would make 'what does z summarise' "
                "unanswerable."
            )
        self._encoders: Dict[str, WisdomEncoder] = {}
        self._decoders: Dict[str, Optional[WisdomDecoder]] = {}

    # ------------------------------------------------------------------

    def _obs_dim(self, group: str) -> int:
        return int(self.observation_spec[group, "observation"].shape[-1])

    def process_env_fun(self, env_fun):
        """Stack the last ``time_steps`` observations into a separate key.

        With ``ns_observe_prev_action`` and ``ns_observe_prev_reward`` on, each
        frame of that window is ``(o_t, a_{t-1}, r_{t-1})``, so consecutive
        frames carry every component of a transition ``(o, a, r, o')`` -- which
        is what the reference's ``update_context(o, a, r, next_o)`` builds, one
        row at a time, in its rollout worker.  The observation itself is left
        untouched; see ``_history.py``.
        """
        return with_observation_history(
            env_fun,
            groups=list(self.group_map.keys()),
            history_len=self.time_steps,
            history_key=CTX_KEY,
        )

    def _build(self, group: str):
        if group in self._encoders:
            return self._encoders[group], self._decoders[group]
        n_agents = len(self.group_map[group])
        obs_dim = self._obs_dim(group)
        encoder = WisdomEncoder(
            n_agents=n_agents,
            obs_dim=obs_dim,
            history_len=self.time_steps,
            latent_dim=self.latent_dim,
            hidden=self.encoder_hidden,
            share_params=self.experiment_config.share_policy_params,
            wavelet_depth=self.wavelet_depth,
            wavelet_kernel=self.wavelet_kernel,
            wavelet_dropout=self.wavelet_dropout,
            device=self.device,
        ).to(self.device)
        decoder = (
            WisdomDecoder(
                n_agents=n_agents,
                obs_dim=obs_dim,
                action_dim=int(self.action_spec[group, "action"].shape[-1]),
                latent_dim=self.latent_dim,
                hidden=self.decoder_hidden,
                share_params=self.experiment_config.share_policy_params,
                device=self.device,
            ).to(self.device)
            if self.encoder_loss == "reconstruction"
            else None
        )
        self._encoders[group] = encoder
        self._decoders[group] = decoder
        return encoder, decoder

    def _input_spec(self, group: str) -> Composite:
        n_agents = len(self.group_map[group])
        return Composite(
            {
                group: Composite(
                    {
                        INPUT_KEY: Unbounded(
                            shape=(n_agents, self._obs_dim(group) + self.latent_dim),
                            device=self.device,
                        )
                    },
                    shape=(n_agents,),
                )
            }
        )

    # ------------------------------------------------------------------

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError(
                "WISDOM here covers continuous actions only: the reference is "
                "SAC with a squashed Gaussian, and the discrete variant is a "
                "different actor."
            )
        n_agents = len(self.group_map[group])
        encoder, _ = self._build(group)

        encoder_module = TensorDictModule(
            encoder,
            in_keys=[(group, "observation"), (group, CTX_KEY)],
            out_keys=[(group, INPUT_KEY)],
        )
        logits_shape = list(self.action_spec[group, "action"].shape)
        logits_shape[-1] *= 2
        actor_module = model_config.get_model(
            input_spec=self._input_spec(group),
            output_spec=Composite(
                {
                    group: Composite(
                        {"logits": Unbounded(shape=logits_shape)}, shape=(n_agents,)
                    )
                }
            ),
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )
        extractor = TensorDictModule(
            NormalParamExtractor(scale_mapping=self.scale_mapping),
            in_keys=[(group, "logits")],
            out_keys=[(group, "loc"), (group, "scale")],
        )
        return ProbabilisticActor(
            module=TensorDictSequential(encoder_module, actor_module, extractor),
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

    def get_continuous_value_module(self, group: str) -> TensorDictModule:
        """``Q(o, y(z), a)`` -- the critic reads the SAME representation.

        The encoder is deliberately NOT a submodule of the critic.  torchrl
        expands the Q network's parameters by ``num_qvalue_nets``, which would
        make two frozen copies of the encoder and leave the one the policy uses
        training alone.  The critic reads the key the ACTOR writes; the loss
        refreshes that key for the current and the next step before it
        delegates, so every term in one update sees the same ``y(z)``.
        """
        n_agents = len(self.group_map[group])
        critic_input_spec = Composite(
            {
                group: self._input_spec(group)[group]
                .clone()
                .update(self.action_spec[group])
            }
        )
        critic_output_spec = Composite(
            {
                group: Composite(
                    {"state_action_value": Unbounded(shape=(n_agents, 1))},
                    shape=(n_agents,),
                )
            }
        )
        return TensorDictSequential(
            self.critic_model_config.get_model(
                input_spec=critic_input_spec,
                output_spec=critic_output_spec,
                n_agents=n_agents,
                centralised=False,
                input_has_agent_dim=True,
                agent_group=group,
                share_params=self.share_param_critic,
                device=self.device,
                action_spec=self.action_spec,
            )
        )

    # ------------------------------------------------------------------

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        encoder, decoder = self._build(group)
        print(
            f"WISDOM {group}: context = {self.time_steps} frames of the "
            f"observation ({self._obs_dim(group)} wide, so "
            f"{self.time_steps * self._obs_dim(group)} inputs), z is "
            f"{self.latent_dim}-dimensional; the wavelet runs "
            f"{self.wavelet_depth} levels of {self.wavelet_kernel}-tap learned "
            f"filters ACROSS THE LATENT COORDINATES (d_model=1, seq_len="
            f"{self.latent_dim}), as released; TD operator gamma="
            f"{self.wavelet_gamma} tau={self.wavelet_tau} coef={self.td_coef}; "
            f"encoder objective = {self.encoder_loss}"
            + (
                " (CEMRL's decoder + KL)"
                if decoder is not None
                else " (KL TO THE PRIOR ALONE -- the released behaviour; z has "
                "nothing to be informative about and will collapse. This is "
                "here to reproduce the release, not to be believed)"
            )
            + f"; the policy and the critic read [observation, y(z)]"
        )
        loss_module = WisdomLoss(
            actor_network=policy_for_loss,
            qvalue_network=self.get_continuous_value_module(group),
            num_qvalue_nets=self.num_qvalue_nets,
            loss_function=self.loss_function,
            alpha_init=self.alpha_init,
            min_alpha=self.min_alpha,
            max_alpha=self.max_alpha,
            action_spec=self.action_spec,
            fixed_alpha=self.fixed_alpha,
            target_entropy=self.target_entropy,
            delay_qvalue=self.delay_qvalue,
            group=group,
            encoder=encoder,
            decoder=decoder,
            kl_coef=self.kl_coef,
            td_coef=self.td_coef,
            wavelet_gamma=self.wavelet_gamma,
            wavelet_tau=self.wavelet_tau,
            normalize_context=self.normalize_context,
        )
        loss_module.set_keys(
            state_action_value=(group, "state_action_value"),
            action=(group, "action"),
            reward=(group, "reward"),
            priority=(group, "td_error"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=self.experiment_config.gamma
        )
        return loss_module, True

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        encoder = self._encoders[group]
        decoder = self._decoders[group]
        representation = {id(p) for p in encoder.parameters()}
        representation |= {id(b) for b in encoder.buffers()}
        if decoder is not None:
            representation |= {id(p) for p in decoder.parameters()}

        def _without_representation(params) -> List[torch.Tensor]:
            return [
                value
                for _, value in params.items(True, True)
                if id(value) not in representation
            ]

        actor = _without_representation(loss.actor_network_params)
        if len(actor) == len(list(loss.actor_network_params.items(True, True))):
            raise RuntimeError(
                "WISDOM could not find the context encoder inside the actor's "
                "parameters. The SAC gradient would then be the encoder's only "
                "objective, which is neither the paper's nor the released "
                "code's."
            )
        items = {
            "loss_actor": actor,
            "loss_qvalue": list(loss.qvalue_network_params.flatten_keys().values()),
            #  Two objectives the SAC gradient never touches, and two
            #  optimisers so they cannot be mixed: `optimizer_encoder` and
            #  `z_optimizer` in ReconstructionTrainer.
            "loss_wisdom_enc": list(encoder.encoder.parameters())
            + (list(decoder.parameters()) if decoder is not None else []),
            "loss_wisdom_z": list(encoder.z_model.parameters()),
        }
        if not self.fixed_alpha:
            items["loss_alpha"] = [loss.log_alpha]
        return items


@dataclass
class WisdomConfig(IsacConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Wisdom`."""

    time_steps: int = MISSING
    latent_dim: int = MISSING
    encoder_hidden: int = MISSING
    wavelet_depth: int = MISSING
    wavelet_kernel: int = MISSING
    wavelet_dropout: float = MISSING
    wavelet_gamma: float = MISSING
    wavelet_tau: float = MISSING
    td_coef: float = MISSING
    kl_coef: float = MISSING
    encoder_loss: str = MISSING
    decoder_hidden: int = MISSING
    normalize_context: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Wisdom

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False
