#  ERNIE -- robust MARL by adversarial regularization.
#
#      Bukharin et al., "Robust Multi-Agent Reinforcement Learning via
#      Adversarial Regularization: Theoretical Foundation and Stable
#      Algorithms", NeurIPS 2023 (arXiv 2310.10810).
#      Reference code: abukharin3/ERNIE -- the README's "simplest version of
#      ERNIE", and `Algorithms/coma.py` (`perturb_actor` branch and
#      `get_adv_reg_loss`) for the variant with a gradient projection.
#
#  BASELINES.md B9: the robust-RL answer -- hedge against the drift instead of
#  identifying it.  ERNIE adds ONE term to the policy loss: a penalty on how far
#  the policy output moves when the observation is pushed in the direction that
#  moves it most.  That is a local Lipschitz penalty on pi, and the paper's
#  claim is that it buys robustness to perturbed observations AND to changing
#  transition dynamics.
#
#      s~ <- s + N(0, 1e-3)
#      repeat perturb_num_steps times:
#          d  <- || f(s) - f(s~) ||_F
#          g  <- d(d)/d(s~)                       (clamped to +-perturb_radius)
#          s~ <- s~ + perturb_alpha * g * |s|     (|s| optional, see below)
#      loss <- loss + lam * || f(s) - f(s~.detach()) ||_F
#
#  ``f`` is the policy network's output head: the softmax over actions for a
#  discrete policy (ERNIE's ``self.actor(obs)`` ends in a softmax), and the
#  concatenated distribution parameters ``[loc, scale]`` for a continuous one,
#  which is the same tensor -- the output of the network before the
#  distribution is built.
#
#  NOT IMPLEMENTED: the Stackelberg / leader-follower gradient correction of
#  the paper's "stable algorithms" section.  See `baselines/docs/ernie.md`.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import List, Tuple, Type

import torch
from tensordict import TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms import _compat  # submodule import: safe while
                                          # benchmarl.algorithms is still
                                          # being initialised
from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.mappo import Mappo, MappoConfig


def policy_output(dist) -> torch.Tensor:
    """ERNIE's ``self.actor(obs)`` -- the policy network's output head.

    Discrete: the probability vector (ERNIE's actor ends in a softmax).
    Continuous: ``[loc, scale]``, which is exactly what BenchMARL's actor model
    emits before the distribution is constructed.
    """
    probs = getattr(dist, "probs", None)
    if probs is not None:
        return probs
    loc = getattr(dist, "loc", None)
    scale = getattr(dist, "scale", None)
    if loc is None or scale is None:
        base = dist
        while hasattr(base, "base_dist"):
            base = base.base_dist
        loc = getattr(base, "loc", None)
        scale = getattr(base, "scale", None)
    if loc is None or scale is None:
        raise TypeError(
            "ERNIE cannot read the policy output from a distribution of type "
            f"{type(dist).__name__}: it exposes neither `probs` nor "
            "`loc`/`scale`."
        )
    return torch.cat([loc, scale], dim=-1)


class ErnieLoss(ClipPPOLoss):
    """MAPPO's clipped loss plus ERNIE's adversarial regularizer."""

    #  Redeclared so torchrl's convert_to_functional does not warn: it
    #  checks the SUBCLASS's own __annotations__, which is empty unless
    #  the names are repeated here.  Same list torchrl's own losses carry.
    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(
        self,
        *args,
        observation_keys: List,
        lam: float,
        perturb_alpha: float,
        perturb_num_steps: int,
        perturb_radius: float,
        perturb_init_std: float,
        scale_by_obs: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.observation_keys = list(observation_keys)
        self.lam = float(lam)
        self.perturb_alpha = float(perturb_alpha)
        self.perturb_num_steps = int(perturb_num_steps)
        self.perturb_radius = float(perturb_radius)
        self.perturb_init_std = float(perturb_init_std)
        self.scale_by_obs = bool(scale_by_obs)

    def _dist_from(self, tensordict, observations) -> object:
        td = tensordict.clone(False)
        for key, value in zip(self.observation_keys, observations):
            td.set(key, value)
        with self.actor_network_params.to_module(self.actor_network):
            return self.actor_network.get_dist(td)

    def adv_reg_loss(self, tensordict: TensorDictBase) -> torch.Tensor:
        clean = [tensordict.get(key).detach() for key in self.observation_keys]

        #  s~ = s + N(0, 1e-3), which is ERNIE's initialisation verbatim.
        perturbed = [
            (obs + torch.randn_like(obs) * self.perturb_init_std).requires_grad_(True)
            for obs in clean
        ]

        f_clean_detached = policy_output(self._dist_from(tensordict, clean)).detach()
        for _ in range(self.perturb_num_steps):
            f_pert = policy_output(self._dist_from(tensordict, perturbed))
            distance = torch.norm(f_clean_detached - f_pert, p="fro")
            grads = torch.autograd.grad(
                outputs=distance,
                inputs=perturbed,
                grad_outputs=torch.ones_like(distance),
                retain_graph=True,
                create_graph=False,
            )
            stepped = []
            for obs, obs_clean, grad in zip(perturbed, clean, grads):
                #  `Algorithms/coma.py` projects the gradient onto a ball before
                #  stepping; the README omits it.  perturb_radius <= 0 disables.
                if self.perturb_radius > 0:
                    grad = grad.clamp(-self.perturb_radius, self.perturb_radius)
                step = self.perturb_alpha * grad
                if self.scale_by_obs:
                    # README: `+ perturb_alpha * grad * torch.abs(obs.detach())`
                    step = step * obs_clean.abs()
                stepped.append((obs.detach() + step).requires_grad_(True))
            perturbed = stepped

        #  `get_adv_reg_loss`: the perturbed observation is DETACHED, both
        #  forwards carry the parameter gradient, and the norm is a Frobenius
        #  norm over the whole minibatch (not a mean) -- reproduced as published.
        f_clean = policy_output(self._dist_from(tensordict, clean))
        f_pert = policy_output(
            self._dist_from(tensordict, [obs.detach() for obs in perturbed])
        )
        return torch.norm(f_clean - f_pert, p="fro")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_out = super().forward(tensordict)
        if self.lam != 0.0:
            reg = self.adv_reg_loss(tensordict)
            #  ERNIE: `actor_loss = actor_loss + lam * adv_reg_loss`
            td_out.set("loss_objective", td_out.get("loss_objective") + self.lam * reg)
            td_out.set("ernie_adv_reg", reg.detach())
        return td_out


class Ernie(Mappo):
    """MAPPO + ERNIE's adversarial regularizer.

    Args:
        lam (float): the regularizer weight, ERNIE's ``config.alg.lam``.
        perturb_alpha (float): ascent step size on the observation.
        perturb_num_steps (int): number of ascent steps.
        perturb_radius (float): elementwise clamp on the ascent gradient
            (the projection in ``Algorithms/coma.py``). ``0`` disables it,
            which is the README's version.
        perturb_init_std (float): std of the Gaussian that seeds the
            perturbation. ERNIE uses ``1e-3`` everywhere.
        scale_by_obs (bool): multiply the step by ``|s|`` as the README does.

    All other arguments are :class:`~benchmarl.algorithms.Mappo`'s.
    """

    def __init__(
        self,
        lam: float,
        perturb_alpha: float,
        perturb_num_steps: int,
        perturb_radius: float,
        perturb_init_std: float,
        scale_by_obs: bool,
        **kwargs,
    ):
        self.lam = float(lam)
        self.perturb_alpha = float(perturb_alpha)
        self.perturb_num_steps = int(perturb_num_steps)
        self.perturb_radius = float(perturb_radius)
        self.perturb_init_std = float(perturb_init_std)
        self.scale_by_obs = bool(scale_by_obs)
        super().__init__(**kwargs)

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        observation_keys = [
            (group, key) for key in self.observation_spec[group].keys(True, True)
        ]
        print(
            f"ERNIE {group}: regularising on {observation_keys}, lam={self.lam}, "
            f"perturb_alpha={self.perturb_alpha}, "
            f"steps={self.perturb_num_steps}, radius={self.perturb_radius}, "
            f"scale_by_obs={self.scale_by_obs}"
        )
        loss_module = ErnieLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            **_compat.coefficient_kwargs(
                ErnieLoss, entropy_coef=self.entropy_coef, critic_coef=self.critic_coef
            ),
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            observation_keys=observation_keys,
            lam=self.lam,
            perturb_alpha=self.perturb_alpha,
            perturb_num_steps=self.perturb_num_steps,
            perturb_radius=self.perturb_radius,
            perturb_init_std=self.perturb_init_std,
            scale_by_obs=self.scale_by_obs,
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


@dataclass
class ErnieConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Ernie`."""

    lam: float = MISSING
    perturb_alpha: float = MISSING
    perturb_num_steps: int = MISSING
    perturb_radius: float = MISSING
    perturb_init_std: float = MISSING
    scale_by_obs: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Ernie
