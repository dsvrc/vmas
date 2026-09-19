#  MF-AC -- Mean Field Actor-Critic.
#
#      Yang, Luo, Li, Zhou, Zhang, Wang, "Mean Field Multi-Agent Reinforcement
#      Learning", ICML 2018.  Reference code: mlii/mfrl, PyTorch port
#      deligentfool/mfrl_pytorch (`algo/base.py`, `algo/q_learning.py`,
#      `senarios/*.py` for how the mean action is formed).
#
#  BASELINES.md B4: "our coupling is a weighted mean of neighbours' exertions;
#  mean-field RL is the classical way to handle exactly that."  B4's own
#  decision is NOT to port it, on the grounds that the information-matched blind
#  arm -- the channels in the observation -- already IS a mean-field-style
#  policy on this problem.  That argument is sound and it is in the docs; this
#  file exists so the paper can show the method rather than argue it, and so the
#  two can be compared: the blind arm puts a WEIGHTED mean in the POLICY's
#  input, mean-field RL puts an UNWEIGHTED mean in the CRITIC's.
#
#  The whole of mean-field RL is one substitution.  Instead of a critic over the
#  joint action -- which grows with N -- the critic sees the agent's own action
#  and the MEAN action of its neighbours:
#
#      Q^j(s, a^1..a^N)   ->   Q^j(s, a^j, abar^j),
#      abar^j = (1/|N(j)|) sum_{k in N(j)} a^k
#
#  and in the reference implementation that mean is carried forward one step
#  (`former_act_prob`), because at decision time the current actions do not
#  exist yet.  Here the mean action of the previous step is already in the
#  observation -- `task.ns_observe_prev_action=true` puts each agent's own last
#  action there -- so it is read off the observation and averaged across agents.
#  Nothing is added to the environment for this arm.
#
#  See `baselines/docs/mfac.md`.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Tuple, Type

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Composite, Unbounded

from benchmarl.algorithms._baseline_math import mean_action
from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.ippo import Ippo, IppoConfig


MEAN_ACTION_KEY = "mf_mean_action"


class MeanAction(torch.nn.Module):
    """``abar^j`` from the previous actions carried in the observation."""

    def __init__(self, n_agents: int, action_slice: slice, include_self: bool):
        super().__init__()
        self.n_agents = n_agents
        self.action_slice = action_slice
        self.include_self = bool(include_self)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        #  mfrl's `former_act_prob = np.mean(one_hot(acts), axis=0)` tiled to
        #  every agent is the include_self=True branch; the paper's mean over
        #  N(j), which excludes j, is the other.
        return mean_action(observation[..., self.action_slice], self.include_self)


class Mfac(Ippo):
    """IPPO with a mean-field critic.

    Args:
        prev_action_start (int): index of the first previous-action component
            in the observation. ``task.ns_observe_prev_action=true`` appends the
            agent's own last action last, so the default ``-action_dim`` is
            correct for every simple_ns host; the algorithm resolves and prints
            the slice and raises if it does not fit.
        prev_action_size (int): how wide that slice is. ``0`` means "the action
            dimension", resolved from the action spec.
        include_self (bool): ``False`` is the paper's ``N(j)``, which excludes
            the agent itself. ``True`` is the reference implementation's team
            mean.

    All other arguments are :class:`~benchmarl.algorithms.Ippo`'s.
    """

    def __init__(
        self,
        prev_action_start: int,
        prev_action_size: int,
        include_self: bool,
        **kwargs,
    ):
        self.prev_action_start = int(prev_action_start)
        self.prev_action_size = int(prev_action_size)
        self.include_self = bool(include_self)
        super().__init__(**kwargs)

    def _action_slice(self, group: str) -> slice:
        obs_dim = int(self.observation_spec[group, "observation"].shape[-1])
        size = self.prev_action_size or int(
            self.action_spec[group, "action"].shape[-1]
        )
        start = self.prev_action_start
        if start == 0:
            start = obs_dim - size
        elif start < 0:
            start += obs_dim
        stop = start + size
        if not (0 <= start < stop <= obs_dim):
            raise ValueError(
                f"MF-AC previous-action slice [{start}, {stop}) does not fit an "
                f"observation of width {obs_dim} for group {group!r}. The mean "
                "action is read off the observation, so launch with "
                "`task.ns_observe_prev_action=true`."
            )
        return slice(start, stop)

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        action_slice = self._action_slice(group)
        width = action_slice.stop - action_slice.start
        print(
            f"MF-AC {group}: mean action = observation[..., "
            f"{action_slice.start}:{action_slice.stop}] averaged over "
            f"{'the whole team (reference code)' if self.include_self else 'N(j), self excluded (paper)'}"
            f"; the critic reads [observation, mean action], the ACTOR does not"
        )

        mean_module = TensorDictModule(
            MeanAction(n_agents, action_slice, self.include_self),
            in_keys=[(group, "observation")],
            out_keys=[(group, MEAN_ACTION_KEY)],
        )

        critic_input_spec = Composite(
            {
                group: self.observation_spec[group]
                .clone()
                .to(self.device)
                .update(
                    {
                        MEAN_ACTION_KEY: Unbounded(
                            shape=(n_agents, width), device=self.device
                        )
                    }
                )
            }
        )
        critic_output_spec = Composite(
            {
                group: Composite(
                    {"state_value": Unbounded(shape=(n_agents, 1))},
                    shape=(n_agents,),
                )
            }
        )
        value_module = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=False,  # Q^j is the agent's OWN critic, as in the paper
            input_has_agent_dim=True,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )
        return TensorDictSequential(mean_module, value_module)


@dataclass
class MfacConfig(IppoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Mfac`."""

    prev_action_start: int = MISSING
    prev_action_size: int = MISSING
    include_self: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Mfac
