#  MAPPO whose centralised critic is additionally handed the true exogenous
#  driver of a non-stationary task.  Execution stays fully decentralised.
#
#  This is the last rung of the PACT ladder.  The gradient that has to move the
#  compensation gain beta is weak and noisy -- beta's effect on return is real
#  but second-order next to the navigation signal -- and a critic that cannot
#  see the driver has to explain that variance as noise.  Telling the critic
#  (and only the critic) what phase the world is in sharpens the value estimate,
#  which is standard CTDE and training-only.
#
#  ---------------------------------------------------------------------------
#  Why the actor provably cannot see the payload
#  ---------------------------------------------------------------------------
#  BenchMARL captures `experiment.observation_spec` while setting up the task,
#  and only afterwards calls `Algorithm.process_env_fun`.  The payload transform
#  is installed in `process_env_fun`, so it is not in the observation spec the
#  actor is built from.  The critic gets it because this class adds the payload
#  spec to the critic's input by hand.  There is no configuration in which the
#  actor sees it.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Callable, Type

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Composite, Unbounded
from torchrl.envs import Compose, EnvBase, TransformedEnv

from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.mappo import Mappo, MappoConfig


class MappoCtde(Mappo):
    """MAPPO + a driver payload on the centralised critic.

    Args:
        critic_payload (bool): if False this is plain MAPPO, which makes the
            "does the payload matter?" ablation a one-line config change.

    All other arguments are :class:`~benchmarl.algorithms.Mappo`'s.
    """

    def __init__(self, critic_payload: bool = True, **kwargs):
        self.critic_payload = bool(critic_payload)
        super().__init__(**kwargs)
        # Imported lazily: benchmarl.algorithms is imported before
        # benchmarl.environments, and this keeps that order free of cycles.
        from benchmarl.environments.vmas_ns.pact import CtdePayloadTransform

        self._payload_transforms = {
            group: CtdePayloadTransform(group=group) for group in self.group_map
        }

    # ------------------------------------------------------------------
    # install the payload on the environment
    # ------------------------------------------------------------------

    def process_env_fun(
        self, env_fun: Callable[[], EnvBase]
    ) -> Callable[[], EnvBase]:
        if not self.critic_payload:
            return env_fun

        transforms = self._payload_transforms

        def fun() -> EnvBase:
            env = env_fun()
            return TransformedEnv(
                env,
                Compose(*(t.clone() for t in transforms.values())),
            )

        return fun

    # ------------------------------------------------------------------
    # critic
    # ------------------------------------------------------------------

    def get_critic(self, group: str) -> TensorDictModule:
        if not self.critic_payload:
            return super().get_critic(group)
        if self.state_spec is not None:
            raise ValueError(
                "MappoCtde adds the driver payload to the per-agent critic input; "
                "it is not implemented for tasks that expose a global state spec."
            )

        n_agents = len(self.group_map[group])
        transform = self._payload_transforms[group]

        if self.share_param_critic:
            critic_output_spec = Composite({"state_value": Unbounded(shape=(1,))})
        else:
            critic_output_spec = Composite(
                {
                    group: Composite(
                        {"state_value": Unbounded(shape=(n_agents, 1))},
                        shape=(n_agents,),
                    )
                }
            )

        # observation + the privileged driver, for the critic only
        group_spec = self.observation_spec[group].clone().to(self.device)
        group_spec["ctde_state"] = transform.payload_spec(n_agents, self.device)[
            group
        ]["ctde_state"]
        critic_input_spec = Composite({group: group_spec})

        value_module = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=True,
            input_has_agent_dim=True,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )
        if self.share_param_critic:
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(
                    *value.shape[:-1], n_agents, 1
                ),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(value_module, expand_module)

        return value_module


@dataclass
class MappoCtdeConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.MappoCtde`."""

    critic_payload: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MappoCtde
