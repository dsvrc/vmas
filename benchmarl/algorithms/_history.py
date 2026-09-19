#  A fixed-length window of each agent's own recent observations.
#
#  Two of the baselines need one: RMA/UP-OSI's adaptation module (B8) and
#  LIAM's encoder (B5) are both history encoders, and neither can be given a
#  history at COLLECTION time by anything inside the algorithm -- the policy is
#  called one step at a time and has no memory of its own.  The window
#  therefore has to be built by the environment, which is what torchrl's
#  ``CatFrames`` is for.
#
#  The observation itself is left untouched: a copy is made under a new key and
#  stacked in place, which is torchrl's own recommended pattern for stacking a
#  key you also want to keep unstacked.  So the critic, the debug CSV and every
#  other consumer of ``(group, "observation")`` see exactly what they see on
#  every other arm.
#
#  When the task sets ``ns_observe_prev_action=true`` each frame of the window
#  is ``(o_t, a_{t-1})``, which is the (state, action) history both RMA and
#  LIAM are defined on.

from __future__ import annotations

from typing import Callable, List

from torchrl.envs import CatFrames, Compose, EnvBase, RenameTransform, TransformedEnv


def with_observation_history(
    env_fun: Callable[[], EnvBase],
    groups: List[str],
    history_len: int,
    history_key: str,
) -> Callable[[], EnvBase]:
    def fun() -> EnvBase:
        env = env_fun()
        transforms = []
        for group in groups:
            transforms.append(
                RenameTransform(
                    in_keys=[(group, "observation")],
                    out_keys=[(group, history_key)],
                    create_copy=True,
                )
            )
            transforms.append(
                CatFrames(
                    N=history_len,
                    dim=-1,
                    in_keys=[(group, history_key)],
                    padding="same",
                )
            )
        return TransformedEnv(env, Compose(*transforms))

    return fun
