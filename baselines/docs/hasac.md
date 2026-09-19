# HASAC — BASELINES.md B1, trust region / monotone improvement (off-policy)

## Source

| what | where |
|---|---|
| paper | Liu et al., *Maximum Entropy Heterogeneous-Agent Reinforcement Learning*, ICLR 2024 ([arXiv 2306.10715](https://arxiv.org/abs/2306.10715)) |
| code read | [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL): `harl/algorithms/actors/hasac.py`, `harl/algorithms/critics/soft_twin_continuous_q_critic.py` (`train`, `update_alpha`), `harl/runners/off_policy_ha_runner.py` (`OffPolicyHARunner.train`, the `algo == "hasac"` branches) |
| implementation | [`benchmarl/algorithms/hasac.py`](../../benchmarl/algorithms/hasac.py), [`benchmarl/conf/algorithm/hasac.yaml`](../../benchmarl/conf/algorithm/hasac.yaml) |

## What the baseline is for

The same B1 question as HAPPO, in the off-policy half of the ladder, so the
MATD3/MASAC rows have a sequential-update competitor too. BASELINES.md calls
HASAC "off-policy, continuous actions → drop-in for the MATD3 host".

## The three things HASAC adds to MASAC

Everything else in `hasac.py` is BenchMARL's MASAC. These are the differences,
and each is a row in the table below:

1. **Sequential actor update.** Agents are updated one at a time in a random
   order; when agent *m*'s turn comes, the joint action fed to `Q` uses the
   **already-updated** policies of the agents before it.
2. **Joint soft target.** The critic target subtracts the entropy of the whole
   joint action, `alpha_c · Σ_i log π_i`, not agent *i*'s own log-prob.
3. **Per-agent temperature**, plus a separate temperature for the critic target
   tuned against the *summed* target entropy.

## The checklist

| # | idea | HARL's line | here | status |
|---|---|---|---|---|
| 1 | one joint soft Q over the centralised state and every agent's action | `SoftTwinContinuousQCritic(share_obs, cat(actions))` | BenchMARL's centralised `state_action_value` with `share_param_critic=True`, **enforced** (raises otherwise) | **met** |
| 2 | twin critics, minimum of the two | `torch.min(self.critic(...), self.critic2(...))` | `num_qvalue_nets: 2`, torchrl takes `.min(0)` | **met** |
| 3 | target critics, polyak-updated | `soft_update` | `delay_qvalue: True` + BenchMARL's `SoftUpdate(polyak_tau)` | **met** |
| 4 | critic target subtracts `alpha_c · Σ_i log π_i(a'_i)` | `next_logp_actions = sum(cat(next_logp_actions), dim=-1, keepdim=True)`; `q_targets = reward + gamma*(next_q - self.alpha*next_logp_actions)*(1-done)` | `HasacLoss._compute_target_v2`, `joint_entropy_target: True` | **met** |
| 5 | the critic's temperature is its **own** scalar | `critic.log_alpha` | `log_alpha_critic`, its own parameter and its own optimiser entry | **met** |
| 6 | the critic's temperature is tuned against the **summed** target entropy | `self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))` | `loss_alpha_critic = -log_alpha_critic * (Σ_i logp_i + n_agents·H*)` | **met** |
| 7 | actors are updated in a **random permutation** | `agent_order = list(np.random.permutation(num_agents))` | `HasacLoss._advance`, redrawn every sweep | **met** (`fixed_order: false`) |
| 8 | the updated agent's action carries gradient; the others do not | `actions[agent_id] = actor[agent_id].get_actions_with_logprobs(...)` inside the loop, the rest from the `no_grad` block | `torch.where(agent_mask, a_reparm, a_reparm.detach())` | **met** |
| 9 | after an agent is updated, its action is **recomputed** so the next agent sees it | `actions[agent_id], _ = actor[agent_id].get_actions_with_logprobs(...)` at the end of the loop body | every call resamples all agents from the live policies, so the later agent always sees the earlier one's updated policy | **met** |
| 10 | the actor loss is `-mean(Q(s, a) - alpha_m · logp_m)` | `actor_loss = -torch.mean(value_pred - self.alpha[agent_id]*logp_action)` | `(alpha * log_prob - min_q) * agent_mask`, meaned | **met** |
| 11 | the actor update uses the **live** critic, with its gradient off | `self.critic.turn_off_grad()` before the actor loop | torchrl's `_cached_detached_qvalue_params` | **met** |
| 12 | one temperature per agent, tuned on that agent's own log-prob | `alpha_loss = -(log_alpha[agent_id] * (logp[agent_id].detach() + target_entropy[agent_id])).mean()` | `log_alpha` replaced by a length-`n_agents` parameter; the loss is masked to the current agent | **met** (`per_agent_alpha: true`), with a documented fallback |
| 13 | `target_entropy = -prod(action_shape)` per agent | `-np.prod(act_space.shape)` | torchrl's `target_entropy: "auto"` | **met** |
| 14 | `policy_freq`: the actors are updated every *k*-th critic update | `if self.total_it % self.policy_freq == 0` | one actor per optimiser call, so a sweep takes `n_agents` calls | **adapted** — see (A) |
| 15 | heterogeneous (non-shared) agent policies | one actor per agent | `experiment.share_policy_params=false`; **warned**, not raised — see (B) | **met, by launch flag** |
| 16 | `use_policy_active_masks` / `valid_transition` | masks dead agents | VMAS agents never die | **not applicable** |
| 17 | `use_proper_time_limits` | uses `term` rather than `done` | BenchMARL's TD0 estimator uses the `terminated` key, which is the same distinction | **met** |
| 18 | value normalisation (`value_normalizer`) | optional | BenchMARL does not normalise values | **not applicable** (off in HARL's default config) |
| 19 | discrete actions via a Gumbel-softmax policy | `StochasticMlpPolicy` + `gumbel_softmax` | raises `NotImplementedError` | **NOT met** — a different actor, not a flag |

### (A) The budget, and the effective `policy_freq`

HARL's `train()` does one critic update and then, if it is a policy step, a
**full sweep** of `n_agents` actor updates — each its own forward, backward and
optimiser step, all on the same minibatch.

BenchMARL's loop gives a loss module one forward per optimiser call, so a sweep
inside one call is not expressible. Instead the sweep is spread over
`n_agents` consecutive calls:

* agent *m+1* is still trained against agent *m*'s **updated** policy — the
  property the method rests on is preserved exactly;
* each agent sees a different minibatch, which are i.i.d. draws from the same
  replay buffer, so the expected update is unchanged;
* the critic is updated on every call, so the actor:critic update ratio is
  `1 : n_agents`. **That is HARL with `policy_freq = n_agents`**, where HASAC's
  own default is 1.

To recover HARL's ratio exactly, multiply
`experiment.off_policy_n_optimizer_steps` by `n_agents` — at `n_agents ×` the
critic budget. The algorithm prints the effective `policy_freq` at
construction, so the row can never be read without knowing it.

### (B) Shared policy parameters

HAPPO **raises** if `share_policy_params=True`, because its factor is
meaningless without per-agent policies. HASAC **warns** instead: with one
shared policy the "update m, refresh its action for m+1" loop still runs, it
just moves every agent at once, so the run is well-defined but is not HASAC.
The launcher always sets `share_policy_params=false`.

### (C) Per-agent temperature: how it is installed

torchrl's `SACLoss` registers `log_alpha` as a scalar parameter.
`HasacLoss.__init__` replaces it with a length-`n_agents` parameter
initialised to the same value, inside a `try` that falls back to the shared
scalar with a warning if a future torchrl makes that impossible. Every place
`self._alpha` would be used is overridden in this class, so the vector never
reaches torchrl's own code paths.

The alpha loss is masked to the current agent, so each agent's temperature is
updated once per sweep — the same rate HARL updates it at. The masked entries
of `log_alpha` carry Adam momentum in the same way the actor parameters do
(see [happo.md](happo.md) (A)); on a 4-element temperature vector clamped to
`[min_alpha, max_alpha]` this is a small effect and is **not** guarded here.

## Running it

```bash
GROUP=b1 ONLY=hasac bash scripts/run_baselines.sh
```

## What to watch

| column | what it tells you |
|---|---|
| `hasac_agent` | which agent the call updated. It must cycle through all `n_agents`. |
| `alpha`, `alpha_critic` | the actor temperatures' mean and the critic's. They should settle; `alpha_critic` sitting at `max_alpha` means the joint target entropy is unreachable and the joint soft target is saturating. |
| `entropy` | `-mean log π`. Compare against `n_agents × target_entropy`. |
| `loss_qvalue` | with the joint target, this is regressing on a different quantity than MASAC's; do not compare the two numbers directly, only the returns. |
