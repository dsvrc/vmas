# HAPPO — BASELINES.md B1, trust region / monotone improvement (on-policy)

## Source

| what | where |
|---|---|
| paper | Kuba et al., *Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning*, ICLR 2022 ([arXiv 2109.11251](https://arxiv.org/abs/2109.11251)) |
| library paper | Zhong et al., *Heterogeneous-Agent Reinforcement Learning*, JMLR 2024 ([arXiv 2304.09870](https://arxiv.org/abs/2304.09870)) |
| code read | [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL): `harl/algorithms/actors/happo.py` (`HAPPO.update`, `HAPPO.train`) and `harl/runners/on_policy_ha_runner.py` (`OnPolicyHARunner.train`) |
| implementation | [`benchmarl/algorithms/happo.py`](../../benchmarl/algorithms/happo.py), [`benchmarl/conf/algorithm/happo.yaml`](../../benchmarl/conf/algorithm/happo.yaml) |

## What the baseline is for

BASELINES.md B1: *the MARL literature's own answer to non-stationarity is to
constrain policy change.* If our drift were learning-induced, HAPPO would
remove it. The prediction B1 makes is that it stabilises learning and plateaus
at **the same asymptote as MAPPO** under the severity dial — because the drift
is exogenous, not induced by the peers' learning.

## The checklist

Every row is a separate idea in the paper or a separate line in HARL. `met`
means the code does that thing; `adapted` and **`NOT met`** are explained below
the table.

| # | idea | HARL's line | here | status |
|---|---|---|---|---|
| 1 | agents are updated **one at a time**, not simultaneously | `for agent_id in agent_order:` | a 0/1 agent mask on the advantage; one agent's mask is 1 per optimiser call | **met** |
| 2 | the order is a **fresh random permutation** each iteration | `agent_order = list(torch.randperm(num_agents).numpy())` | `HappoLoss._advance`, redrawn when the call counter wraps | **met** (`fixed_order: false`) |
| 3 | `fixed_order` selects a deterministic order | `if self.fixed_order:` | same flag | **met** |
| 4 | the factor starts at 1 | `factor = np.ones(...)` | `updated_mask` is all-zero for the first agent, so `exp(0)=1` | **met, tested** (`verify.py`) |
| 5 | the factor is the **product of the already-updated agents' ratios** | `factor = factor * prod(exp(new_logprob - old_logprob))` | `happo_log_factor`: the SUM of those agents' log-ratios, exponentiated | **met, tested** (`verify.py` checks it against the explicit product) |
| 6 | `old_logprob` is the agent's log-prob **before its own update** | `old_actions_logprob` computed before `.train()` | the behaviour log-prob stored at collection; identical, because an agent's policy is untouched until its own block | **met** |
| 7 | `new_logprob` is its log-prob **after** its update | recomputed after `.train()` | recomputed from the current parameters at every minibatch; identical, because the freeze (row 12) stops it moving afterwards | **met** |
| 8 | the factor is a **constant** — no gradient flows through it | `_t2n(...)`, i.e. numpy | `.detach()` inside `torch.no_grad()` | **met** |
| 9 | the objective is `-sum(factor * min(surr1, surr2))` | `policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), ...)` | the advantage is multiplied by the factor before the clipped objective | **met, tested** — `min(rMA, clip(r)MA) == M·min(rA, clip(r)A)` for `M>0` is checked in `verify.py` |
| 10 | the importance ratio aggregates over the action dimensions | `getattr(torch, action_aggregation)(exp(...), dim=-1)`, default `prod` | torchrl's `_log_weight` returns the joint log-prob per agent, so `exp` of it is that product | **met** |
| 11 | entropy is added to the policy loss | `(policy_loss - dist_entropy * entropy_coef).backward()` | `loss_entropy`, masked to the current agent, folded into `loss_objective` | **met** |
| 12 | only the current agent's parameters move | one `actor_optimizer` per agent | the objective is masked, and the other agents are written back from an authoritative copy after every step | **adapted** — see (A) |
| 13 | the advantage is standardised once over the whole batch | `advantages = (advantages - nanmean) / (nanstd + 1e-5)` | `Happo.process_batch`, population std, over the whole collected batch | **met** |
| 14 | the advantage comes from a **shared V critic** on the centralised state | `self.critic` is one `VCritic` | BenchMARL's `share_param_critic=True` centralised critic, expanded to the agents | **met** |
| 15 | the advantage is computed **before** any actor update | in the runner, before the agent loop | BenchMARL computes GAE once in `process_batch` | **met** |
| 16 | the critic is trained once per iteration, after the actors | `critic_train_info = self.critic.train(...)` | the critic is trained on every optimiser call | **adapted** — see (B) |
| 17 | each agent gets `ppo_epoch` epochs over `actor_num_mini_batch` minibatches | `for _ in range(self.ppo_epoch)` | contiguous blocks of BenchMARL's optimiser-call budget | **adapted** — see (C) |
| 18 | heterogeneous (non-shared) agent policies | one actor per agent | `experiment.share_policy_params=false`, **enforced**: the algorithm raises otherwise | **met** |
| 19 | recurrent policies (`use_recurrent_policy`) | `recurrent_generator_actor` | raises `NotImplementedError` | **NOT met** — see (D) |
| 20 | `use_policy_active_masks` / `active_masks` | masks dead agents | VMAS agents never die; all agents are always active | **not applicable** |
| 21 | `use_popart` value normalisation | `value_normalizer` | BenchMARL does not normalise values for PPO | **not applicable** (it is off in HARL's own default config too) |
| 22 | the FP/EP state-type distinction | `state_type` | BenchMARL's group critic is one object; EP (environment-provided, shared) is what it is | **met**, as EP |

### (A) The freeze — why it is there, and what it costs

HARL gives every agent its own `actor_optimizer`, so "update agent m" is
literally one optimiser step on one network. torchrl stores the `n_agents`
non-shared policies as **one stacked parameter tensor** (`MultiAgentMLP`), so:

* there is no way to give each agent its own optimiser;
* there is no way to hand Adam a `None` gradient for the others — their
  gradient is a tensor of exact zeros, not absent.

Adam keeps stepping a parameter whose gradient is zero, from the momentum left
over by its own block, by about `lr/(1-beta1) ≈ 10·lr` per iteration. That is
not noise over a 3 M-frame run, and it breaks precisely the property HAPPO
exists to provide: agents that have not had their turn must not move, or the
factor does not describe the joint update.

So `HappoLoss` keeps an authoritative copy of the actor parameters, restores
every agent except the current one at the top of each forward, and adopts an
agent's parameters into the copy when its block ends.
`_SequentialFreezeCallback` does the final restore after the last optimiser
step of the iteration, so **the parameters used for the next collection are the
sequential ones too**.

Turn it off with `freeze_non_updating_agents: false` to measure what it is
worth. If the actor parameters ever stop carrying a leading agent dimension,
the loss warns once and disables the guard rather than silently corrupting
anything.

### (B) The critic is trained on every call, not once at the end

HARL trains the critic for `critic_epoch` epochs after the agent loop. Here it
is trained on every optimiser call, so with the launcher's budget it gets
`n_agents ×` MAPPO's number of critic steps.

This does **not** affect the actors: the advantage is computed once, in
`process_batch`, before any update, so the ordering HARL relies on is preserved
exactly. What it changes is how well fitted the critic is by the end of the
iteration. It is a documented over-training of the critic, in HAPPO's favour if
anything; the alternative — zeroing the critic loss outside one block — would
put Adam's momentum back into the critic, which is the problem (A) exists to
avoid.

To match HARL's critic budget exactly, divide `experiment.critic` learning
effort by running with `HAPPO_EPOCHS=$((45 / n_agents))`, which returns the
total call count to MAPPO's — at the cost of each agent getting `45/n_agents`
epochs instead of 45.

### (C) The budget

BenchMARL performs `on_policy_n_minibatch_iters × ceil(batch / minibatch)`
optimiser calls per collection iteration, and an algorithm cannot change that
number. HAPPO spends it in `n_agents` contiguous blocks.

`scripts/baselines_common.sh` therefore launches HAPPO with

```
experiment.on_policy_n_minibatch_iters = n_agents * HAPPO_EPOCHS   (default 45)
experiment.share_policy_params         = false
```

so each agent gets exactly `HAPPO_EPOCHS` epochs — the same number MAPPO's
shared policy gets — at `n_agents ×` the compute. **HAPPO is n_agents times
more expensive per frame than MAPPO here, and that is inherent to the method,
not to this implementation.** The algorithm prints the resolved block sizes and
the implied epoch count at construction.

`verify.py` checks the split is even and that no agent gets an empty block,
including for indivisible budgets.

### (D) Recurrent policies

HARL supports a recurrent actor. Here it raises: the factor would have to be
recomputed through the recurrent state of every already-updated agent, and the
stored hidden states belong to the collection-time policy. Use
`algorithm=mappo model=layers/gru` for the memory baseline, which is B2 and a
separate row.

## Hyper-parameters: MAPPO's, not HARL's

The shared hyper-parameters in `happo.yaml` are **BenchMARL's MAPPO values**,
not HARL's, and this is deliberate: the row exists to answer "does the
sequential trust-region update change the asymptote under the dial", and it
only answers that if everything except the sequential update is held at the
MAPPO values. HARL's own defaults are recorded next to each key
(`entropy_coef: 0.01`, `gae_lambda: 0.95`, `ppo_epoch: 5`) so the alternative
run is one override away.

`factor_clip` is **not** part of HAPPO and defaults to `0.0`, which is HARL's
behaviour. It exists because the factor is a product of N importance ratios and
can overflow to `inf`, which puts a NaN in the objective and ends the run; if
that happens, `factor_clip: 10` is the declared escape hatch and the doc row
for it says so.

## Running it

```bash
GROUP=b1 ONLY=happo bash scripts/run_baselines.sh
```

## What to watch

| column | what it tells you |
|---|---|
| `happo_agent` | which agent's block the call was in. Over an iteration it must sweep all `n_agents` values; if it is constant, the block schedule is wrong. |
| `happo_factor` | the mean factor. It starts at exactly 1 for the first agent and drifts away from 1 as the sweep proceeds. A mean far from 1, or a growing one, is the overflow `factor_clip` guards against. |
| `clip_fraction` | as in MAPPO, but measured on the current agent only. |
| `ns_live_frac` | did the disturbance fire at all. |
