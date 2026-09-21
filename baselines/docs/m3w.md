# M3W — X6, MoE-based multi-agent world model, with planning

## Source

| what | where |
|---|---|
| paper | Zhao, Zhao, Xu, Fu, Chai, Zhu, Zhao, *Learning and Planning Multi-Agent Tasks via a MoE-based World Model*, NeurIPS 2025 ([OpenReview `fi24ry0BX5`](https://openreview.net/forum?id=fi24ry0BX5)) |
| code read | [zhaozijie2022/m3w-marl](https://github.com/zhaozijie2022/m3w-marl): `m3w/models/world_models.py` in full (`SimNorm`, `NormedLinear`, `create_mlp`, `MLPEncoder`, `CenMoEDynamicsModel`, `NoisyTopKRouter`, `SelfAttnExpert`, `CenMoERewardModel`, `TwoHotProcessor`, `RunningScale`), `m3w/runners/world_model_runner.py` (`init_config`, `plan`, `estimate_value`, `model_train`, `actor_train`), `m3w/algorithms/actors/world_model_actor.py`, `m3w/algorithms/critics/world_model_critic.py`, `configs/mujoco/installtest/m3w/config.json` |
| built on | HARL, TDMPC2, Light-SoftMoE — the reference says so |
| implementation | [`benchmarl/algorithms/m3w.py`](../../benchmarl/algorithms/m3w.py), [`benchmarl/conf/algorithm/m3w.yaml`](../../benchmarl/conf/algorithm/m3w.yaml), the `X6` block of [`_baseline_math.py`](../../benchmarl/algorithms/_baseline_math.py) |

## Why this row belongs in the table

Every other baseline here is **policy-centric**: it learns what to do. M3W
learns **what happens** and then searches. On this instance the disturbance is
a known-form function of the neighbours' exertions, so a model that can
represent it should be able to plan around it *without being told the form* —
which is the strongest possible version of the "you did not need a declared
estimator" objection. And the multi-task MoE has a natural reading here: the
driver `A(t)` cycles, so the dynamics visit distinct regimes within one
episode, which is the "bounded similarity in task dynamics" the mixture exists
to exploit.

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | per-agent observation encoder into a **SimNorm** latent | `MLPEncoder(..., act=SimNorm(simnorm_dim))` | `M3wWorldModel.encoders`, one per agent | **met, tested** (every group of `simnorm_dim` sums to 1) |
| 2 | `latent_dim = 128`, `simnorm_dim = 8`, `num_enc_layers = 2` | `config.json` | the same three | **met** |
| 3 | `NormedLinear` = Linear + LayerNorm + Mish, dropout on the first layer only | `create_mlp` with `normed=True` | transcribed | **met** |
| 4 | **centralised SoftMoE dynamics** over the agent tokens | `CenMoEDynamicsModel.predict` | `SoftMoeDynamics`, same einsums, dispatch softmax over TOKENS and combine softmax over slots | **met, tested** — see (A) |
| 5 | `phi` is `(d_z + d_a, n_experts, 1)`, i.e. **one slot per expert** | `torch.randn(d_z+d_a, n_experts, 1) * (1/sqrt(d_z+d_a))` | the same | **met** |
| 6 | dynamics experts are MLPs `[512, 512]` with a SimNorm output | `mlp_dims=[512,512], act=SimNorm(...)` | `dynamics_hidden: 512` | **met** |
| 7 | **centralised SparseMoE reward**: noisy top-k router + self-attention experts + head | `CenMoERewardModel` | `SparseMoeReward` | **met** |
| 8 | the router is noisy top-k with a load-balancing auxiliary | `NoisyTopKRouter` | `NoisyTopKRouter` | **met**, one term simplified — see (B) |
| 9 | auxiliary = `cv_squared(importance) + cv_squared(load) + z_loss(logits)` | `load_balancing = ...` | `cv_squared` + `router_z_loss` | **met, tested** |
| 10 | experts are `MultiheadAttention` + LayerNorm + FFN across the agent tokens | `SelfAttnExpert` | `SelfAttnExpert` | **met** |
| 11 | `num_dynamics_experts = 16`, `num_reward_experts = 16`, `top_k = 2` | `config.json` | the same three | **met** |
| 12 | **two-hot distributional** reward head over `num_bins` on a symlog scale | `TwoHotProcessor` | `two_hot_encode` / `two_hot_decode` / `two_hot_loss` | **met, tested** (sums to 1, round-trips, symlog inverts) |
| 13 | `num_bins = 101`, `reward_min = -10`, `reward_max = 10` | `config.json` | the same three | **met** |
| 14 | **centralised twin Q**, two-hot, last layer zero-initialised | `DisRegQNet`; `critic.mlp[-1].weight.data.fill_(0)` | `DisRegQNet` | **met** |
| 15 | target critics, polyak-updated | `soft_update`, `polyak = 0.01` | `M3wWorldModel.polyak`, driven by `experiment.polyak_tau` | **met, tested** (exactly `(1-tau) t + tau p`) |
| 16 | **h-step latent rollout**, losses weighted by `rho^t` | `model_train`'s `for t in range(self.horizon)` | the same loop | **met** |
| 17 | dynamics loss `MSE(z_pred, enc(o_{t+1}))` | `dynamics_loss += F.mse_loss(z_pred, next_zs[:, t]) * rho^t` | the same | **met** |
| 18 | reward loss = two-hot cross-entropy | `reward_processor.dis_reg_loss` | `two_hot_loss` | **met** |
| 19 | **n-step Q targets** off the target critic and the actor | `q_targets = nstep_reward + nstep_gamma * Q'(...) * (1-term)` | `nstep_return` + `q_value(..., target=True)` | **met, tested** — see (C) |
| 20 | `n_step = 20` | `config.json` | **`n_step: 5`** | **adapted, compute** — see (C) |
| 21 | `total = q_coef Q + reward_coef R + dynamics_coef D + balance_coef B`, one optimiser | `model_optimizer` with four param groups | one BenchMARL loss, `loss_model`, over the same four modules | **met** |
| 22 | coefficients `0.1 / 0.1 / 20 / 0.0005`, `step_rho = 0.5` | `config.json` | the same five | **met** |
| 23 | gradient clip at norm 20 | `clip_grad_norm_(group['params'], 20)` | `experiment.clip_grad_val=20` in the launcher; the algorithm **warns** if it is left at BenchMARL's 5 | **met via config**, checked at construction |
| 24 | actor trained on **imagined** latents, detached from the model | `actor_train(zs)` with `zs.detach()` | the same | **met** |
| 25 | the actor sweep is **sequential**, in a fresh random order, with actions refreshed after each agent | `for agent_id in agent_order: ... update action via after-trained actor` | `M3wLoss._actor_step`, per-agent optimisers | **met** — see (D) |
| 26 | actor loss `entropy_coef * logp - scale(V)`, weighted by `rho^t` | `actor_loss += (entropy_coef*logp - value_pred).mean() * rho^t` | the same | **met** |
| 27 | `RunningScale` on the value, updated from `value_pred[0]` | `self.critic.scale.update(value_pred[0])` | `RunningScale`, updated at `t == 0` | **met** |
| 28 | squashed Gaussian actor with a bounded log-std | `WorldModelPolicy` | `M3wPolicy`, same `_log_std` clamp | **met** |
| 29 | **multi-agent MPPI planner** in latent space | `plan()` | `M3wPlanner.plan` | **met** — see (E) |
| 30 | a fraction of candidates comes from the policy, rolled through the model | `pi_actions`, `num_pi_trajs` | the same | **met** |
| 31 | `estimate_value`: discounted model rewards + `gamma^H Q` at the horizon | `estimate_value` | the same | **met**, one redundancy removed — see (F) |
| 32 | MPPI moment update: `softmax(temperature * (v - max v))`, weighted mean and std, clamped | `plan()`'s inner loop | `mppi_update` | **met, tested** |
| 33 | the executed action is **drawn from the elites** by score, then perturbed by the fitted std | `np.random.choice(p=score)`, `+= randn * act_std` | the same | **met** |
| 34 | `horizon 3, iterations 6, num_samples 128, num_pi_trajs 8, num_elites 16, min_std .05, max_std 1, temperature .5` | `config.json` | all eight unchanged | **met** |
| 35 | warm-start the plan from the previous step, **reset at `t0`** | `if not t0[thread]: act_mean[:-1] = running_mean[1:]` | always warm-started | **adapted** — see (G) |
| 36 | `warmup_steps` of random actions before training | `config.json` `warmup_steps: 10000` | `experiment.off_policy_init_random_frames=10000` in the launcher | **met via config** |
| 37 | `use_plan` ablation | `plan.use_plan` | `use_plan: false` → `m3w_noplan`, which prices the SEARCH | **met** |
| 38 | trajectory chunks from the buffer | its own `world_model_buffer` samples sequences | **frame stacking**: `horizon + n_step + 1` observations per transition | **adapted** — see (H) |
| 39 | termination masking inside the rollout | `(1 - nstep_term)` | not needed: a `CatFrames` window never spans two episodes | **met by construction** — see (H) |
| 40 | **multi-task** training across several environments | `env_args.envs`, `n_tasks` | one task whose dynamics vary in time | **adapted, structural** — see (I) |
| 41 | `enc_lr_scale` (a smaller learning rate for the encoders) | commented out in the reference | not implemented, as in the reference | **NOT met**, off in the reference |
| 42 | discrete actions | — | not implemented; the planner refits a Gaussian over action *sequences* | **NOT met** |

### (A) SoftMoE

Every agent is a token. Each expert's slot receives a softmax-weighted average
**over tokens** (the dispatch), every expert runs on its own slots, and each
token reads back a softmax-weighted average **over slots** (the combine). No
token is dropped and no expert is empty — which is the property that separates
SoftMoE from a top-k router.

`verify.py` checks it two ways: with a flat router (`phi = 0`) and an identity
expert the output is exactly the token mean; with a random router every token
still reads back the same value (there is one slot) and it is finite.

### (B) One simplification in the router

When the gating is noisy, the reference estimates `load` with a Gaussian CDF
(`_prob_in_top_k`), which makes the auxiliary differentiable through the noise
scale. The counting form `(gates > 0).sum(0)` is used here for both branches —
it is what the reference itself falls back to whenever the module is in eval
mode or `k == num_experts`. The effect is on the *auxiliary* loss only
(`balance_coef = 5e-4`), not on the routing.

### (C) `n_step = 5`, not 20

The window depth is `horizon + n_step + 1`, and the whole window is stored per
transition. At the published `n_step = 20` that is a 24-frame window — the
replay buffer would be 24x wider than any other row's, for a 100-step episode.
`n_step: 5` gives a 9-frame window. **This is the one place this row's compute
was traded against a published value**, and it is a real change: a shorter
n-step return is a more biased, lower-variance target. Raise it if the memory
is there.

`verify.py` checks `nstep_return` against its definition, including that a
termination truncates both the sum and the bootstrap.

### (D) The sequential actor sweep

Agent `m+1` is trained against agent `m`'s **already-updated** policy. One
summed loss and one backward pass cannot express that, so the actor step is
performed inside the loss with its own per-agent optimisers — the same reason
HASAC's sweep is spread over optimiser calls, and the same reason the actors
are deliberately absent from `_get_parameters`.

### (E) Cost

Planning runs `plan_iterations * horizon` dynamics **and** reward forwards over
`n_envs * num_samples` agent-token sets at **every environment step**. At the
default 300 workers that is 38 400 token-sets, 18 times per step, through a
16-expert attention mixture. The launcher therefore drops the worker count
(`M3W_ENVS=32`, `M3W_BATCH=3200`) rather than touching the planner's published
settings, because the planner's settings are the method and the worker count is
not. Raise `M3W_ENVS` if you have the compute.

### (F) One return, not `N` identical ones

The reference builds one return **per agent** from the same centralised reward
prediction — so they are identical — and then averages them across agents
before selecting elites. The average is computed once here. Arithmetically the
same number.

### (G) The plan is always warm-started

The reference resets the sampling mean at the first step of an episode
(`t0`). The acting module here is a `TensorDictModule` over the observation and
is not told where an episode began, so the mean is always shifted by one. The
cost is one stale plan per episode out of `max_steps`, and MPPI re-fits it in
the first of its six iterations anyway.

### (H) Sequences without a sequence sampler

M3W trains on trajectory chunks; BenchMARL's off-policy buffer samples flat
transitions. Rather than replace the sampler, the **sequence is put inside the
transition**: `process_env_fun` stacks `horizon + n_step + 1` observations per
step, and with `ns_observe_prev_action` and `ns_observe_prev_reward` on, frame
`k` carries `(o_k, a_{k-1}, r_{k-1})` — so frames `k` and `k+1` are one
complete transition and `W` frames are `W-1` of them. `M3wLoss._unpack` does
that, and `verify.py`'s shape test checks each of the four extracted tensors
against the frame it must come from.

Two consequences, both good: `CatFrames` refills its window on reset, so **a
window never spans two episodes** and no termination mask is needed inside the
rollout; and the frames before the first real step of an episode repeat the
reset observation, whose action and reward columns the environment zeroes —
which is the same zero padding the reference's `add_zero_elements` puts in its
own buffer.

### (I) One task

The MoE's stated motivation is multi-task transfer with "bounded similarity in
task dynamics". There is one task here. What there *is* instead is a driver
that cycles with period 100, so the dynamics pass through distinct regimes
within every episode — which is a defensible reading of the same structure, and
is why the mixture is kept at its published 16 experts rather than shrunk.
A reviewer should know the multi-task claim is not what this row tests.

### (J) BenchMARL's `model=` config is ignored

An MoE dynamics model and a two-hot distributional critic are not expressible
as an MLP width. The world model's architecture is the paper's, built in
`m3w.py`, and the construction banner says so explicitly.

## Running it

```bash
GROUP=x6 bash scripts/run_extra_baselines.sh    # m3w, m3w_noplan
```

## What to watch

| column | what it means |
|---|---|
| `m3w_dynamics` | the h-step latent prediction error. **If this does not fall the world model is not a model** and everything downstream is noise. |
| `m3w_reward` / `m3w_reward_err` | the two-hot reward loss and the mean absolute decoded error. `reward_err` is in reward units and is the readable one. |
| `m3w_q` | the distributional critic loss. |
| `m3w_balance` | the router's load-balancing auxiliary. Large and growing means the reward mixture has collapsed onto one expert. |
| `m3w_actor` / `m3w_pi_scale` | the actor loss and the `RunningScale` normaliser. |

Read `m3w` against `m3w_noplan` before reading either against anything else:
that pair is what prices the search, with the same world model behind both.
