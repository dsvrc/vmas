# LCPO — BASELINES.md B6, non-stationary RL with an observed context

## Source

| what | where |
|---|---|
| paper | Hamadanian, Gao, Schwarzkopf, Alizadeh, *Online Reinforcement Learning in Non-Stationary Context-Driven Environments*, ICLR 2025 ([OpenReview](https://openreview.net/pdf?id=l6QnSQizmN)) |
| code read | [pouyahmdn/LCPO](https://github.com/pouyahmdn/LCPO), the `windy-gym` benchmark (continuous-control hosts): `agent/core_alg/core_lcpo.py` (`locopo`, `trpo_step`, `linesearch`), `agent/core_alg/core_trpo.py` (`conjugate_gradients`), `agent/core_alg/core_pg.py` (`gae_advantage`, `value_train`, `policy_gradient`), `buffer/buffer_ood.py` (`OutOfDSampler`), `agent/lcpo.py` (the loop), `env/hopper.py` (`is_different`, `only_context`), `param.py` (defaults) |
| implementation | [`benchmarl/algorithms/lcpo.py`](../../benchmarl/algorithms/lcpo.py), [`benchmarl/algorithms/_baseline_math.py`](../../benchmarl/algorithms/_baseline_math.py), [`benchmarl/conf/algorithm/lcpo.yaml`](../../benchmarl/conf/algorithm/lcpo.yaml) |

## What the baseline is for

BASELINES.md B6 calls this *the most direct recent competitor*: non-stationarity
induced by an **exogenous observed context** — which is exactly the driver
`A(t)` — with a method designed for it. The prediction: *it prevents forgetting
across wet/dry contexts but cannot cancel; knowing the weather is not knowing
the neighbours' load under it.*

LCPO is **required** to be run with `task.ns_observe_driver=true`. The
algorithm raises if it cannot locate the context in the observation, because
LCPO without an observed context is not LCPO.

## What the method is

A trust-region step with **two** KL constraints:

```
maximise   E_local[ -A * pi_new/pi_old ]  +  entropy_factor * H
s.t.       KL over the CURRENT context           <= kl_in
           KL over OUT-OF-DISTRIBUTION contexts  <= kl_out     <- the method
```

The second constraint is what makes it LCPO. States whose *context* is far from
the context the agent is living in now are drawn from a reservoir of everything
it has ever seen, and the policy is forbidden to move much on them. `kl_out`
(1e-3) is two orders of magnitude tighter than `kl_in` (1e-1): learn here,
don't forget there.

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | an unclipped surrogate, not PPO's clip | `action_loss = -adv * exp(log_pi_act - log_pi_act_before)` | identical | **met** |
| 2 | an entropy bonus in the objective | `- entropy.mean() * entropy_factor` | identical | **met** |
| 3 | GAE advantages from a value network | `gae_advantage(...)` | BenchMARL's GAE, `lmbda: 0.95` to match | **met** |
| 4 | the value network is trained by MSE | `value_train`, `MSELoss`, clip 0.5 | `loss_critic` with `loss_critic_type: l2`, BenchMARL's optimiser and grad clip | **met** |
| 5 | conjugate gradients for the natural-gradient direction | `conjugate_gradients(q_out_dot_v, -loss_grad, 15)` | same function, ported to `_baseline_math.py` | **met, tested** (`verify.py` solves a random SPD system) |
| 6 | Fisher-vector products with damping | `flat_grad_grad_kl + v * damping` | identical, over torchrl's functional actor parameters | **met** |
| 7 | the **out-of-distribution** KL constraint | `get_kl_out` on `states_global_torch` | `kl_out()` on the reservoir batch | **met — this is the method** |
| 8 | the **in-distribution** KL constraint | `get_kl_in` on the current batch | `kl_in()` | **met** |
| 9 | the two-constraint **dual solve** | the `solve_dual` branch: two CG solves, the quadratic in `s`, `get_qu` | ported line for line | **met**; `get_qu` tested in `verify.py` |
| 10 | the single-constraint fallback when the dual has no positive root | `l_out = sqrt(2*max_kl_out/vout)` | identical, including the `diff[1]**2 <= 4*diff[2]` test | **met** |
| 11 | a backtracking line search that must satisfy **both** KLs | `linesearch(..., max_kl_out, max_kl_in)` with `ratio > accept_ratio and actual_improve > 0` | ported line for line | **met** |
| 12 | the parameters are **written**, not stepped by an optimiser | `set_flat_params_to(model, new_params)` | `set_flat_params`, and **no optimiser is bound to the actor** — see (A) | **met** |
| 13 | reservoir sampling over every state seen | `OutOfDSampler.add_many_exp` | `OutOfDistributionSampler.add` | **met**, with a bounded capacity — see (B) |
| 14 | a FIFO window of the recent past | `self.recent_states`, size `ood_mini_len` | `ood_window`, set to one rollout by the launcher | **met** |
| 15 | the OOD test is on the **context** part of the state | `only_context(obs)` then `dist > thresh` | `context_slice`, resolved from `context_start`/`context_size` and printed | **met, tested** (`verify.py` walks the buffer through a context change) |
| 16 | the `l2` distance variant | `dist_func_type == 'l2'` | implemented | **met** |
| 17 | the `mahala` / `mahala_full` variants | also in `env/hopper.py` | not implemented | **NOT met** — the paper's headline runs use `--lcpo_ood_type l2` and `--lcpo_thresh 1`; the Mahalanobis variants need a covariance over a 6-dim context and ours is 1-dim, where they reduce to a scaled `l2` |
| 18 | **fall back to plain policy gradient** when no OOD batch can be found | `if len(ood_obs_np) > 0: locopo(...) else: policy_gradient(...)` | `_policy_gradient_step`, with LCPO's own Adam (`weight_decay=1e-4, eps=1e-5`) and its 0.5 grad clip | **met** — see (A) |
| 19 | one policy step and one value step per rollout, on the whole rollout | the training loop | `process_batch` stashes the whole batch; the first optimiser call of the iteration consumes it. The launcher sets minibatch = batch and one iteration | **met, by launch flag** |
| 20 | entropy decay: `entropy_factor = max(entropy_factor - decay, min)` per rollout | `tune_entropy`, decay branch | identical | **met** |
| 21 | `--auto_target_entropy`, the automatic entropy tuning | `tune_entropy`, the other branch | not implemented | **NOT met** — a second mechanism; the paper's `run_config_phase1.py` uses `--auto_target_entropy 0.1`, so this is a real gap and is listed in the README |
| 22 | single agent | LCPO is single-agent | one joint trust region over the product policy | **adapted** — see (C) |
| 23 | no episode boundaries (continual online RL) | `windy-gym` runs without resets | `balance` has 100-step episodes; the driver's clock persists across them (NS-3.4), so the context process is continual even though the task is episodic | **met in the part that matters** |

### (A) How a trust-region step lives inside BenchMARL

BenchMARL trains through `loss -> backward -> optimizer.step()`. TRPO does not
descend a gradient: it solves a constrained quadratic and writes the parameters.

So `Lcpo._get_parameters` returns **only** `loss_critic`. There is no optimiser
bound to the actor at all, and the `loss_objective` the loss reports is a
detached number for the log. The policy step — the CG solves, the dual, the
line search, the parameter write — happens inside `LcpoLoss.forward`, on the
whole rollout that `process_batch` stashed.

The A2C fallback branch needs a gradient step, so `LcpoLoss` builds LCPO's own
Adam over the actor leaves the first time that branch is taken. It is the only
optimiser that ever touches the actor, and it only runs when the reservoir
cannot produce an OOD batch.

### (B) The reservoir is bounded

LCPO's `ood_len = master_batch * num_epochs`: it keeps **every** state ever
seen. At a 3 M-frame budget that is 3 M joint states, which is 4 GB at this
observation width.

Here `ood_capacity: 50000` joint states (~16 MB at 4 agents and a 20-wide
observation) with `ood_subsample: 10`, LCPO's own knob, so one state in ten is
offered to the reservoir. Reservoir sampling keeps the retained set uniform
over everything offered, so the *distribution* the constraint is evaluated on
is unchanged; only its resolution is. Both numbers are in the yaml with this
note next to them.

### (C) One agent to many

LCPO is a single-agent algorithm. Here the KL is computed per agent and then
**meaned** over agents and batch, and one trust-region step is taken over the
whole policy parameter vector.

* With `share_policy_params=True` (the default) there is literally one policy,
  and the mean over (state, agent) pairs is the natural per-sample statistic —
  it coincides with LCPO's when there is one agent.
* With `share_policy_params=False` this is a **joint** trust region over the
  product policy. The KL of a product is the *sum* of the per-agent KLs, so
  using the mean keeps the radius on the same scale as the published constant
  rather than `n_agents` times tighter. A strictly per-agent trust region would
  need a separate conjugate-gradient solve per agent, which torchrl's stacked
  parameters do not express.

### (D) Compute

The conjugate gradients run two Fisher-vector products per iteration, each a
double backward over `trpo_batch` joint states — `cg_iters: 15` means 30 such
products per policy step, plus up to 10 line-search evaluations. `trpo_batch:
1024` bounds that; `0` uses the whole rollout, which is LCPO's behaviour (its
rollouts are 512–8192 transitions and ours is 6000 joint states). This is a
compute bound, not an algorithmic change, and it is the one knob here that is
neither from the paper nor from the code.

## Running it

```bash
GROUP=b6 bash scripts/run_baselines.sh
```

The launcher sets `task.ns_observe_driver=true`,
`experiment.on_policy_minibatch_size=$BATCH`,
`experiment.on_policy_n_minibatch_iters=1` and `algorithm.ood_window=$BATCH`.

## What to watch

| column | what it tells you |
|---|---|
| `lcpo_branch` | 0 = the A2C fallback, 1 = the trust-region step. **If this stays at 0 the method never ran**: the reservoir never found a batch of out-of-distribution contexts. Lower `ood_threshold` (the paper sweeps 0.25/0.5/1/2/4) or check that the driver is actually in the observation. |
| `lcpo_ood_batch` | how many OOD states the step used. 0 with `branch=1` is impossible. |
| `lcpo_kl_out_of_d` | the realised out-of-distribution KL. It must sit at or under `kl_out`; if it is above, the line search is failing. |
| `lcpo_kl_in_d` | the realised in-distribution KL, against `kl_in`. |
| `lcpo_linesearch_ok` | 0 means every backtrack was rejected and the policy did not move at all. A long run of zeros means the step direction is bad — usually `damping` too small. |
| `lcpo_step_norm` | mean absolute parameter change. Zero for many iterations with `branch=1` is the same failure. |
| `lcpo_entropy_factor` | the annealed coefficient, falling from 0.1 by 2e-5 per rollout. |
