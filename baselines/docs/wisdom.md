# WISDOM — X4, wavelet predictive representations for non-stationary RL

## Source

| what | where |
|---|---|
| paper | Wang, Li, He, Li, Bennis, Islam, Wang, *Wavelet Predictive Representations for Non-Stationary Reinforcement Learning*, [arXiv 2510.04507](https://arxiv.org/abs/2510.04507) |
| code read | [MinWangcs/WISDOM](https://github.com/MinWangcs/WISDOM): `wisdom/networks.py` (`Y_Network`, `forward_fading`), `wisdom/agent.py` (`_product_of_gaussians`, `infer_posterior`, `compute_kl_div`, `get_action`), `wisdom/reconstruction_trainer.py` (the whole training step), `wisdom/sac.py` (`PolicyTrainer.training_step`), `wisdom/rollout_worker.py` (`rollout`, `update_context`, `build_encoder_input`), `wisdom/stacked_replay_buffer.py`, `configs/default.py`, `runner.py` |
| implementation | [`benchmarl/algorithms/wisdom.py`](../../benchmarl/algorithms/wisdom.py), [`benchmarl/conf/algorithm/wisdom.yaml`](../../benchmarl/conf/algorithm/wisdom.yaml), the `X4` block of [`_baseline_math.py`](../../benchmarl/algorithms/_baseline_math.py) |

The released tree is a fork of **CEMRL**, itself a fork of PEARL's `rlkit` —
the release still ships `wisdom/__pycache__/cemrl_algorithm.cpython-36.pyc`.
That matters; see (A).

## The method

1. **Context encoder.** `q(z|c)` over the last `time_steps` transitions
   `(o, a, r, o')`, built as PEARL's permutation-invariant product of
   Gaussians. `z` is "which MDP am I in".
2. **A learnable wavelet over `z`.** `Y_Network`: a causal à-trous
   decomposition with *learned* low-pass `h0` and high-pass `h1` filters, run
   to `depth` levels at doubling dilation. Its output `y` is a learned weighted
   sum of every detail band, the final approximation band, and the input.
3. **A wavelet TD operator.** `res_lo(z_t) <- z_t + gamma res_lo(z_{t+1})`,
   bootstrapped off a polyak-averaged target copy. Its fixed point is
   `sum_k gamma^k z_{t+k}` — **the discounted future of the task
   representation**. That is what makes it predictive rather than a filter, and
   it is the paper's contribution.
4. The policy and the critic see `[o, y(z)]`.

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | context = the last `time_steps` transitions `(o, a, r, o')` | `self.context = torch.zeros((time_steps, obs + act + 1 + obs))`, `update_context(o, a, r, next_o)` | a `CatFrames` window of `time_steps` observations; with `ns_observe_prev_action` and `ns_observe_prev_reward` on, frame `k` is `(o_k, a_{k-1}, r_{k-1})`, so consecutive frames carry every component of a transition | **met**, differently assembled — see (B) |
| 2 | the window is zeroed at the start of an episode | `self.context = torch.zeros(...)` per rollout | `CatFrames` refills on reset; the environment zeroes the action and reward columns on reset | **met** |
| 3 | one encoder per context element, then a **product of Gaussians** | `MlpEncoder` + `_product_of_gaussians` | `product_of_gaussians` over the window axis | **met, tested** (`verify.py`: `N(1,1) x N(3,1) = N(2, 1/2)`) |
| 4 | encoder `[200, 200, 200]`, `latent_size = 5`, `time_steps = 30` | `runner.py`, `configs/default.py` | `encoder_hidden: 200`, `latent_dim: 5`, `time_steps: 30` | **met** |
| 5 | `z` is **sampled** from the posterior, not the mean | `z = [d.rsample() for d in posteriors]` | `sample_z` | **met** |
| 6 | the wavelet: learned `h0`, `h1`, mixing vector `w` of length `depth + 2` | `Y_Network.__init__`, `forward_fading` | `YNetwork`, transcribed | **met, tested** — see (C) |
| 7 | `depth = 2`, `filter_size = 2`, `dimension = 1`, `dropout = 0.2`, GELU | `configs/default.py`, `runner.py` | the same five values | **met** |
| 8 | the wavelet runs over the LATENT COORDINATES, not over time | `task_z.view(batch, 1, latent_dim)` with `d_model = 1` | reproduced exactly, and the banner says so | **met**, and see (D) — this is not the obvious reading |
| 9 | prediction loss `MSE(pred_z, z)` | `pred_loss = F.mse_loss(pred_z, task_z)` | `pred_loss` | **met** |
| 10 | **wavelet TD loss**: `res_lo <- z + gamma res_lo_target(z')`, RMS error | `target_res_lo = task_z + gamma * next_res_lo`; `td_loss = sqrt(mean((res_lo - target)^2))` | `wavelet_td_target` + RMS | **met, tested** (fixed point is `1/(1-gamma)`) |
| 11 | a **target** wavelet network, soft-updated | `ptu.soft_update_from_to(z_model, target_z_model, 5e-3)` | `soft_update_target(wavelet_tau)`, called once per training step after the TD loss | **met** |
| 12 | `gamma = 0.99`, `tau = 5e-3`, `td_loss_coefficient = 0.1` | `ReconstructionTrainer` | `wavelet_gamma`, `wavelet_tau`, `td_coef` | **met** |
| 13 | `z_loss = pred_loss + td_coef * td_loss`, its own optimiser | `z_optimizer` | `loss_wisdom_z`, its own BenchMARL optimiser | **met** |
| 14 | the encoder has its **own** optimiser and no SAC gradient | `optimizer_encoder`; `new_task_z.detach()` in the SAC step | `loss_wisdom_enc`; the encoder's output is detached into the policy | **met** |
| 15 | the encoder's objective is the KL to `N(0, I)` **and nothing else** | `kl_loss = 0.1 * kl_div; kl_loss.backward()` — there is no decoder in the tree | `encoder_loss: kl_only` reproduces it; **the default adds CEMRL's decoder** | **adapted, both available** — see (A) |
| 16 | `kl` weight | code hardcodes `0.1`; the config says `alpha_kl_z = 1e-3` and is never read | `kl_coef: 0.1` — the code wins, and both are recorded | **met**, discrepancy noted |
| 17 | SAC: twin Q, `delay_qvalue`, auto alpha, squashed Gaussian | `PolicyTrainer` | inherited from BenchMARL's ISAC, verbatim from `isac.yaml` | **met** |
| 18 | the policy and critic read `[o, pred_z]` | `obs = cat((obs, new_task_z.detach()))` for BOTH `obs` and `next_obs` | `[observation, y(z)]`, recomputed for the current and the next step | **met**, one inconsistency removed — see (E) |
| 19 | at COLLECTION time the policy reads `[s, z]`, not `[s, pred_z]` | `agent.get_action`: `policy_input = cat([state, task_z])` | `y(z)` is used at both collection and training | **adapted** — see (E) |
| 20 | `z` is read from the buffer (stale) during training | `task_z = batch['task_indicators']` | recomputed from the stored window with the current encoder | **adapted** — see (E) |
| 21 | context normalisation | `use_data_normalization=True`, `StackedReplayBuffer.normalize_data` | `RunningNormalizer` (Welford), updated from the training path only | **met** — see (F) |
| 22 | per-agent encoders / a multi-agent lift | the reference is **single-agent** | ISAC host: per-agent encoder, per-agent critic, per-agent wavelet share parameters exactly as `share_policy_params` says | **adapted, structural** |
| 23 | `use_parametrized_alpha` (alpha conditioned on `z`) | `alpha_net` | **not implemented**; the config's own default is `False` | **NOT met**, off in the reference too |
| 24 | the meta-RL task sampler (`n_train_tasks`, `num_train_tasks_per_episode`) | `configs/default.py` | **not applicable**: there is one task whose dynamics vary in time, which is the setting the paper's title is about | **out of scope** |
| 25 | discrete actions | — | **not implemented**; the reference is SAC with a squashed Gaussian | **NOT met** |

### (A) The released code cannot work as released — and what was done about it

`ReconstructionTrainer.training_step` trains the encoder like this, and only
like this:

```python
_, z_means, z_vars = self.agent.infer_posterior(encoder_input)
kl_div  = self.agent.compute_kl_div(z_means, z_vars)
kl_loss = 0.1 * kl_div
...
self.optimizer_encoder.zero_grad()
kl_loss.backward()
self.optimizer_encoder.step()
```

There is **no decoder anywhere in the released tree** — `encoder_decoder_networks.py`
contains `MlpEncoder` and nothing else — and the SAC trainer takes `task_z`
from the replay buffer, detached, so no critic gradient reaches the encoder
either. An encoder whose only objective is `KL(q(z|c) || N(0,I))` is minimised
by **ignoring its input**: `z` collapses to the prior and the wavelet acts on
noise.

The upstream codebase this is forked from (CEMRL — the `.pyc` is still in the
release) trains the encoder with a **decoder**: predict `(o', r)` from
`(o, a, z)`, plus `beta * KL`. That is almost certainly what was dropped when
the tree was pared down for release.

Both are available and neither is hidden:

* `encoder_loss: kl_only` — the release, exactly. Run it if a reviewer asks
  what the published code does. `wisdom_z_std` should fall toward 0.
* `encoder_loss: reconstruction` — **the default**. CEMRL's decoder restored,
  which is the only version in which the method described in the paper can do
  anything.

The launcher runs both (`wisdom`, `wisdom_release`).

### (B) The context window

The reference's rollout worker maintains `(o, a, r, o')` rows explicitly. Here
the environment puts the agent's own previous action and the reward of the
transition that produced the observation *into the observation*
(`ns_observe_prev_action`, `ns_observe_prev_reward`), so a `CatFrames` window
of observations already **is** a window of transitions: frame `k` carries
`o_k`, `a_{k-1}`, `r_{k-1}`, and frame `k+1` supplies `o'`. Same information,
no new transform, and the window is built by the same machinery RMA (B8) and
LIAM (B5) already use.

### (C) What can be checked about a learned wavelet

The filters are learned, so there is no fixed answer to compare against. What
`verify.py` checks is the *structure*, which is where a transcription error
would live: with an identity low-pass and a zero high-pass, every approximation
band must equal the input; and `w[:, -1]` must be the input's own skip
connection. Both hold, so the padding, the dilation schedule and the band
bookkeeping line up with `forward_fading`.

### (D) `d_model = 1`: the wavelet runs across the latent, not across time

This is the one thing in the reference that is easy to get wrong by assuming.
`Y_Network` is fed `task_z.view(batch, 1, latent_dim)` with
`wavelet_params.dimension = 1`, i.e. **one channel and a "sequence" whose
length is the latent dimension**. The decomposition is therefore across the
coordinates of `z`. The *temporal* structure enters only through the TD
operator, which relates `res_lo(z_t)` to `res_lo(z_{t+1})`. Reproduced as
released, and the construction banner prints it so nobody has to re-derive it
from the code.

### (E) Three staleness decisions, and the reference's own inconsistency

The released code is inconsistent with itself here:

* at **collection** the policy reads `[s, z]` (`agent.get_action`);
* at **training** it reads `[s, pred_z]` (`sac.py`), with `pred_z` obtained by
  pushing a **buffer-stored** `z` through the *current* wavelet.

So the network that acts and the network that is trained see different inputs,
and the `z` used in training was produced by an encoder thousands of gradient
steps old.

Here: `y(z)` is used at **both** collection and training, and `z` is
**recomputed** from the stored window with the current encoder. The two
properties the method rests on are preserved — the representation the policy
sees is the wavelet's output, and neither the encoder nor the wavelet receives
any SAC gradient (the output is detached, as `new_task_z.detach()` is) — while
the acting and trained inputs are the same object. This is an adaptation and it
is a departure from the released code; it is the one place where "reproduce the
reference exactly" and "reproduce a coherent method" disagree, and coherence
was chosen. A reviewer should know.

Mechanically, the critic reads the key the *actor* writes, and the loss
refreshes it for the current and next step before delegating to SAC, so every
term in one update sees the same `y(z)`. The encoder is deliberately **not** a
submodule of the critic: torchrl expands the Q network's parameters by
`num_qvalue_nets`, which would make two frozen copies of the encoder and leave
the one the policy uses training alone.

### (F) Normalisation is not cosmetic

The context carries a **reward** channel next to observation channels that are
`O(1)`; on `balance` a per-step reward can be two orders of magnitude larger.
The reference normalises with the buffer's running statistics and it has to.
`RunningNormalizer` updates only from the training path, so the statistic does
not depend on how many parallel workers collected.

## Running it

```bash
GROUP=x4 bash scripts/run_extra_baselines.sh   # wisdom, wisdom_release
```

**Memory.** The window is stored per transition, so the off-policy buffer grows
by `memory_size * n_agents * time_steps * obs_dim`. The launcher sets
`off_policy_memory_size=100000` (`WISDOM_BUFFER`) rather than shortening the
window, which is the published value.

## What to watch

| column | what it means |
|---|---|
| `wisdom_z_std` | the spread of `z` across the batch. **Falling toward 0 means the latent has collapsed** — which is exactly what `encoder_loss=kl_only` should do, and what `reconstruction` exists to prevent. |
| `wisdom_recon` | the decoder's error. Should fall; if it does not, `z` is not informative about the dynamics and the wavelet has nothing to predict. |
| `wisdom_kl` | the KL to the prior. |
| `wisdom_pred` | `MSE(y(z), z)`. |
| `wisdom_td` | the wavelet TD error — **the paper's own contribution**. If this does not fall, the representation is not predictive and the row is ISAC with a wider input. |
