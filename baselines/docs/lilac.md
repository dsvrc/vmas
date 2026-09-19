# LILAC — BASELINES.md B7, latent-variable non-stationary RL

## Status: **NOT IMPLEMENTED**

BASELINES.md B7 itself allows this: *"MAPDN `case33`, Tier 2 (**or skip and
cite if time is short**)"*. This document records what was read, what the
method needs, exactly which piece BenchMARL cannot express today, and the
design that would close it — so the decision is reviewable and the work is not
repeated from scratch.

## Source

| what | where |
|---|---|
| paper | Xie, Harrison, Finn, *Deep Reinforcement Learning amidst Continual Structured Non-Stationarity*, ICML 2021 ([PMLR v139](https://proceedings.mlr.press/v139/xie21c.html)) |
| code read | [EmiliyanGospodinov/LILAC](https://github.com/EmiliyanGospodinov/LILAC), a softlearning fork: `softlearning/softlearning/algorithms/sac.py` — `_init_encoder_update` (the context encoder, the LSTM prior, the KL, the decoder) and the `env_latents` threading through `_init_actor_update` / the Q updates |

## What the method is

SAC conditioned on a latent that is inferred **per episode**, with a learned
prior over the *sequence* of episodes:

1. **Context encoder** `q(z | o, a, r, o')`. Deterministic by default
   (`stochastic_encoder=False`); a stochastic Gaussian head is the option.
2. **Per-episode latent**: with `batch_tasks`, the transition latents are
   averaged within an episode and the mean is used for every transition in it.
3. **Latent prior**: an `LSTMCell` run over the sequence of *all past episode
   latents*, `z_0 = 0`; the prior for episode `t` is the LSTM's output at
   `t-1`.
4. **KL term**: deterministic case `sum((mu - prior)^2)`; stochastic case the
   Gaussian KL against a fixed `prior_sigma`.
5. **Decoder** `p(r, Δo | o, a, z)` — in the released code the Δo term is
   commented out, so the reconstruction is the **reward** only.
6. SAC's policy and both Q functions are conditioned on `(o, z)`, and the
   encoder is trained by the Q loss **as well as** by the KL and the
   reconstruction (`_Q_encoder_training_op`).

## Why it is not implemented here

Pieces 2 and 3 need something BenchMARL's off-policy path does not have: an
**episode identity carried with every stored transition**.

* The replay buffer stores flat transitions. Nothing in them says which episode
  they came from, and `done` alone does not survive random sampling.
* The prior is a function of the *whole history of episodes*, so a sampled
  transition needs to index into a table of cached episode latents. That means
  a global episode counter that is consistent between collection, the buffer
  and the loss.
* The LSTM over that table cannot be run inside a loss that is called ~1000
  times per collection iteration; it has to be computed once per iteration and
  cached, which factorises the joint objective into two coordinate steps.

None of those three is impossible. All three are **untestable offline**: this
machine has no `torchrl`, `tensordict` or `vmas`, so an implementation of them
could not be checked before shipping — and an off-by-one in an episode index is
exactly the kind of error that produces a plausible training curve and a wrong
conclusion.

BASELINES.md permits the skip, so the skip is taken rather than a version that
would have to be marked "probably correct".

## What a correct implementation would need

Written down so the next pass is a day, not a week:

1. **Episode index.** In `Algorithm.process_batch` the off-policy batch is
   still shaped `(n_envs, T)`. Cumulative-sum the `done` flags along `T`, add a
   per-world running counter, and assign each `(world, episode)` segment a
   unique global index. Write it into the group tensordict as
   `(group, "lilac_episode")`; the buffer stores it with everything else.
2. **Latent cache.** A bounded table of the most recent `max_episodes` episode
   latents, written once per collection iteration from the mean encoder output
   of each completed segment. Bounded, not unbounded: LILAC keeps every episode
   and at a 3 M-frame budget that is ~30 000.
3. **Prior table.** Run the `LSTMCell` over the cache once per iteration in
   `process_batch`, step its own optimiser on
   `sum((z_cached.detach() - prior)^2)`, and cache the prior **detached** for
   the loss to gather by episode index. This is coordinate descent on LILAC's
   joint objective, not a different objective, and it is the only way the
   sequence model can be afforded at 1000 optimiser calls per iteration.
4. **The rest is ISAC.** The encoder and decoder are two MLPs; the policy and
   Q input becomes `[o, z]` via the same `TensorDictModule` pattern
   `benchmarl/algorithms/rma.py` already uses; the encoder's parameters go in
   their own optimiser entry so the KL and the reconstruction reach it
   separately from the Q loss.

## What is cited instead

* **LILAC** itself, as the representative of B7's class, with the prediction
  BASELINES.md states: *it infers a latent per episode, which lags a
  within-episode driver and, again, cannot cancel.* Our driver has a 100-step
  period and the episodes are 100 steps, so a per-episode latent is constant
  exactly where the driver is doing all its moving — the prediction is
  structural here, not empirical.
* **MBCD** ([arXiv 2105.09452](https://arxiv.org/abs/2105.09452),
  [LucasAlegre/mbcd](https://github.com/LucasAlegre/mbcd)) — change-point
  detection, which assumes discrete context switches; ours is continuous.
* **FANS-RL** ([NeurIPS 2022](https://arxiv.org/abs/2203.16582)) — no public
  code.

The rows that *are* run and that cover the same objection from the other side:

* **LCPO** (B6) is the observed-context member of the same family and is
  implemented in full.
* **RMA/UP-OSI** (B8) infers a latent from history, which is the inference half
  of LILAC without the episode-level prior.

Between them, "the agent should infer the non-stationarity" is tested with both
an observed context and an inferred one. What is missing is specifically the
*structured prior over the sequence of contexts*, and the README says so.
