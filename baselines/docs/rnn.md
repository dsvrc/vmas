# GRU-MAPPO / R-MAPPO — BASELINES.md B2, memory

## Source

| what | where |
|---|---|
| paper | Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022 ([arXiv 2103.01955](https://arxiv.org/abs/2103.01955)) |
| reference implementation | [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy) — `R_MAPPOPolicy`, `R_Actor`, `R_Critic`: an MLP base, a GRU layer over it, then the action head |
| host implementation | BenchMARL's own `Gru` model — [`benchmarl/models/gru.py`](../../benchmarl/models/gru.py), [`benchmarl/conf/model/layers/gru.yaml`](../../benchmarl/conf/model/layers/gru.yaml) |

## What the baseline is for

BASELINES.md B2, the cheapest reviewer objection: *a history-conditioned policy
could infer the drift from its own past*. The prediction is that it does not
recover, because the quantity the agent needs — the peers' next actions through
`W` — is not a function of the agent's own history. Degradation persists.

**No code was written for this row.** BASELINES.md B2 says so explicitly ("a
configuration change, no code"), and it is true: BenchMARL ships the recurrent
model, the hidden-state transforms and the sequence-shaped replay buffer.

## The checklist

| # | R-MAPPO's construction | here | status |
|---|---|---|---|
| 1 | a recurrent layer between the feature extractor and the action head | `model=layers/gru`: BenchMARL's `Gru` is `MLP -> GRU -> MLP`, one GRU layer of 128 by default | **met** |
| 2 | the hidden state is carried across steps during collection and reset on `done` | BenchMARL installs `_add_rnn_transforms` when `model_config.is_rnn`, which adds the hidden-state primer and the reset handling | **met** |
| 3 | training happens on **sequences**, not shuffled transitions | `Algorithm.has_rnn` switches the replay buffer to sequence storage and the experiment stops flattening the batch | **met** |
| 4 | the critic is recurrent too (`R_Critic`) | the launcher leaves `model@critic_model=layers/mlp`, so only the **actor** has memory | **adapted** — see below |
| 5 | shared parameters across agents | BenchMARL's `share_policy_params=True` default, as in MAPPO | **met** |
| 6 | everything else is MAPPO | it is literally `algorithm=mappo` | **met** |

### Why the critic stays an MLP

The objection B2 answers is about the **policy**: "give the policy memory and
it will infer the drift". A recurrent critic changes the advantage estimate as
well, which mixes two effects into one row. Leaving the critic as an MLP makes
`mappo_gru` differ from `mappo` in exactly one thing.

If a reviewer asks for the full R-MAPPO with both networks recurrent, that is
one more override:

```bash
ONLY=mappo_gru EXTRA="model@critic_model=layers/gru" bash scripts/run_baselines.sh
```

BenchMARL's `has_rnn` already accounts for a recurrent critic
(`self.critic_model_config.is_rnn and self.has_critic`), so nothing else
changes.

## Two rows, not one

`scripts/baselines_common.sh` runs both:

* `mappo_gru` — the centralised-critic version, which is R-MAPPO proper;
* `ippo_gru` — the independent-critic version, because the blind arm the
  degradation is measured against is run for both IPPO and MAPPO.

## Running it

```bash
GROUP=b2 bash scripts/run_baselines.sh
```

## What to watch

* The GRU makes each optimiser step markedly more expensive and shrinks the
  effective minibatch count (BenchMARL divides the memory and sampling sizes by
  the sequence length). Compare against `mappo_blind` at the **same frame
  budget**, not the same wall clock.
* If `mappo_gru` recovers a large part of the gap to B0, the drift is
  inferable from the agent's own history and the paper's central claim is in
  trouble — that is exactly what this row is for.
