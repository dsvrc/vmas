# LIAM — BASELINES.md B5, agent modelling under partial observability

## Source

| what | where |
|---|---|
| paper | Papoudakis, Christianos, Albrecht, *Agent Modelling under Partial Observability for Deep Reinforcement Learning*, NeurIPS 2021 ([PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/a03caec56cd82478bf197475b48c05f9-Paper.pdf)) |
| code read | [uoe-agents/LIAM](https://github.com/uoe-agents/LIAM): `lb_foraging/models.py` (`Encoder`, `Decoder`, `PolicyNet`) and `lb_foraging/agent.py` (`class A2C`: `compute_embedding`, `act`, `evaluate`, `eval_decoding`, `update`) |
| implementation | [`benchmarl/algorithms/liam.py`](../../benchmarl/algorithms/liam.py), [`benchmarl/conf/algorithm/liam.yaml`](../../benchmarl/conf/algorithm/liam.yaml) |

## What the baseline is for

BASELINES.md B5: the opponent/teammate-modelling line. The prediction it tests
is sharp — *it models **who** the peers are (their policies), which at σ=0 is
stationary; it does not model **how much they matter**, which is what drifts.*

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | the encoder reads the controlled agent's **own** observation and **own last action** | `input_tensor = torch.cat((obs, action), dim=-1)` into an LSTM | an LSTM over a window of the agent's own observations; `ns_observe_prev_action=true` makes each frame `(o_t, a_{t-1})` | **met, by launch flag** |
| 2 | the encoder is `LSTM -> fc1(ReLU) -> embedding` | `models.Encoder` | `LiamEncoder`, same three layers | **met** |
| 3 | the encoder is recurrent over the **whole episode** | the hidden state is carried and reset per episode | a fixed `history_len` window, default 25 of a 100-step episode | **adapted** — see (A) |
| 4 | the decoder reconstructs the **modelled agents' observations** | `out1 = self.out1(h1)`, loss `0.5*((obs - out)**2).sum(-1)` | `LiamDecoder.obs_head`, the same squared error | **met** |
| 5 | the decoder reconstructs the **modelled agents' actions** | `out2 = softmax(...)`, loss `-log(sum(probs * onehot))` | continuous actions: the same squared-error head as the observation branch. Discrete actions keep the softmax/cross-entropy branch | **adapted** — see (B) |
| 6 | "modelled agents" means every agent except the controlled one | the reference has one controlled agent and models the rest | `others_view`: row `i` holds every agent's value except `i`'s | **met, tested** (`verify.py`, including that nobody appears in their own target row) |
| 7 | the decoder targets are **training-time only**; the policy never sees them | the decoder is not in the policy path | the decoder is not part of the actor network at all | **met** — execution stays decentralised |
| 8 | the policy acts on `[own obs, embedding]` | `torch.cat((obs.unsqueeze(0), embedding), dim=-1)` | `LiamPolicyInput` | **met** |
| 9 | **the embedding is detached before the policy** | `embeddings.detach()` in `evaluate` | `embedding.detach()` in `LiamPolicyInput` | **met — this is the method.** Without it the RL gradient shapes the embedding and the reconstruction loss is decoration |
| 10 | **two optimisers**: RL on the policy, reconstruction on encoder+decoder | `optimizer1` = actor-critic, `optimizer2` = encoder+decoder | three: `loss_objective` (policy), `loss_critic`, `loss_liam` (encoder+decoder). The partition is **checked at construction** and raises if it fails | **met** |
| 11 | both losses are masked on terminal transitions | `(1 - dones_batch.float()) * (...)` | `mask_done: true`, using `("next", group, "done")` | **met** for the reconstruction loss; the RL loss's terminal handling is torchrl's GAE, which is the same intent |
| 12 | gradient clipping on all three networks | `clip_grad_norm_(..., max_grad_norm)` | BenchMARL's `experiment.clip_grad_norm` / `clip_grad_val`, applied per optimiser | **met** |
| 13 | the RL half is **A2C** | `advantages = returns - values`, `-adv * log_prob` | IPPO | **adapted** — see (C) |
| 14 | the value head shares the trunk with the policy and also sees the embedding | `PolicyNet` has both heads | BenchMARL's critic is a separate network on the plain observation | **adapted** — see (D) |
| 15 | a stochastic encoder / VAE | the reference's `MLP` class has `m_z`/`var_z` heads but `A2C` does not use them | not used here either | **met, as the reference has it** |

### (A) The window, instead of the whole episode

LIAM's LSTM carries its hidden state across the episode. A policy in BenchMARL
is called one step at a time during collection with no memory of its own, so a
history can only reach it from the environment. `benchmarl/algorithms/_history.py`
stacks the last `history_len` observations into a separate key with torchrl's
`CatFrames` (behind a `RenameTransform` copy, so `(group, "observation")` is
untouched and the critic, the debug CSV and every other consumer see exactly
what they see on every other arm).

The LSTM then runs over that window and takes its last output — the same
architecture, over a truncated history. `history_len: 25` on a 100-step
episode; raise it to 100 to remove the truncation entirely, at four times the
encoder cost.

The alternative — making the policy itself recurrent through BenchMARL's RNN
plumbing — would put the encoder *inside* the policy and destroy row 9, which
is the method.

### Memory: the history window is stored per transition

`CatFrames` writes a `history_len * obs_dim` vector into every transition, and
the on-policy replay buffer holds a whole collection batch. At the launcher's
defaults that is

    BATCH x n_agents x history_len x obs_dim x 4 bytes
    = 30000 x 4 x 25 x ~20 x 4  ~  228 MB

on top of everything else. If that is too much, lower `BATCH` for this row
rather than `history_len` -- the window length is the published value and the
batch size is not.

### (B) The action reconstruction head

The reference's action head is a softmax over a discrete action set scored by
cross-entropy. This host's actions are continuous, so the head becomes the same
Gaussian/squared-error head the observation branch already uses — which is what
LIAM's own observation branch is. The discrete branch is kept in the code and
is used when the task is run with discrete actions.

### (C) PPO in place of A2C

BenchMARL ships no A2C. IPPO is the same setting — independent learners, an
independent critic per agent, an entropy bonus — with a clipped surrogate
instead of the vanilla policy gradient. Everything LIAM adds sits on top
unchanged, and the comparison that matters is `liam` against `ippo` at the same
σ, which the launcher runs.

`entropy_coef` is left at `0.01` rather than the `0.0` the other on-policy
rows use, because LIAM's reference passes an entropy coefficient and this row
is read against IPPO rather than against MAPPO. Set it to `0.0` for the
strictly matched pair.

### (D) The critic does not see the embedding

LIAM's `PolicyNet` produces the policy and the value from the same trunk, so
its value head sees the embedding too. BenchMARL's critic is a separate network
built from the observation spec.

What is lost is the value function's access to the teammate embedding, which
makes the advantage estimate slightly noisier. What is kept — and is what B5 is
about — is that the **policy** acts on the embedding. This is a listed
deviation, not a silent one.

## Running it

```bash
GROUP=b5 bash scripts/run_baselines.sh
```

## What to watch

| column | what it tells you |
|---|---|
| `liam_recon_obs` | how well the decoder predicts the peers' observations. If it never falls, the encoder is learning nothing and the row is IPPO with a wider input. |
| `liam_recon_act` | the same for their actions. This is the one that should fall first — the peers' policies are the stationary part. |
| `loss_liam` | the weighted sum the encoder+decoder optimiser actually steps. |

The prediction to check: `liam_recon_*` falls (it **is** modelling the peers)
while the return does **not** recover (modelling who they are does not tell it
how much they matter). If both happen, B5's argument is made by measurement
rather than by assertion.
