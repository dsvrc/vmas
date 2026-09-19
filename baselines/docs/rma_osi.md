# RMA / UP-OSI — BASELINES.md B8, meta-RL / online system identification

## Source

| what | where |
|---|---|
| paper (RMA) | Kumar, Fu, Pathak, Malik, *RMA: Rapid Motor Adaptation for Legged Robots*, RSS 2021 ([PDF](https://www.roboticsproceedings.org/rss17/p011.html)) — code [antonilo/rl_locomotion](https://github.com/antonilo/rl_locomotion) |
| paper (UP-OSI) | Yu, Liu, Turk, *Preparing for the Unknown: Learning a Universal Policy with Online System Identification*, RSS 2017 ([arXiv 1702.02453](https://arxiv.org/abs/1702.02453)) — code [VincentYu68/policy_transfer](https://github.com/VincentYu68/policy_transfer) |
| implementation | [`benchmarl/algorithms/rma.py`](../../benchmarl/algorithms/rma.py), [`benchmarl/algorithms/_history.py`](../../benchmarl/algorithms/_history.py), [`benchmarl/conf/algorithm/rma.yaml`](../../benchmarl/conf/algorithm/rma.yaml) |

## What the baseline is for

BASELINES.md B8 asks for exactly this, **in-house**:

> a teacher policy conditioned on the true β\*(t)·A(t) — available inside the
> environment — then a history encoder distilled to predict it.

and calls it *the closest methodological neighbour of PACT: it also conditions
on an identified quantity, but the identifier is a learned history encoder
rather than an estimator on a declared basis.* The prediction: *it works while
excitation exists (training) and degrades in greedy play, and it never returns
to the host exactly (no floor property).*

## What the privileged quantity is

β\*(t) = σ · L · A(t) · send_m, with σ, L and `send` fixed before training. So
β\*(t) is **A(t) times a constant vector**: a teacher conditioned on A(t) is a
teacher conditioned on β\*(t), and μ's first linear layer absorbs the constant.

The environment publishes A(t) with `task.ns_observe_driver=true`, and this
algorithm removes it from everything the student sees — **by construction, not
by configuration**: the context columns are dropped from the policy's own input
and from every frame of the history window. There is no setting in which the
student is handed the quantity it exists to infer.

## The checklist

| # | idea | the reference | here | status |
|---|---|---|---|---|
| 1 | **phase 1**: a base policy conditioned on a latent of the true environment parameters | RMA §III-A: `a = pi(o, z)`, `z = mu(e)` | `RmaEncoder.forward` with `phase == 1` | **met** |
| 2 | μ is trained **jointly with the policy** by RL | RMA trains `mu` end-to-end in phase 1 | μ's parameters are inside the actor network and carry the policy gradient | **met** |
| 3 | the privileged parameters never reach the policy directly | `e` enters only through `mu` | the context slice is dropped from `o_public`; only `mu(e)` is concatenated | **met** |
| 4 | **phase 2**: an adaptation module φ predicts the latent from a **history of (state, action)** | RMA §III-B: a 1-D CNN over the last 50 state-action pairs | an MLP over a stacked window of the agent's own observations, which carry its own last action when `ns_observe_prev_action=true` | **adapted** — see (A) |
| 5 | φ is trained by **regression onto z**, not by RL | `|| zhat - z ||^2` | `loss_adapt`, with `z` detached and its **own optimiser** | **met** |
| 6 | in phase 2 the **base policy and μ are frozen** | RMA trains only φ in phase 2 | the base parameters are snapshotted at the phase boundary and written back after every step, and the latent the policy consumes is `phi(...).detach()` | **met** — see (B) |
| 7 | phase-2 data is collected **with φ in the loop** | RMA rolls out with `zhat` | phase 2 switches the policy's input to `zhat`, so all subsequent collection uses it | **met** — this is what makes φ cover its own induced state distribution |
| 8 | UP-OSI: the universal policy consumes the **parameters themselves**, and OSI regresses onto them | no latent | `predict_latent: false`, run as the `uposi` row | **met, both variants run** |
| 9 | φ is a feed-forward network over a fixed window | 1-D CNN | 3-layer MLP | **adapted, cosmetic**: same architecture class, same input, same target. A CNN over a 50-frame window and an MLP over the flattened window differ in parameter sharing across time, not in what is available to them |
| 10 | RMA's latent width | 8 | `latent_dim: 8` | **met** |
| 11 | RMA's history length | 50 | `history_len: 50` | **met** |
| 12 | the RL half | PPO | MAPPO (BenchMARL's PPO with a centralised critic) | **met in kind**; the launcher runs it against `mappo_blind` |
| 13 | training over a **distribution** of environment parameters | RMA randomises the terrain/dynamics | here the driver already sweeps A(t) over its whole range every 100 steps, so the teacher sees the full context distribution without randomisation | **met, structurally** — and if you want the parameter distribution widened too, that is the DR row (`ns_dr_enabled`), which composes with this one |

### (A) The history window

Neither RMA nor UP-OSI can get a history from inside a BenchMARL policy: the
policy is called one step at a time during collection and has no memory. So the
window is built by the environment, in `benchmarl/algorithms/_history.py`:
torchrl's `RenameTransform(create_copy=True)` makes a copy of the observation
under a new key and `CatFrames` stacks that copy in place, leaving
`(group, "observation")` untouched for the critic and every diagnostic.

`ns_observe_prev_action=true` puts the agent's own last executed action
(divided by its action range) in its observation, so each frame of the window
is `(o_t, a_{t-1})` — RMA's and UP-OSI's input verbatim. It is proprioception:
the jack already measures the force it delivered, which is what the `residual`
column reports.

### (B) The phase-2 freeze

RMA phase 2 trains **only** φ. Here the policy loss is still computed in phase
2, but every path into π and μ is through a detached latent, so their gradient
is exactly zero — and Adam would still move them from leftover momentum, about
`lr/(1-beta1)` per iteration. `RmaLoss` snapshots the base parameters at the
phase boundary and writes them back at the top of every forward, and
`_RmaFreezeCallback` does the final write-back after the last optimiser step of
the iteration. Same guard as HAPPO's, same reason; see [happo.md](happo.md) (A).

The partition between "base" and "φ" parameters is by identity against
`encoder.phi.parameters()` (torchrl's `convert_to_functional` keeps the same
`Parameter` objects), with a key-path fallback, and it is **checked at
construction**: if the counts do not match, the algorithm raises rather than
train φ with the policy gradient.

### Memory: the history window is stored per transition

`CatFrames` writes a `history_len * obs_dim` vector into every transition, and
the on-policy replay buffer holds a whole collection batch. At the launcher's
defaults that is

    BATCH x n_agents x history_len x obs_dim x 4 bytes
    = 30000 x 4 x 50 x ~20 x 4  ~  457 MB

on top of everything else. If that is too much, lower `BATCH` for this row
rather than `history_len` -- the window length is the published value and the
batch size is not.

### (C) `context_start` must match the host

simple_ns appends its extra observation columns in this order:

```
[ ... stock observation ... , residual, driver, prev_action_0 ... prev_action_{D-1} ]
```

so with `ns_observe_prev_action=true` and a D-dimensional action the driver
sits at `-(D + 1)`. D = 2 on every simple_ns host, hence `context_start: -3`.
The algorithm resolves and **prints** the slice, and raises if it does not fit
— it will not silently condition the teacher on the wrong column.

## Running it

```bash
GROUP=b8 bash scripts/run_baselines.sh      # runs rma and uposi
```

The launcher sets `task.ns_observe_driver=true`,
`task.ns_observe_prev_action=true` and `algorithm.phase1_frames=$((FRAMES/3))`.

## What to watch

| column | what it tells you |
|---|---|
| `rma_phase` | 1 or 2. The switch is also printed to stdout with the frame count. |
| `rma_adapt_mse` | `\|\|zhat - z\|\|^2`. **Read this at the phase boundary**: if it is still falling steeply when phase 2 starts, phase 1 was too short and the frozen policy is being handed a bad latent. |
| the return, across the boundary | RMA's own figure. A drop at the switch that recovers is the adaptation module learning; a drop that does not recover is the "no floor property" B8 predicts. |
