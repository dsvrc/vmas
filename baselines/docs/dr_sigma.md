# Domain randomisation over σ, and the oracle-driver grant

Two environment flags, no new algorithm. BASELINES.md calls the first a
**must**.

---

## 1. Domain randomisation over σ — BASELINES.md B9

> **Must (zero cost): domain randomisation over σ** — resample σ ∈ [0, 3] per
> episode (a NEW environment flag, ≈ 10 lines), train the stock learner,
> evaluate at the committed σ.
>
> Reference for the wrapper pattern: [RRLS](https://arxiv.org/html/2406.08406v1),
> [SafeRL-Lab/Robust-RL-Baselines](https://github.com/SafeRL-Lab/Robust-RL-Baselines).

### The checklist

| # | what B9 asks for | here | status |
|---|---|---|---|
| 1 | σ resampled **per episode** | `ExertionMixin.reset_world_at` calls `_draw_sigma(env_index)`; nowhere else | **met** |
| 2 | σ ~ U[0, 3] | `ns_dr_low: 0.0`, `ns_dr_high: 3.0` | **met** |
| 3 | the draw is per **parallel world** | `self._sigma` is a `(batch_dim,)` tensor and each world resets independently | **met** |
| 4 | the stock learner is trained, unchanged | `algorithm=mappo`, nothing else | **met** |
| 5 | evaluate at the committed σ | a **separate run** — see below | **met, as a two-step procedure** |
| 6 | the randomisation must not perturb anything when off | `_scale_severity` returns its argument unchanged when `ns_dr_enabled` is false — not multiplied by 1.0, *skipped* | **met, checked** by `verify.py` |

### How it works

β\* is **linear in σ**, so drawing σ per episode is exactly "compute the
disturbance at σ=1 and scale it". When `ns_dr_enabled` is on the layer sets its
nominal `ns.severity` to 1.0 — printing a loud banner saying it has ignored
`ns_severity` — and multiplies the load by the per-world draw. No division, no
special case, and the σ=0 identity is untouched because the whole hook is
skipped when the flag is off.

`ns_sigma` appears in the info dict, so "what severity was this trajectory
actually at" is answerable from the logs rather than from the config.

### Evaluating at the committed σ

The training and evaluation environments are built from the same task config,
so a DR run evaluates under DR too. That is not what B9 asks for. The procedure
is:

```bash
# 1. train with the randomisation on (the launcher does this)
ONLY=dr_sigma bash scripts/run_baselines.sh

# 2. evaluate the checkpoint at the committed severity
python benchmarl/evaluate.py \
  runs/baselines/balance/sigma2.0/s0/dr_sigma/*/checkpoints/checkpoint_*.pt
```

`benchmarl/evaluate.py` reloads the experiment from its checkpoint; override
`task.ns_dr_enabled=false task.ns_severity=2.0` in the reloaded config so the
evaluation environment is the committed one. `experiment.checkpoint_at_end` is
already `true` in `scripts/baselines_common.sh`, so the checkpoint exists.

**This step is not automated** and it is the one place where reading this row
requires a manual command. Doing it inside the training run would need the
scenario to know whether it is the training or the evaluation environment, and
BenchMARL builds both through the same `get_env_fun` with no signal that
distinguishes them.

---

## 2. The oracle-driver grant — BASELINES.md §D item 4

> Oracle-driver blind (`--set ns_observe_driver=true`) — MAPDN `case33` (flag
> exists) and URB if the observation change is cheap.

`task.ns_observe_driver=true` appends the driver `A(t)` to every agent's
observation. This is **not a published method** — it is an upper bound on what
observing the context alone can buy, and it is the control for LCPO (B6) and
for RMA's teacher (B8).

### What is granted, and what is not

| granted | withheld |
|---|---|
| `A(t)`, the driver | σ, the severity |
| | β\*(t), the per-class gains |
| | the peers' draw `x`, i.e. the channels |
| | any other agent's residual |

`A(t)` is a function of observable time that no agent influences, so publishing
it is a grant of **public** information. Knowing the weather is not knowing the
neighbours' load under it — that gap is the whole of what the peer channels
buy, and this row is what measures it.

### Running it

```bash
GROUP=grants bash scripts/run_baselines.sh
```

---

## 3. `ns_observe_prev_action`

A third flag, added for B8 (RMA/UP-OSI needs a state-action history), used also
by B3's GNN, B4's mean-field critic and B5's LIAM encoder. It appends the
agent's **own** last executed action, divided by its action range.

This is proprioception, not privilege: the agent already measures the force it
delivered — that is what the `residual` column reports — so its own last action
is not new information to it. Off by default, and `verify.py` checks that all
four task yamls leave it off.
