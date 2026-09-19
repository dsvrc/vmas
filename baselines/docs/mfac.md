# MF-AC — BASELINES.md B4, mean-field MARL

## Source

| what | where |
|---|---|
| paper | Yang, Luo, Li, Zhou, Zhang, Wang, *Mean Field Multi-Agent Reinforcement Learning*, ICML 2018 ([PMLR v80](https://proceedings.mlr.press/v80/yang18d/yang18d.pdf)) |
| code read | [mlii/mfrl](https://github.com/mlii/mfrl); the PyTorch port [deligentfool/mfrl_pytorch](https://github.com/deligentfool/mfrl_pytorch): `algo/base.py` (`prob_emb_linear`, `calc_target_q`, `act`, `train`), `senarios/*.py` (how `former_act_prob` is formed) |
| implementation | [`benchmarl/algorithms/mfac.py`](../../benchmarl/algorithms/mfac.py), [`benchmarl/conf/algorithm/mfac.yaml`](../../benchmarl/conf/algorithm/mfac.yaml) |

## Read BASELINES.md B4 first

B4's own decision is **do not port**:

> The information-matched blind arm (the channels, i.e. the weighted mean
> neighbour action, in the observation) *is* a mean-field-style policy on this
> problem; report it under that name and cite MF-Q/MF-AC. Zero cost.

That argument is sound and it stands. This row exists anyway, because showing
the method costs about a hundred lines and lets the paper report the pair
rather than argue it:

| | where the neighbours enter | what the mean is |
|---|---|---|
| information-matched blind | the **policy's** input | the **weighted** mean through `W`, per declared class |
| **MF-AC (this row)** | the **critic's** input | the **unweighted** mean over `N(j)` |
| PACT | a declared estimator below the policy | the weighted mean, with the gain estimated |

If they behave the same, B4's argument is confirmed by measurement. If they do
not, the paper needs to know which of the three differences did it.

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | the joint action is replaced by the agent's own action plus a **mean action** | `Q^j(s, a^j, abar^j)` | the critic reads `[observation, mean action]` | **met** |
| 2 | `abar^j` is the mean over the **neighbourhood N(j)**, which excludes `j` | paper, Eq. (4) | `mean_action(..., include_self=False)`, the default | **met, tested** (`verify.py`) |
| 3 | the reference implementation uses the **team mean**, self included, tiled to every agent | `former_act_prob = np.mean(one_hot(acts), axis=0)` then `np.tile(...)` | `include_self: true`, run as the `mfac_team` row | **met, both variants run** |
| 4 | the mean is the **previous** step's actions | `former_act_prob` is passed to `act` and only updated afterwards | read off the observation, where `ns_observe_prev_action` puts each agent's last executed action | **met** — see (A) |
| 5 | the mean enters through its own embedding | `prob_emb_linear = Linear -> ReLU -> Linear` | the critic MLP sees `[observation, mean action]` concatenated; BenchMARL's model config decides the layers | **adapted, cosmetic**: a separate two-layer embedding before the concatenation is an architecture choice, not a property of the method |
| 6 | the actor is the agent's own policy, without the mean | MF-AC's actor is `pi^j(a | s)` | the actor input is untouched — only `get_critic` is overridden | **met** |
| 7 | each agent has its **own** critic | `Q^j` | `centralised=False`, which is IPPO's independent critic | **met** |
| 8 | the action is a probability vector (discrete) | one-hot actions, averaged | continuous actions, averaged | **adapted**: the mean of one-hot vectors is the empirical action distribution; the mean of continuous action vectors is the direct analogue, and it is what the coupling operator actually averages here |
| 9 | MF-**Q**: a Boltzmann policy over `Q(s, ·, abar)` | `softmax(e_q / temperature)` | **not implemented** | **NOT met** — MF-Q needs a discrete action set; this host is continuous. MF-AC is the continuous member of the pair and the one BASELINES.md names. |
| 10 | the mean action is iterated to a fixed point | paper, Alg. 1 | the reference code does not iterate either: it carries the previous step's mean | **met**, as the reference has it |

### (A) Where the mean comes from, and why nothing was added to the environment

The reference keeps `former_act_prob` as training-loop state because at
decision time the current actions do not exist yet. Here the previous step's
actions are already in the data: `task.ns_observe_prev_action=true` appends
each agent's own last executed action to its observation, so the mean over
agents is a two-line reduction of a tensor that is already in the batch.

That means:

* no new environment state, no new transform, no extra key in the replay
  buffer;
* the mean is available identically during collection and during training;
* the slice is resolved from the action spec and **printed at construction**,
  and the algorithm raises if it does not fit — so it cannot silently average
  the wrong columns.

`verify.py` checks `mean_action` against both definitions by hand, including
that a lone agent's `N(j)` mean is exactly zero — the same category-C identity
everything else in this repo is built on.

## Running it

```bash
GROUP=b4 bash scripts/run_baselines.sh      # runs mfac and mfac_team
```

## What to watch

The banner prints which mean it took. Beyond that, this row has no diagnostics
of its own: it is IPPO with a wider critic input, so read it against
`mappo_blind`/`ippo` at the same σ and against the information-matched blind
arm.
