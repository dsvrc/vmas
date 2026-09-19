# GNN-MAPPO — BASELINES.md B3, graph / communication-structured policies

## Source

| what | where |
|---|---|
| the class | Graph-structured MARL policies. BASELINES.md's named alternatives are DGN ([ICLR 2020](https://openreview.net/forum?id=S8icDSeqfvy), [PKU-RL/DGN](https://github.com/PKU-RL/DGN)) and DCG ([ICML 2020](https://proceedings.mlr.press/v119/boehmer20a.html), [wendelinboehmer/dcg](https://github.com/wendelinboehmer/dcg)) |
| chosen representative | BenchMARL's `Gnn` model — [`benchmarl/models/gnn.py`](../../benchmarl/models/gnn.py), [`benchmarl/conf/model/layers/gnn.yaml`](../../benchmarl/conf/model/layers/gnn.yaml), documented in the BenchMARL paper (JMLR 2024) |

## What the baseline is for

BASELINES.md B3: *PACT uses the peers' broadcast actions through a **declared**
operator. The natural objection is that a policy receiving the neighbours'
actions through a learned graph network could learn the same coupling.* The
prediction is that it learns the σ=0 coupling and cannot track the drift,
because the gain is not identified as a separate quantity.

BASELINES.md chooses GNN-MAPPO as the representative and marks DGN/DCG
"optional port" — both are value-based and discrete, and this host's actions are
continuous. See [skipped.md](skipped.md).

**No code was written for this row either**, with one exception noted below.

## The checklist

| # | what the class needs | here | status |
|---|---|---|---|
| 1 | a message-passing layer over the agents | `model=layers/gnn`, `torch_geometric.nn.conv.GraphConv` with `aggr: add` | **met** |
| 2 | a topology | `topology: full` — every agent is a neighbour of every other, which is the topology the coupling operator `W` actually has (a distance kernel with full support) | **met** |
| 3 | no self-loops in the aggregation | `self_loops: False`, so the aggregated message is strictly over `j != i` — the same zero diagonal `W` has | **met, and it matters**: it makes the GNN's channel category-C clean, exactly like PACT's |
| 4 | the node features carry the **neighbours' actions** | BenchMARL's GNN passes **observations** as node features | **adapted** — see below |
| 5 | everything else is MAPPO | `algorithm=mappo` | **met** |

### The one thing that is not free: the neighbours' actions

BASELINES.md B3 asks for a policy that receives *the neighbours' actions*.
BenchMARL's GNN aggregates whatever is in the node features, which by default
is the observation.

The launcher therefore runs this row with `task.ns_observe_prev_action=true`,
which appends each agent's own last executed action (divided by its action
range) to its observation. The node features are then `(o_j, a_{j,t-1})` and
the graph carries the neighbours' actions, which is what the objection is
about.

This is proprioception, not privilege: the agent already measures the force it
delivered — that is what the `residual` column reports — so its own last action
is not new information to it. It is `false` everywhere else, and
`baselines/verify.py` checks that.

### Why not `topology: from_pos`

`from_pos` builds the graph from an `edge_radius`, which would make the
neighbourhood a tuning knob and, worse, a *different* neighbourhood from the
one the disturbance actually uses. `full` with `self_loops: False` is the
honest match to `W`, which is dense with a distance-decaying kernel. BenchMARL
also refuses `from_pos` in PPO critics; that restriction does not bite here
because only the actor is a GNN.

## Running it

```bash
GROUP=b3 bash scripts/run_baselines.sh
```

Requires `torch_geometric`.

## What to watch

* This is the row where "a learned graph over the agents is enough" is tested.
  If `mappo_gnn` closes most of the gap to B0 at the committed σ, the declared
  operator is not carrying its weight and the paper must say so.
* Compare it against `mfac` as well as against `mappo_blind`: the GNN gets the
  peers' actions through a **learned** aggregation, the mean-field critic gets
  them through a **fixed unweighted** one, and PACT through a **declared
  weighted** one. Those three are the ladder B3 and B4 together define.
