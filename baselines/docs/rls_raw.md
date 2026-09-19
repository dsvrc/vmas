# Unstructured RLS on the raw peer actions — BASELINES.md B10

## Source

This one has no external paper: it is the ablation BASELINES.md B10 specifies,
against PACT's own estimator.

> **Per-agent unstructured RLS on raw peer actions** (regress the residual on
> the N−1 broadcast actions directly, no classes): isolates the value of the
> *declared basis* (r parameters vs N−1).
>
> Prediction: fails at N = 22/38 (under-excited, ill-conditioned). ≈ 30 lines.

It also answers BASELINES.md §C's row *"Why not learn the coupling instead of
declaring it"*.

| what | where |
|---|---|
| implementation | [`simple_ns/baselines.py`](../../simple_ns/baselines.py) (`RlsRawMixin`) |
| the estimator it reuses | [`pact1/core.py`](../../pact1/core.py) (`RLS`, `confidence`) — **the same object**, unchanged |

## The checklist

| # | what B10 asks for | here | status |
|---|---|---|---|
| 1 | regress the residual on the **N−1 broadcast actions directly** | `psi_i = [1, x_i1, ..., x_iN]` over `j != i` | **met** |
| 2 | **no classes** | the design matrix has one column per peer, not one per declared class | **met** |
| 3 | per-agent | one estimator row per agent per parallel world, as PACT has | **met** |
| 4 | the same RLS | `pact1.core.RLS`, the same `mu`, `p0`, `p_trace_max`, the same dead-row skip, the same covariance bound | **met — literally the same class** |
| 5 | the same everything else | `pact_trust`, `pact_warmup`, `pact_corr_clip`, `pact_y_clip`, the same `confidence()` gate, the same droop feed-forward | **met** |
| 6 | `r` parameters against `N−1` | `dim = N` (intercept + N−1 peers) against PACT's `1 + n_types` | **met, and printed at construction** |

## What the regressor is

```
x_ij(t) = rho * x_ij(t-1) + (1 - rho) * c_ij * ||u_j(t-1)|| / u_range_i
psi_i   = [ 1, x_i1, ..., x_i(i-1), x_i(i+1), ..., x_iN ]
```

* `rho` is the dial's own transmission memory, so the arm sees the same
  smoothing PACT's channels do;
* `||u_j||` is the peer's draw, which on a shared supply is a scalar — the same
  quantity `Coupling.step_draw` aggregates;
* the division by `u_range_i` is P-3.3's scaling in the only form available
  without classes: make the regressor dimensionless against the receiver's own
  action range. There is no centring — the intercept column carries the mean;
* `c_ij = 1` by default, which is B10's literal wording ("directly"). With
  `rls_raw_use_operator: true`, `c_ij = W_ij`.

## Two versions, and why both are run

| row | `c_ij` | what it isolates |
|---|---|---|
| `rls_raw` | `1` | BASELINES.md's literal baseline: no operator, no classes |
| `rls_raw_w` | `W_ij` | the same raw per-peer regression, **handed the declared operator** — which is public information (built from declared plumbing and observable geometry) |

The second exists so the claim does not rest on withholding something public.
If `rls_raw_w` also fails, the failure is about the **number of parameters**,
which is what B10 predicts. If it succeeds, the operator was doing the work and
the class reduction is only a compute saving — and the paper has to say that.

## The zero diagonal is structural

`_peer_idx` is built once as "the peers of `i`, in increasing order", so agent
`i`'s own draw can never enter its own regressor. That is the same `j != i`
property the coupling, PACT's basis and the mean-field arm all have, and it is
why a lone agent reads exactly zero on every channel here too.

## Restricted to the droop channel

On the droop channel a peer's draw is a scalar and "the raw peer action" is
unambiguous. On the shove channel it is a vector, and regressing the residual on
it would need a declared projection — which is precisely the structure this arm
exists to do without. `RlsRawMixin._build_estimator` raises rather than invent
one.

## Running it

```bash
GROUP=b10 ONLY="rls_raw rls_raw_w" bash scripts/run_baselines.sh
```

`pact_enabled=true` together with `ns_baseline=rls_raw` is an error, not a
double feed-forward.

## What to watch

The arm writes PACT's diagnostic columns, so read them side by side:

| column | what it tells you |
|---|---|
| `pact_n_bounded` | how often the covariance bound had to act. **This is the row's headline diagnostic**: B10 predicts the raw design matrix is under-excited, and a rising `n_bounded` is exactly what an under-excited direction looks like before it diverges. |
| `pact_n_diverged` | non-finite predictions. Should be 0; if it is not, the raw regression blew up and the arm is reporting the guard rail rather than a method. |
| `pact_confidence` | `psi' P psi`. Compare against PACT's at the same σ: the whole claim is that `N-1` columns make this worse than `1 + r`. |
| `pact_corr_vs_d` | how much of the disturbance was actually cancelled. |

**At N = 4 (balance) this arm has 4 parameters against PACT's 4** — the two are
the same size, so `balance` is the host where the ablation is *least*
informative. B10's prediction is about N = 22/38. Run it on a host with more
agents, or raise `n_agents`, before drawing the conclusion B10 states; on
balance it is a control that should come out roughly even, and if it does not,
something other than the parameter count is responsible.
