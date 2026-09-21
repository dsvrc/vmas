# DORAEMON — X5, domain randomisation via entropy maximisation

## Source

| what | where |
|---|---|
| paper | Tiboni, Klink, Peters, Tommasi, D'Eramo, Chalvatzaki, *Domain Randomization via Entropy Maximization*, ICLR 2024 ([arXiv 2311.01885](https://arxiv.org/abs/2311.01885)) |
| code read | [gabrieletiboni/doraemon](https://github.com/gabrieletiboni/doraemon): `doraemon/doraemon/doraemon.py` in full (`DomainRandDistribution`, `DORAEMON.step`, `objective_fn`, `kl_constraint_fn(_prime)`, `performance_constraint_fn(_prime)`, `variance_IS_estimator`, `get_feasible_starting_distr`), `train_doraemon.py`, `exps_launcher_configs/scripts/train_doraemon/*.yaml`, `reproduce_paper_results.md` |
| implementation | [`benchmarl/algorithms/doraemon.py`](../../benchmarl/algorithms/doraemon.py), [`benchmarl/conf/algorithm/doraemon.yaml`](../../benchmarl/conf/algorithm/doraemon.yaml), [`simple_ns/dr_state.py`](../../simple_ns/dr_state.py), the `X5` block of [`_baseline_math.py`](../../benchmarl/algorithms/_baseline_math.py) |

## Why this row exists

BASELINES.md B9's must-run row is **fixed** domain randomisation: draw
`sigma ~ U[0, 3]` per episode and train through it. Its weakness is the one
every DR paper opens with — too little variability and the policy does not
generalise, too much and it goes conservative — and the range is a guess nobody
can justify. DORAEMON removes the guess. Between training rounds it solves

```
min_phi  KL( phi || phi_target )
s.t.     E_phi[ 1(return >= tau) ] >= alpha        (performance)
         KL( phi_i || phi ) <= epsilon             (trust region)
```

with `phi` a Beta on the declared severity range and `phi_target` the
maximum-entropy distribution on it (the uniform). **Minimising the KL to the
uniform is maximising entropy**, subject to the policy still solving the task
often enough — so the distribution starts narrow and widens exactly as fast as
the policy can take it. `doraemon` vs `dr_sigma` prices the **curriculum**,
with everything else held fixed.

The performance constraint is evaluated by **importance sampling** over the
episodes just collected, so testing a candidate distribution costs no
environment steps. That is what makes the outer loop affordable, and it is why
the environment has to report the severity each episode actually ran at
(`info["ns_sigma"]`, which it has published since B9).

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | the DR distribution is a **Beta** rescaled onto `[m, M]` | `DomainRandDistribution` with `dr_type='beta'`; `y = x(M-m) + m` | `simple_ns/dr_state.BetaDr` + `ExertionMixin._draw_sigma`'s beta branch | **met** |
| 2 | the objective is `KL(phi_new || phi_target)` | `objective_fn` | `_kl_target`, minimised by trust-constr | **met** |
| 3 | `phi_target` is the maximum-entropy distribution on the support | `target_distr` | `Beta(1,1)` = the uniform | **met, tested** (`H[Beta(1,1)] on [0,3] = log 3`) |
| 4 | trust region `KL(phi_i || phi_new) <= epsilon` | `kl_constraint_fn` | `_kl_step` | **met** |
| 5 | performance constraint `E_phi[success] >= alpha`, by **importance sampling** | `performance_constraint_fn`: `exp(proposed.pdf - current.pdf) * (values >= cond)` | `_performance`, `importance_ratio` | **met, tested** against direct sampling |
| 6 | the success indicator is a **constant** in the optimisation | `perf_values = torch.tensor(values.detach() >= cond, ...)` | `solved` is computed once, outside the objective | **met** |
| 7 | analytic jacobians for the objective **and both constraints** | `objective_fn` returns `(value, grad)`; `kl_constraint_fn_prime`, `performance_constraint_fn_prime` | `_with_grad` wraps each in `torch.autograd.grad` | **met** — see (A), this is load-bearing |
| 8 | solved with scipy `trust-constr`, `jac=True`, `gtol=1e-4`, `xtol=1e-6` | `minimize(..., method="trust-constr", ...)` | the same solver; `gtol=1e-8, xtol=1e-10, maxiter=1000` | **met**, tighter tolerances — see (A) |
| 9 | the parameters are optimised through a **sigmoid** inside `[min_bound, max_bound]` | `DomainRandDistribution.sigmoid` / `inv_sigmoid` | `sigmoid_bounds` / `inv_sigmoid_bounds` | **met, tested** (round-trips) |
| 10 | `min_bound = 0.8`, `max_bound = init_beta_param + 10` | `DORAEMON.__init__` | `min_bound: 0.8`, `max_bound: 110.0` with `init = 100` | **met** |
| 11 | `init_beta_param = 100` — start narrow | `DORAEMON.__init__` | `init_a: 100`, `init_b: 100` | **met** |
| 12 | keep the old parameters unless the result is **both feasible and better** | `if not (all(constraints_satisfied) and result.fun < old_f): new_x_opt = x0_opt` | the same test, and the row prints which branch it took | **met** |
| 13 | `train_until_performance_lb`: do not move until the constraint holds once | `if self.train_until_lb and not self.train_until_done: ... return` | `train_until_lb: True` | **met** |
| 14 | `hard_performance_constraint` | the reference's launcher sets it `true` | `hard_constraint: True`; if the CURRENT distribution already violates the constraint the round is **skipped** | **adapted** — see (B) |
| 15 | the inverted problem that finds a feasible restart point | `get_feasible_starting_distr` | **not implemented** | **NOT met** — see (B) |
| 16 | `success_rate_condition = 0.5` | the `succRate50` config, used for every reported DORAEMON run | `success_rate: 0.5` | **met** |
| 17 | `kl_ub` swept over `{0.1, 0.05, 0.01, 0.005, 0.001}` | `reproduce_paper_results.md` | `kl_upper_bound: 0.05`, and the yaml says to sweep it | **met** |
| 18 | the budget is split into `n_iters` training rounds | `max_ts_per_iter = timesteps / n_iters` | `n_iters: 20`; ~150 k frames per round, the same order as the reference's 100 k | **met** |
| 19 | the episode buffer is **reset each round** | `self.training_subrtn.reset_buffer()` | `DoraemonState.reset_buffer` | **met** |
| 20 | `(dynamics, return)` pairs per episode | `get_buffer()` from a per-env wrapper | read out of the collected batch: reward stream + done flags + `info["ns_sigma"]` | **met**, differently plumbed |
| 21 | `max_dynamics_samples` | `DORAEMON.__init__` (marked deprecated) | `max_episodes: 1000` | **met** |
| 22 | `robust_estimate` / `alpha_ci`: use the lower confidence bound | `variance_IS_estimator` + `_get_ci` | **not implemented**; the reference's own default is off and every reported run leaves it off | **NOT met**, off in the reference |
| 23 | `prior_constraint`: keep the density at a prior point above uniform | `prior_constraint_fn` | **not implemented**; `prior_constraint: false` in the reference's own default config | **NOT met**, off in the reference |
| 24 | `performance_lb_percentile` | an alternative constraint form | **not implemented**; unused in the reported runs | **NOT met**, off in the reference |
| 25 | `test_on_target_distr`: evaluate on the max-entropy distribution each round, for best-model selection | `test_on_target_distr` | **not implemented** | **NOT met** — see (C) |
| 26 | multi-dimensional DR (one Beta per dynamics parameter) | `DomainRandDistribution` is `ndims`-wide | **one dimension**: `sigma` | **adapted, structural** — see (D) |
| 27 | `stopAtRewardThreshold`, `reset_agent`, `bootstrap_values` | `TrainingSubRtn.train` | **not implemented**; all off in the reference's default config | **NOT met**, off in the reference |

### (A) The float32 bug, and why the jacobians are not optional

This is the one place where a plausible implementation is silently inert, so it
is worth stating.

`KL(Beta(100,100) || Beta(99.994, 99.994))` is `9.0e-10`. Computed in float32
— which is what `torch.tensor(float(a))` gives you by default — the same
expression evaluates to **`-4.2e-05`**: negative, for a divergence, because it
is a difference of log-Gammas of order 400. A trust-region constraint function
with `1e-5` of noise makes finite differences meaningless.

Measured, before this was fixed: the solver reported `success=True` after 341
iterations having moved the distribution by a KL of `1.9e-4` against a bound of
`5e-2`. The entropy did not change. **DORAEMON ran and did nothing.**

Two things fix it and both are in the reference:

1. every Beta quantity is computed in **float64** (`_f64` in
   `_baseline_math.py`, and `verify.py` checks the KL is positive and `~9e-10`);
2. **analytic jacobians** are supplied for the objective and for *both*
   constraints, via autograd — which is exactly what `kl_constraint_fn_prime`
   and `performance_constraint_fn_prime` do in the reference, and which the
   first draft here omitted.

With both, a 12-round simulation walks `Beta(100,100)` to the uniform on
`[0, 3]` with the trust region binding at `0.05000` every round and the entropy
rising monotonically from `-0.827` to `log 3 = 1.0986`. That is the published
behaviour.

### (B) The hard constraint, without the inverted problem

When the current distribution already violates the performance constraint, the
reference solves a second problem — maximise performance subject to the trust
region — to find a feasible starting point, and only gives up if that fails.
Here the round is **skipped** instead and the reason is printed. The policy
gets another round of training at the current distribution, which is the same
outcome the reference reaches whenever its inverted problem also fails. What is
lost is the case where a *different* distribution inside the trust region would
have been feasible; on a one-dimensional support that set is small.

### (C) No evaluation on the target distribution

The reference tests the policy on the max-entropy distribution every round and
keeps the best. BenchMARL evaluates on the training environment, so there is no
second evaluation environment to point at a different `sigma` mid-run. This is
the same situation as B9's `dr_sigma`, and the same answer: **evaluate the
checkpoint afterwards**, exactly as [`dr_sigma.md`](dr_sigma.md) prescribes.
What is lost is best-model selection — this row reports its final policy, not
its best one.

### (D) One dimension

The reference randomises a whole dynamics vector (masses, frictions, damping).
Here the randomised quantity is `sigma`, the single declared severity, because
that is what B9's row randomises and the pair has to be comparable. The
optimisation is therefore over two scalars `(a, b)` rather than `2 * ndims`.
Nothing in the method assumes more than one dimension.

### (E) How the algorithm reaches the environment

The distribution has to be readable by the environment at every episode reset
and writable by the algorithm between rounds.
[`simple_ns/dr_state.py`](../../simple_ns/dr_state.py) is a process-wide
registry: `Doraemon._publish` writes, `ExertionMixin._beta_params` reads. This
works because a VMAS environment is a vectorised torch object **in this
process** — there is no worker process and no pickling between the algorithm
and the scenario, which is the same reason the PACT debug CSV can be written
from inside the scenario.

It is inert unless `ns_dr_dist=beta`: with the default `uniform`, `current()`
is never called and `_draw_sigma` takes exactly the branch it took before this
file existed. The environment **raises** if the published support disagrees
with the task's, rather than silently drawing from another range.

## Running it

```bash
DORAEMON_SUCCESS=<your B0 median return> GROUP=x5 \
  bash scripts/run_extra_baselines.sh
```

`success_return` is the only host-dependent number in the method and the
reference has no default for it (it sets 1600 for Hopper, and so on). **Take
the median return of your `sigma = 0` reference arm.** The launcher prints a
warning if it is left at 0.

Evaluate the checkpoint at the committed `sigma` afterwards, as for `dr_sigma`.

## What to watch

| column | what it means |
|---|---|
| `doraemon_entropy` | the distribution's entropy. **This is the method.** It should rise monotonically toward `log(high - low)`. Flat = the curriculum never started. |
| `doraemon_success` | the measured success rate. **Pinned at 0** → `success_return` is unreachable, every round prints SKIPPED, and the row is fixed narrow DR. **Pinned at 1** → the threshold is trivial and DORAEMON walks to the uniform in a few rounds, i.e. becomes `dr_sigma`. |
| `doraemon_kl_step` | the KL actually taken. Should sit at `kl_upper_bound` while the constraint is slack — if it is orders of magnitude below, the solver is not moving (see (A)). |
| `doraemon_a` / `doraemon_b` | the live Beta parameters. |
| `doraemon_solver_ok` | 0 means the solve failed and the parameters were kept. |
| `doraemon_skipped` | how many rounds were skipped. A large count with a flat entropy is the failure mode in (B). |
| `doraemon_median_return` | printed each round so `success_return` can be set from data. |
