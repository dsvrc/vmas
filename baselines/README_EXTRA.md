# The six EXTRA baselines (X1 … X6), on VMAS

Six more published methods, implemented against the paper **and** the authors'
code where code exists, one baseline per algorithm, on the same
`simple_ns` hosts and through the same launcher as
[`README.md`](README.md)'s BASELINES.md rows.

```bash
python baselines/verify.py               # ~2 s, torch only. Run this first.
LIST=1 bash scripts/run_extra_baselines.sh    # print every launch line, run nothing

# SMOKE FIRST -- one iteration of every row, into a throwaway OUT_ROOT:
FRAMES=12000 BATCH=6000 ENVS=60 SEEDS=0 OUT_ROOT=runs/smoke_extra \
  EXTRA="experiment.off_policy_n_optimizer_steps=20 experiment.evaluation=false" \
  KEEP_GOING=1 bash scripts/run_extra_baselines.sh

DORAEMON_SUCCESS=<your B0 median return> bash scripts/run_extra_baselines.sh
```

---

## 1. The six

| # | class | baseline | paper | code | how it runs | doc |
|---|---|---|---|---|---|---|
| X1 | prior-free NS-RL, detect-and-restart | **QCD+ / RR** | [2410.13772](https://arxiv.org/abs/2410.13772) | none released (pseudocode transcribed) | `algorithm=qcd` | [qcd](docs/qcd.md) |
| X2 | mean field / fictitious play | **DEDA-FP** | [2510.22158](https://arxiv.org/abs/2510.22158) | none released | `algorithm=dedafp` | [dedafp](docs/dedafp.md) |
| X3 | performative games, independent learning | **IPGA / INPG** | [2504.20593](https://arxiv.org/abs/2504.20593) | none released | `algorithm=ipga` | [ipga](docs/ipga.md) |
| X4 | NS-RL, predictive latent | **WISDOM** | [2510.04507](https://arxiv.org/abs/2510.04507) | [MinWangcs/WISDOM](https://github.com/MinWangcs/WISDOM) | `algorithm=wisdom` | [wisdom](docs/wisdom.md) |
| X5 | adaptive domain randomisation | **DORAEMON** | [2311.01885](https://arxiv.org/abs/2311.01885) | [gabrieletiboni/doraemon](https://github.com/gabrieletiboni/doraemon) | `algorithm=doraemon` | [doraemon](docs/doraemon.md) |
| X6 | model-based, planning | **M3W** | [NeurIPS 2025](https://openreview.net/forum?id=fi24ry0BX5) | [zhaozijie2022/m3w-marl](https://github.com/zhaozijie2022/m3w-marl) | `algorithm=m3w` | [m3w](docs/m3w.md) |

Launcher classes: `x1 x2 x3 x4 x5 x6`. Rows:

```
x1  qcd_glr qcd_rr qcd_none      x4  wisdom wisdom_release
x2  dedafp dedafp_br             x5  doraemon
x3  ipga inpg                    x6  m3w m3w_noplan
```

Each class contains its own **ablation pair**, so the row is read against
something: `qcd_glr` vs `qcd_rr` is the paper's own central comparison,
`dedafp` vs `dedafp_br` is the average policy vs the last best response,
`wisdom` vs `wisdom_release` is the decoder question, `m3w` vs `m3w_noplan`
prices the search.

### Which base each one sits on

Rule: **on-policy methods go on the PPO host, off-policy methods on the SAC
host**, and the shared hyper-parameters are copied verbatim from that host's
yaml so the comparison isolates what the method adds.

| baseline | base | why |
|---|---|---|
| QCD+ / RR | MAPPO | a black-box wrapper; the base learner is whatever the reference row uses |
| DEDA-FP | IPPO | the best-response step is PPO (the paper says "SAC or PPO"); an MFG's representative agent has its own value function, so the critic is independent |
| IPGA / INPG | IPPO | *independent* learning is the paper's headline; a centralised critic would contradict it |
| WISDOM | ISAC | the reference is single-agent SAC; the per-agent lift is independent SAC |
| DORAEMON | MAPPO | it wraps whatever trains; B9's `dr_sigma` row uses MAPPO, and the pair has to match |
| M3W | its own | a SAC-shaped actor and a centralised twin Q, but the critic is trained inside the world-model loss on imagined rollouts, so it subclasses `Algorithm` directly |

---

## 2. How each one was built

The same four steps as [`README.md`](README.md) §1:

1. **Get the authors' code.** Three of the six release code; it was cloned and
   read, and every doc's "Source" table lists the files and functions by name.
   Three release none — those are transcribed from the pseudocode, and the docs
   say so.
2. **Make a checklist from the paper and the code, before writing anything.**
   That is the table in each `docs/<name>.md`: one row per idea, with the
   reference's own line next to it.
3. **Implement it row by row**, marking each `met` / `adapted` / **`NOT met`**.
4. **Check what can be checked offline.** `baselines/verify.py` grew from 89 to
   **124 checks**; 36 of them are the new arithmetic. They found three real
   bugs — see §5.

---

## 3. Everything adapted or missing, in one place

### Adapted

| baseline | adaptation | why |
|---|---|---|
| QCD+ | the reward stream is mapped onto `[0,1]` by a declared affine range and clipped | the Bernoulli GLR is defined on `[0,1]`; a VMAS reward is not. `qcd_clip_frac` reports saturation ([qcd](docs/qcd.md) F) |
| QCD+ | a restart restores the **initial** parameters and clears the optimiser moments | `H_B <- {}` is unambiguous for a bandit and has to be turned into something for a network ([qcd](docs/qcd.md) B) |
| DEDA-FP | the population density enters the **policy's input**, not the reward | a VMAS task's reward is shared by every arm in this repo; rewriting it would make the row incomparable with its own reference ([dedafp](docs/dedafp.md) C) |
| DEDA-FP | affine coupling flow in place of an autoregressive neural spline flow | same object (exactly invertible, tractable log-determinant, MLE, conditioned on `t`), simpler transform; the log-determinant is checked against autograd ([dedafp](docs/dedafp.md) B) |
| DEDA-FP | `M_SL` is a bounded reservoir; `Gbar` is fitted to its states | the reference never forgets, and rolling out `pibar` would need a second collection pass ([dedafp](docs/dedafp.md) A, E) |
| IPGA | the proximal term is the closed-form `L2` distance between the two Gaussian **densities** | the paper's `\|\|pi-pi'\|\|_2` is over probability vectors; the continuous counterpart is the `L2(da)` norm, which has a closed form ([ipga](docs/ipga.md) B) |
| INPG | a Fisher-preconditioned step by conjugate gradients | the multiplicative update **is** NPG under the softmax parameterisation; the closed form is checked against it ([ipga](docs/ipga.md) D) |
| WISDOM | `y(z)` is used at collection **and** training, and `z` is recomputed rather than read stale from the buffer | the released code is inconsistent with itself here ([wisdom](docs/wisdom.md) E) |
| WISDOM | the context window is built from observations carrying `a_{t-1}` and `r_{t-1}` | same information as the reference's explicit `(o,a,r,o')` rows, via machinery RMA and LIAM already use ([wisdom](docs/wisdom.md) B) |
| DORAEMON | when the current distribution is infeasible the round is **skipped** | the reference solves an inverted problem to find a feasible restart point ([doraemon](docs/doraemon.md) B) |
| DORAEMON | one randomised dimension (`sigma`) | that is what B9's row randomises, and the pair has to be comparable |
| M3W | `n_step = 5` instead of 20 | the window is stored per transition; 20 makes the buffer 24 frames deep ([m3w](docs/m3w.md) C) |
| M3W | trajectory chunks by frame stacking instead of a sequence sampler | puts the sequence inside the transition; a window never spans two episodes, so no termination mask is needed ([m3w](docs/m3w.md) H) |
| M3W | the plan is always warm-started | the acting module is not told where an episode began ([m3w](docs/m3w.md) G) |

### Not implemented

| baseline | what is missing | why |
|---|---|---|
| QCD+ | **MASTER itself** (Algorithm 1) | its recursive scheduling is substantial and Theorem 4 says the result is already known: at any horizon here it never fires and reduces to random restarting — which `detector=random` runs. The theorem is checked arithmetically and printed at construction ([qcd](docs/qcd.md) D) |
| DEDA-FP | the population literally playing `pibar` (the finite-`N` correction) | would cost `(N-1)/N` of every batch and make the reported return a mixture. What runs is the **mean-field limit** of Algorithm 3 ([dedafp](docs/dedafp.md) F) |
| DEDA-FP | exploitability as the reported metric | needs a full best-response training run per measurement ([dedafp](docs/dedafp.md) G) |
| IPGA | the log-barrier-regularised INPG | there are no action probabilities to keep away from zero in a continuous Gaussian policy ([ipga](docs/ipga.md) E) |
| IPGA | the occupancy-measure special case | tabular: `\|S\| x \|A\|` variables per agent and a projection onto the occupancy polytope ([ipga](docs/ipga.md) G) |
| WISDOM | `use_parametrized_alpha` | off in the reference's own default config |
| DORAEMON | `get_feasible_starting_distr`, `robust_estimate`, `prior_constraint`, `performance_lb_percentile`, `test_on_target_distr` | the first is (B) above; the next three are **off in every reported run of the reference**; the last is best-model selection, which this row replaces with "evaluate the checkpoint afterwards", exactly as `dr_sigma` already does |
| M3W | the Gaussian-CDF `load` estimate in the noisy router | affects the auxiliary loss only, at `balance_coef = 5e-4`; it is the form the reference itself falls back to in eval mode ([m3w](docs/m3w.md) B) |
| M3W | multi-task training | there is one task here, whose dynamics vary in time ([m3w](docs/m3w.md) I) |
| WISDOM, DEDA-FP, M3W | discrete actions | each would need a different head, not a flag |

---

## 4. New environment knobs — all OFF by default

Three new observation channels and three new DR keys, added the same way B8's
`ns_observe_prev_action` was:

| key | for | what |
|---|---|---|
| `ns_observe_prev_reward` | X4, X6 | the reward of the transition that produced this observation, so a **window of observations is a window of transitions** |
| `ns_observe_time`, `ns_time_horizon` | X2 | `t / horizon`; a finite-horizon mean field equilibrium is a function of `t` |
| `ns_dr_dist`, `ns_dr_a`, `ns_dr_b` | X5 | a Beta DR distribution whose shape DORAEMON replaces between rounds, through [`simple_ns/dr_state.py`](../simple_ns/dr_state.py) |

`verify.py` checks that every one of them is present in all four task yamls,
declared in all four `TaskConfig` dataclasses, and **left inert**, so with the
defaults `simple_ns` is bit-for-bit the environment every existing arm was run
on.

---

## 5. Three real bugs the offline checks caught

1. **DORAEMON was inert.** `KL(Beta(100,100) || Beta(99.994,99.994))` is
   `9e-10`; in float32 the same expression is **`-4.2e-05`** — negative, for a
   divergence. Measured before the fix: the solver reported success after 341
   iterations having moved the distribution by `1.9e-4` against a bound of
   `5e-2`, and the entropy never changed. Fixed by computing every Beta
   quantity in float64 **and** supplying analytic jacobians for both
   constraints, which is what the reference does.
2. **The GLR could fire on its first sample.** `bernoulli_kl`'s boundary guard
   used `1e-12`, and `1 - 1e-12` rounds to exactly `1.0` in float32 — so an
   all-zero segment divided by zero and returned `inf`.
3. **Phantom scenario keys.** A quoted word inside a comment in `NS_KWARGS`
   became a fake environment key, because `check_plumbing.py` reads the kwarg
   names by scraping quoted strings out of that block.

---

## 6. Status

**Not run.** Every file parses on this interpreter, `verify.py`'s 124 checks
pass, `check_plumbing.py` passes, the launcher prints a complete plan for all
twelve rows, and M3W's world model, planner and loss were exercised end to end
against a stubbed torchrl (shapes, gradients, the polyak update, and the
window-to-transition unpacking). **None of that is a training run.** torchrl is
not installable on the machine this was written on, so the torchrl-facing
wiring — key names in the collected batch, the functional parameter split, the
loss/optimiser mapping — has not been executed.

**Smoke-test before spending a queue slot**, with the command at the top of
this file. What to check in a smoke run, per row: the construction banner
resolves the slices and budgets you intended, the dial fires
(`ns_live_frac = 1.0`), and the method's own diagnostic moves — `qcd_restarts`,
`dedafp_sl_nll`, `ipga_dist_move`, `wisdom_td`, `doraemon_entropy`,
`m3w_dynamics`. Each doc's "what to watch" table says what a dead row looks
like.

Two numbers are **host-dependent and must be set** before the rows mean
anything: `DORAEMON_SUCCESS` (the return at which an episode counts as solved —
take your B0 median) and `algorithm.reward_low/high` for QCD's detector. Both
are called out by the launcher and by their docs.
