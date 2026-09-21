# IPGA / INPG — X3, independent learning in performative Markov potential games

## Source

| what | where |
|---|---|
| paper | Sahitaj, Sasnauskas, Yalın, Mandal, Radanović, *Independent Learning in Performative Markov Potential Games*, [arXiv 2504.20593](https://arxiv.org/abs/2504.20593) |
| read | the performative Markov game definition `G(pibar)`, Assumption 1 (sensitivity), the definition of a performatively stable equilibrium, the IPGA update, the INPG update, and the repeated-retraining special case |
| implementation | [`benchmarl/algorithms/ipga.py`](../../benchmarl/algorithms/ipga.py), [`benchmarl/conf/algorithm/ipga.yaml`](../../benchmarl/conf/algorithm/ipga.yaml), the `X3` block of [`_baseline_math.py`](../../benchmarl/algorithms/_baseline_math.py) |

No code is released. The two algorithms are one update rule each.

## Why this is the one baseline whose assumptions this instance satisfies

A **performative** Markov game is one whose reward and transition kernels
depend on the *deployed* joint policy:

```
G(pibar) = (N, S, {A_i}, {r_{i,pibar}}, P_pibar, gamma, rho)
```

`simple_ns` is exactly that, and not by analogy. The disturbance an agent feels
is a weighted mean of its **neighbours' exertions**, so the transition kernel
each agent faces is a function of what the other agents' policies do: change
the deployed policies and you change the environment. That is the paper's
setting verbatim.

The equilibrium concept is the **performatively stable equilibrium** (PSE):

```
V_{i,pi}^{pi_i, pi_-i}(rho) >= V_{i,pi}^{pi'_i, pi_-i}(rho) - eps
```

— note the subscript `pi` on `V`, which is what separates a PSE from a Nash
equilibrium of a fixed game. The paper proves existence under a sensitivity
assumption (rewards and transitions Lipschitz in policy distance, with
constants `omega_r` and `omega_p`) and gives two independent algorithms that
reach it.

## The two updates

```
IPGA   pi^{t+1}_i(.|s) = argmax_{pi_i}  <pi_i, Qbar^t_i(s,.)>
                                        - 1/(2 eta) ||pi_i - pi^t_i(.|s)||_2^2
                          best-iterate convergence to an approximate PSE

INPG   pi^{t+1}_i(a|s) prop-to pi^t_i(a|s) exp( eta/(1-gamma) Abar^t_i(s,a) )
                          ASYMPTOTIC LAST-ITERATE convergence
```

The second is the stronger result and the reason both are run.

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | **independent** learners: no centralised critic, no communication | the paper's title and setting | IPPO host (`centralised=False`); the launcher sets `share_policy_params=false` and the algorithm warns if it is not | **met** |
| 2 | `Abar^t_i` is the agent's own **marginalised** advantage | `Qbar^t_{i,t}(s, a_i)` / `Abar^t_{i,t}(s, a_i)` | with an independent critic and per-agent GAE, the advantage each agent sees IS its marginal | **met by construction** |
| 3 | IPGA's objective is `<pi, Qbar> - 1/(2 eta) ||pi - pi^t||^2` | the update rule | `-E[(pi/pi^t) Abar] + prox/(2 eta)`, minimised | **met** — see (A) |
| 4 | the proximal term is the **squared Euclidean distance between the two action distributions** | `||pi_i(.|s) - pi^t_i(.|s)||_2^2` | discrete: `categorical_l2_sq`. continuous: `gaussian_l2_sq`, the closed-form `L2(da)` distance between the two densities | **met** — see (B), **tested** against numerical quadrature |
| 5 | the reference policy is the **deployed** one, `pi^t` | the `t` superscript | `Ipga.process_batch` freezes `pi^t`'s distribution parameters into the batch before a single update runs, so every minibatch of the iteration is anchored to the policy that produced the data | **met** — see (C) |
| 6 | the surrogate is **unclipped** | there is no clipping anywhere in the paper | `IpgaLoss.forward` computes the surrogate directly; `clip_epsilon` is inherited and unused | **met** |
| 7 | INPG is `pi exp(eta/(1-gamma) Abar)`, i.e. natural policy gradient | the update rule; the paper notes this is NPG for the softmax parameterisation | a Fisher-preconditioned step solved by conjugate gradients, with multiplier `eta/(1-gamma)` | **met** — see (D), **tested**: the closed form is checked in `verify.py` |
| 8 | INPG's step is **fixed size**, with no trust region and no line search | the update rule | exactly that; `max_kl: 0.0` (off) | **met** |
| 9 | the regularised INPG (log-barrier on small action probabilities) | "adds log-barrier regularization to overcome issues with small action probabilities" | **not implemented** | **NOT met** — see (E) |
| 10 | repeated retraining: optimise a surrogate against the deployed environment, then redeploy | the third procedure | **BenchMARL's loop already is this**: collect under `pi^t`, run `on_policy_n_minibatch_iters` inner steps against that data, redeploy | **met by the loop** — see (F) |
| 11 | the occupancy-measure special case (agent-independent transitions) | `argmax_{mu_i in D_i} {<grad Phi, mu_i> - lambda/2 ||mu_i||^2}` | **not implemented** | **NOT met** — see (G) |
| 12 | Assumption 1's sensitivity constants `omega_r`, `omega_p` | Assumption 1 | not estimated; what is logged is `ipga_dist_move`, the quantity Assumption 1 multiplies by `omega` to bound how far the environment moves | **adapted, diagnostic** — see (H) |
| 13 | `eta` is set from the game's smoothness constants | the convergence theorems | a declared sweep point; the yaml says so, and the algorithm prints the effective coefficient | **adapted** — see (I) |
| 14 | the potential function `Phi_pibar` | the MPG definition | not constructed: this instance is not known to be a potential game, and the algorithms do not need `Phi` to run — only the *theory* does | **out of scope, stated** — see (J) |
| 15 | the experiments (safe-distancing game, stochastic congestion game) | §6 | not reproduced; those are the paper's evidence for its own claims | **out of scope, deliberate** |

### (A) The advantage in place of the marginalised action value

`<pi, Qbar>` and `<pi, Abar>` differ by `sum_a pi(a|s) V(s) = V(s)`, which does
not depend on `pi`. The argmax is therefore unchanged and the estimator has far
lower variance. This is the standard baseline-subtraction identity, not an
approximation.

### (B) `||pi - pi'||_2` for a continuous action set

For a finite action set the paper's distance is the Euclidean norm of the
difference of the probability *vectors*. Its continuous counterpart is the
`L2(da)` norm of the difference of the two *densities* — the same functional
with the sum over actions replaced by the integral — and for diagonal Gaussians
it has a closed form, because a product of two Gaussian densities integrates to
a Gaussian density at the difference of the means:

```
int p^2 = N(0; 0, 2 Sigma_p)
int q^2 = N(0; 0, 2 Sigma_q)
int pq  = N(mu_p - mu_q; 0, Sigma_p + Sigma_q)
||p-q||^2 = int p^2 + int q^2 - 2 int pq
```

No sampling and no estimator: the quantity itself. `verify.py` checks it
against numerical quadrature to 1e-5 relative, and checks it is zero between
identical policies.

`use_tanh_normal: true` means the acting distribution is a `TanhNormal`. The
distance is computed between the *base* Normals; the tanh-and-affine map onto
the action box is a fixed bijection shared by both policies, so it identifies
them completely. (Unlike the KL, the `L2` distance of the densities is **not**
invariant under that map, so this is a choice: the proximal term measures
movement in the pre-squash parameters. Stated.)

### (C) "Deployed" is not "previous minibatch"

This is the whole content of *performative*. The proximal term is anchored at
the policy whose induced environment produced the data. Anchoring it at the
previous minibatch's policy instead would make it a smoothing term and the
algorithm something else. `process_batch` runs once per iteration, before the
batch reaches the buffer, so the frozen parameters ride along into every
minibatch of that iteration.

### (D) Natural policy gradient for continuous actions

For the softmax parameterisation, `pi exp(eta/(1-gamma) A)` **is** the natural
policy gradient step (Kakade; Agarwal et al.). For a Gaussian policy the
multiplicative form has no closed analogue, so what is taken is the natural
gradient itself: `theta <- theta + eta/(1-gamma) F^-1 g`, with `F` the Fisher
(the Hessian of the KL) and the solve done by conjugate gradients — the same
machinery LCPO uses, minus the trust region and the line search, because INPG
has neither.

`verify.py` checks the closed form this generalises: the multiplicative update
matches `softmax(log pi + eta/(1-gamma) A)` exactly.

`max_kl` is **off by default and is not part of the method**. A positive value
rescales the step so its quadratic form under the Fisher is at most `max_kl` —
a declared safety bound for a run where the unconstrained natural gradient
diverges. If you turn it on, the row is INPG-with-a-trust-region and the paper
should say so.

### (E) The log-barrier regularisation

The regularised INPG adds a log-barrier to keep action probabilities away from
zero. For a continuous Gaussian policy there are no action probabilities to
keep away from zero, and the corresponding construction (a barrier on the
scale) is not in the paper. Left out rather than invented.

### (F) Repeated retraining

The paper's third procedure is: agents independently optimise a surrogate
objective against the environment induced by the currently deployed policy,
then redeploy. **BenchMARL's training loop is already that shape** — one
deployment per iteration, `experiment.on_policy_n_minibatch_iters` inner
optimisation steps per deployment. Nothing had to be added; what the knob
means is documented here so a reviewer can see the correspondence. For INPG the
inner count is 1 by construction (one natural-gradient step per rollout).

### (G) The occupancy-measure special case

The paper's finite-time last-iterate result is for a special case with
agent-independent transitions, where agents optimise over **occupancy
measures** `mu_i in D_i` directly and recover policies as occupancy ratios.
That is a tabular object: it needs `|S| x |A|` variables per agent and a
projection onto the occupancy polytope. There is no continuous-control
analogue, and inventing one would not be the published algorithm.

### (H) What replaces the sensitivity constants

`omega_r` and `omega_p` bound how far the *environment* moves when the deployed
policy moves by `||pi - pi'||`. They are properties of the game and are not
measurable from inside a training run without a second, counterfactual rollout.

What is logged instead is `ipga_dist_move` — the `L2` distance the action
distribution actually moved in one deployment — which is the quantity those
constants multiply. **If it does not settle, there is no last iterate to
converge to**, and the INPG row's headline claim is not being exercised.

### (I) `eta`

The theory fixes `eta` only through smoothness constants that are not
measurable here, and it enters **differently in the two variants**: as the
proximal weight `1/(2 eta)` in IPGA and as the multiplier `eta/(1-gamma)` on
the natural gradient in INPG. It does **not** transfer between the rows. Sweep
each separately; the algorithm prints the effective coefficient at
construction.

### (J) Potential game

The convergence results assume the game is a Markov *potential* game. Whether
`simple_ns` is one is not established here, and it is not needed to *run* the
algorithms — only to predict that they converge. A reviewer should read this
row as "the published independent-learning updates for performative games, run
on a performative game", not as a verification of the theorems.

## Running it

```bash
GROUP=x3 bash scripts/run_extra_baselines.sh      # ipga, inpg
```

Both launch with `experiment.share_policy_params=false`.

## What to watch

| column | what it means |
|---|---|
| `ipga_dist_move` | how far the action distribution moved in one deployment. This is the last-iterate quantity — see (H). |
| `ipga_prox` | the proximal penalty (IPGA only). If it is ~0 the step size is too small to be doing anything; if it dominates the surrogate the policy is frozen. |
| `ipga_surrogate` | `E[(pi/pi^t) Abar]`. Should be positive and shrinking toward 0 as the policy stops improving. |
| `ipga_step_norm` / `ipga_param_move` | INPG's step size in parameter space. |
| `ipga_fisher_ok` | 0 means the conjugate-gradient solve returned a non-finite direction — raise `damping`. |
