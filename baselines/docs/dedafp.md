# DEDA-FP — X2, deep fictitious play for continuous mean field games

## Source

| what | where |
|---|---|
| paper | Magnino, Shao, Wu, Shen, Laurière, *Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics*, [arXiv 2510.22158](https://arxiv.org/abs/2510.22158) (NeurIPS 2025; OpenReview `Jw5TFF3HkH`) |
| read | Algorithm 3 (the outer loop), the supervised-averaging loss `L_NLL`, the conditional-normalising-flow MLE objective, and the exploitability bound |
| implementation | [`benchmarl/algorithms/dedafp.py`](../../benchmarl/algorithms/dedafp.py), [`benchmarl/conf/algorithm/dedafp.yaml`](../../benchmarl/conf/algorithm/dedafp.yaml), the `X2` block of [`_baseline_math.py`](../../benchmarl/algorithms/_baseline_math.py) |

No code is released. The method is Algorithm 3 plus two standard training
objectives, and all three are written out in the paper.

## The method, in the paper's three lines

```
for k = 1..K:
    1. BEST RESPONSE   pi*_k  = argmax_pi J(pi, Gbar_{k-1})     [deep RL]
    2. AVERAGE         pibar_k = argmin_theta L_NLL over M_SL   [supervised]
    3. DISTRIBUTION    Gbar_k  = argmax_phi log q_phi(x | t)    [cond. flow]
return pibar_K, Gbar_K
```

Three different kinds of learning, and that is the point. The best response is
RL. The **average policy is a maximum-likelihood fit to the actions every past
best response took** — which is what makes it fictitious play rather than an
average of weights. And the population distribution is a **conditional**
normalising flow over states *given the time index*, because a non-stationary
mean field game's equilibrium measure is a function of `t`.

## Why this baseline belongs in the table

Every other non-stationarity row treats the disturbance as exogenous:
something to observe (LCPO), identify (RMA), average over (DR) or detect
(QCD+). A mean field game says the thing that varies **is the population**, and
the right object to learn is the population's distribution. On this instance
the disturbance an agent feels is literally a functional of what the other
agents are doing, so "the mean field *is* the non-stationarity" is a live
hypothesis, and this row is what tests it.

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | an outer **fictitious-play loop** of `K` iterations | Alg. 3 | `fp_iters`; the frame budget is split evenly, and the boundary is detected in `process_batch` | **met** |
| 2 | step 1: a **deep-RL best response** against a FIXED mean field | `pi*_k = argmax_pi J(pi, Gbar_{k-1})`, "SAC or PPO" | PPO on BenchMARL's IPPO host; the flow is frozen for the whole iteration (`torch.no_grad()` in `FpDensityInput`) | **met** |
| 3 | step 2: an SL buffer `M_SL` of `(t, s, a)` from **every** past best response | "the replay buffer M_SL accumulates (time, state, action) tuples from all previous best-response policies" | `SlReservoir`, added to every iteration | **met**, bounded — see (A) |
| 4 | step 2: the average policy is a **Gaussian fitted by maximum likelihood** | `L_NLL(thetabar) = -1/M sum_i log N(a_i; mu(s_i,t_i), sigma(s_i,t_i))` | `gaussian_nll`, `sl_epochs` minibatches per FP iteration, its **own** Adam | **met, tested** against the density's definition |
| 5 | the average policy is a separate network from the best response | `pi_k*` and `pibar_k` are distinct | two models built from the same `model_config`; the partition is checked by identity and the algorithm raises if it fails | **met** |
| 6 | step 3: a **conditional normalising flow** over states given `t`, trained by MLE | `L_NLL = -1/N sum_i [log p0(f^-1(x_i,t_i)) + log|det ...|]` | `ConditionalFlow`: affine coupling layers with alternating masks, each conditioner reading `t`; exact log-determinant | **met**, different transform — see (B) |
| 7 | the flow is **autoregressive neural spline flows, 16 layers** | "Neural Spline Flows with autoregressive layers ... 16 transformation layers" | 8 affine coupling layers | **adapted, architecture** — see (B) |
| 8 | time conditioning: the spline parameters are functions of `t` | "Time conditioning is incorporated by making spline parameters functions of the time variable t" | every coupling layer's conditioner takes `cat([x * mask, t])` | **met** |
| 9 | the flow answers **density queries** `mu(x)` | "the learned normalizing flow provides direct density queries mu(x)" | `FpDensityInput` evaluates `exp(log q(x_own | t))` at the agent's own state | **met** |
| 10 | the density enters the **reward**: `r(x, a, mu)` | "For problems with local density dependence ... the reward depends on the population density at the agent's location" | the density is appended to the **policy's input** instead | **adapted** — see (C) |
| 11 | the algorithm **returns `pibar_K`**, not `pi*_K` | Alg. 3's return line | `deploy_average_last: true` makes `pibar` the acting policy for the last FP iteration and zeroes the best-response objective | **met** — see (D) |
| 12 | `Gbar_k` is trained on trajectories from `pibar_k` | Alg. 3 step 4 | trained on the states in `M_SL`, which is the union over every best response so far | **adapted** — see (E) |
| 13 | the population is `N-1` agents playing the average policy | "the distribution induced by the N-1 agents using the average policy from the previous iteration" | **not implemented**: every agent plays `pi*_k` and the population enters only through the frozen `Gbar_{k-1}` | **NOT met** — see (F) |
| 14 | the mean field enters through the **distribution only** | the MFG abstraction itself | met, and that is what (13) leans on: `Gbar_{k-1}` is the population | **met** |
| 15 | `K` in the range 5–20 | "typically 5-20" | `fp_iters: 10` | **met** |
| 16 | policy network 2 x 256, SAC/PPO lr `3e-4` | hyper-parameter list | BenchMARL's `model` config and `experiment.lr` decide the BEST RESPONSE's; the flow and the average policy carry the paper's `3e-4` in their own optimisers | **adapted, deliberate**: the BR must use the same network and learning rate as every other row or the comparison is about the architecture |
| 17 | exploitability as the reported metric | `e_k^true < ...` | **not implemented**: exploitability needs a best response to the *deployed* policy, i.e. a second full training run per measurement | **NOT met** — see (G) |
| 18 | `N_sa` samples per iteration, `N` population size | hyper-parameter list | `sl_subsample` controls the first; the second is the host's `n_agents`, which is not free here | **adapted, structural** |

### (A) `M_SL` is bounded

The reference never forgets. A 3 M-frame run at 4 agents would put ~12 M
triples in the buffer, so it is a **reservoir** of `sl_capacity` (200 000 by
default), which keeps a *uniform* sample of everything ever added — so
`pibar` is still fitted to the uniform mixture over iterations that fictitious
play calls for, rather than to the recent ones. That is the property that
matters; the size is not.

### (B) An affine coupling flow, not a neural spline flow

Both are conditional normalising flows: an exactly invertible map with a
tractable log-determinant, trained by maximum likelihood, conditioned on `t`.
They differ in the elementwise transform (affine vs monotonic rational
quadratic spline). The spline is more expressive per layer; the coupling layer
is a few dozen lines and has an exact, checkable log-determinant.

`verify.py` checks the two properties that make it a flow at all: the layer is
exactly invertible, and **the log-determinant it reports is the one autograd
computes from the Jacobian**. If those disagree the "density" is not a density
and the fit is not maximum likelihood.

This is a real reduction in capacity and it is the first thing to change if the
flow underfits. Watch `dedafp_flow_nll`.

### (C) The density goes to the policy, not the reward

In the paper the density enters the reward, because an MFG's reward is
*defined* to depend on the population. Here it goes into the policy's input
instead, and that is deliberate:

**a VMAS task's reward is the task's, and every arm in this repo shares it.**
`simple_ns` is built so that the dial's physics reach every arm identically
(NS-3.1); a baseline that rewrote the reward function would not be comparable
with any other row in the table — including its own reference arm. So the
density is supplied through the other channel that carries the same
information.

This is the one place this row departs from Algorithm 3, and it should be
stated in the paper as such.

### (D) The last iteration measures `pibar`

Fictitious play's output is the average, not the last best response. BenchMARL
reports the return of whatever policy collected, so with
`deploy_average_last: true` the final FP iteration switches the acting policy
to `pibar`, zeroes the best-response objective, and stops adding to `M_SL`.
That iteration **measures** `pibar_K` and trains nothing but the critic. It
costs one tenth of the budget.

`deploy_average_last: false` reports the last best response instead. **Say
which one the table shows**; they are different policies.

### (E) The flow is fitted to `M_SL`'s states

Algorithm 3 fits `Gbar_k` to trajectories drawn from `pibar_k`. Rolling out
`pibar` would need a second collection pass per iteration. What is used instead
is the state distribution already in `M_SL`, which is the union over the best
responses so far — i.e. the empirical occupancy of the fictitious-play
*average*. In the occupancy-measure formulation of fictitious play those are
the same object; as a distribution over states they are an approximation,
because the average of the occupancies of `pi*_1..k` is not exactly the
occupancy of their behavioural mixture. Stated, not hidden.

### (F) The population does not literally play `pibar`

In the finite-`N` reading of Algorithm 3, the representative agent best
responds while the other `N-1` play `pibar_{k-1}`. Implementing that would mean
masking the PPO loss to one agent per iteration and throwing away `(N-1)/N` of
every batch, and the reported team return would then be a mixture of one best
response and `N-1` average policies — a number that is hard to read against any
other row.

What is implemented is the **mean-field reading**: every agent plays `pi*_k`,
and the population enters only through the frozen `Gbar_{k-1}`, which is what
the mean-field abstraction says the population *is*. The finite-`N` correction
is the part that is missing, and at `N = 4` it is not a small one. A reviewer
should be told this row is the mean-field limit of the algorithm and not its
`N`-player instantiation.

### (G) Exploitability is not measured

The paper's headline metric needs, for each measurement, a full best-response
training run against the deployed policy. That is a second experiment per
checkpoint, and it is out of scope for a baseline row. What is logged instead
is `dedafp_sl_nll` (how well `pibar` fits the best responses it is averaging)
and the return of `pibar` itself, which is the quantity the rest of the table
is measured in.

## Running it

```bash
GROUP=x2 bash scripts/run_extra_baselines.sh      # dedafp, dedafp_br
```

`dedafp` reports `pibar_K`; `dedafp_br` is the same run reporting the last best
response. **Requires `task.ns_observe_time=true`** — the launcher sets it.

## What to watch

| column | what it means |
|---|---|
| `dedafp_iter` | which fictitious-play iteration is running. Should reach `fp_iters - 1`. |
| `dedafp_mode` | 0 while the best response acts, 1 once `pibar` takes over. |
| `dedafp_sl_nll` | the average policy's fit to `M_SL`. If it does not fall, `pibar` is not an average of anything and the row is just PPO. |
| `dedafp_flow_nll` | the flow's negative log-likelihood. If it plateaus high, the coupling flow is underfitting — see (B). |
| `dedafp_density` | the mean density handed to the policy. **If it sits at `density_clip` the flow has collapsed** onto a spike and the channel is noise. |
| `dedafp_buffer` | how full `M_SL` is. |
