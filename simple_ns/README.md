# Interaction-mediated, **invertible** non-stationarity on stock VMAS scenarios

One layer, four stock hosts, no edits to the installed vmas.

```bash
python simple_ns/check_plumbing.py              # config consistency, torch only  ~1 s
python simple_ns/conformance.py                 # 11 offline checks, torch only   ~2 s
python simple_ns/smoke.py                       # 10 in-simulator checks          ~1 min
python simple_ns/calibrate.py --hosts transport # pick sigma, before any training
```

Run `check_plumbing.py` first. It catches the failures that otherwise surface
only on the cluster: a task enum listed in `benchmarl/environments/__init__.py`
but never imported (which killed the first launch with a `NameError` inside
`import benchmarl`), a yaml key with no dataclass field, a dataclass field with
no yaml value, a knob the layer never reads, and any drift in the dial or the
method between the four hosts.

## Which cell this is

`NS_design_guide.md` sorts non-stationarity three ways, and `PACT_NS_SPEC` II.6
cuts one of them again:

| | invertible channel | no inverse |
|---|---|---|
| **(C) interaction-mediated** | **this package** — an additive force in the agent's own action space | `road_ns` — a congested lanelet cannot be un-congested |
| **(B) exogenous** | `ns_direct: true`, the control | — |

`road_ns` sits in the bounded cell: the method can only *steer*, so the
recoverable margin is capped by the coordination gap and the measured effect at
the anchored severity was 2–3 %. This package sits in the cell where a correct
estimate **cancels** the disturbance, which is where a return curve can fall a
long way and be brought back.

## The mechanism

Agents act on a shared medium — a rigid payload in `transport` and `balance`,
the surrounding fluid in `sampling` and `navigation` — and the medium transmits
each agent's exertion to the others. A redundant actuator feels its partners
fighting it through the body it is holding; a rotorcraft flies through its
neighbours' downwash. Both are named effects with their own literature.

How strongly the medium transmits drifts with an exogenous driver `A(t)`:
bearing compliance over a deployment, air density over the operating day.

```
Q_m,i(t) = rho Q_m,i(t-1) + (1-rho) * sum_{j != i, type(j)=m} W_ij u_j(t-1)
q_i      = sum_m Q_m,i        e_i = q_i / ||q_i||        PUBLIC
x_m,i    = <Q_m,i , e_i>                                 PUBLIC   psi = [1, x]
d_i      = e_i * ( beta*(t) . psi_i )                    PRIVATE
beta*_m  = sigma * L * A(t) * send_m
u_exec   = clip( u_command - g*e_i*pred_i  +  d_i )
```

The **direction is public and the gain is unknown** — an agent can see where its
neighbours pushed and cannot see what that costs it today. That split is what
makes a scalar estimate able to cancel a vector disturbance, and it is why the
channel is invertible.

Every sum is strictly over `j != i`, so at N=1 the disturbance is **exactly**
zero at any severity. Category C structurally, not approximately.

## The result, before any training

`simple_ns/calibrate.py` on `transport`'s own shipped heuristic (3 seeds, 32
envs, 250 steps). `oracle` is the free-answer controller of ANT §2.2 — handed
the true disturbance — so it is the ceiling, not a competitor:

| σ | blind | **PACT** | oracle | clip |
|---|---|---|---|---|
| 0 | 100.0 % | 100.0 % | 100.0 % | 0 % |
| 1 | 87.8 % | 103.8 % | 108.1 % | 10 % |
| 1.5 | 64.5 % | 90.0 % | 109.7 % | 15 % |
| **2** | **50.3 %** | **80.9 %** | **111.7 %** | 17 % |
| 3 | 37.1 % | 62.7 % | 110.3 % | 22 % |

σ = 2 is the committed operating point: the blind arm loses half its return
while the ceiling is still full recovery, so the row measures the **method**
rather than the environment. Nothing here is past σ\*.

## Host verdicts, measured

| host | verdict |
|---|---|
| **transport** | **the headline.** Falls far, monotonically, ceiling stays high. |
| sampling | second host, same layer |
| navigation | cheapest; debug here. Its shipped heuristic needs `cvxpy`, so `calibrate.py` skips it. |
| balance | **not monotone** — 87.7 % of B0 at σ=1 and 91.1 % at σ=16. Its return is dominated by a sparse catastrophic term (the package falling), so being shoved around sometimes *prevents* the fall. Not a headline row. |

## Three traps this build already paid for

**The dial has to be normalised to the host's own geometry.** A uniform-random
reference separates agents by ~1.0 while `balance` puts its supports ~0.25
apart, and with a 0.35 length scale that is a factor of ~14 in transmitted
force: σ=1 delivered 0.50 of the action range instead of 0.14, and the ladder
stopped being monotone. The reference is now drawn from the scenario's own
`reset_world_at`.

**The oracle must be an arm, not a wrapper.** Computed outside the environment
the best available answer is one step stale, which understated the ceiling
enough that PACT appeared to *beat* it — 78.9 % against 65.0 % at σ=2, not a
possible result.

**A diverged estimator can score well by accident.** Unbounded, a meaningless
correction (prediction error 1e34) is clamped by the action box into a large
saturating kick that scored **better** than a correct estimate: 96.2 % of B0
against 84.7 %. `pact_corr_clip` closes that route, and μ is selected on
prediction error, never on return.

## μ is a stability constraint here, not only a bias/variance choice

Swept at σ=2 in two excitation regimes, because they disagree:

| μ | memory | %B0 | err (heuristic) | err (random actions) |
|---|---|---|---|---|
| 0.999 | 1000 | 74.7 % | 0.640 | 0.0282 |
| 0.99 | 100 | 76.9 % | 0.611 | 0.0267 |
| 0.98 | 50 | 79.6 % | 0.592 | 0.0248 |
| **0.97** | **33** | **80.9 %** | **0.576** | **0.0231** |
| 0.95 | 20 | 82.2 % | 0.517 | **diverged** |
| 0.90 | 10 | 81.2 % | 9.5e33 | **diverged** |

0.95 is best under the heuristic and blows up under a random action stream,
where the channels are small and the covariance inflates. Early RL training is
near-random, so 0.95 would have died in the first iterations of every run. The
proper fix is a covariance bound in the RLS, which is **not implemented** — so
this is a limitation to state, not a tuned value to present as a default.

## What is shared, and what is not

`pact1/core.py` is the *same object* used by `road_ns`: the RLS with its
dead-row skip, the inverted trust prior, and the prediction-based confidence
gate. Only the channel differs — `compensate` (exact inverse) here,
`steer` (ranking shift) there. A cross-cell comparison run on two separately
implemented methods would measure the implementations.

`vmas/scenarios/balance_ns.py` and `navigation_ns.py` are earlier attempts at
the same idea and are **not used**: their severity is a module-level constant so
the dial cannot be read from the task config (NS-3.1), and their channel is a
multiplicative derate with no inverse — the same bounded cell `road_ns` is
already in.

## Not implemented

- **P-6.2, trust through the objective.** The correction is a deterministic
  transform of the sampled action, so `∂ log π / ∂g = 0` and a trust head gets
  no gradient. `off` and `fixed` are implementable; `learned` is not.
- **A covariance bound in the RLS** — see the μ section.
- **NS-2.4's published anchor.** `road_ns`'s σ=1 is the HCM heavy-rain factor,
  a number a traffic engineer defends. These are synthetic arenas with no such
  constant, so σ=1 is a *stated calibration procedure* instead. Say so; do not
  present it as an anchor it is not.
- **II.9 gates 5–7 during training** — `fit_gain` and `cond_psi` are computed
  offline, not logged per iteration.
