# Coupling Under Drift, ported to VMAS `road_traffic`

An instantiation of `PACT_NS_SPEC` in `vmas/road_traffic` — the SigmaRL lanelet
scenario VMAS ships, on the CPM Lab CommonRoad map.

```bash
python road_ns/conformance.py   # I.7, 14 checks, offline      ~5 s
python road_ns/report.py        # I.6 + I.5, commit the output ~20 s
python pact1/selftest.py        # build-order step 5           ~16 s
```

All three run with **torch alone** — no vmas, no torchrl, no learning
framework. That is deliberate: the spec's build order puts the cheapest
disqualifying result first, and none of steps 2–5 should need a simulator.

---

## Why `road_traffic`

The Part III.1 questionnaire, answered honestly. Four of the five structural
questions transfer verbatim from URB, which is what makes this an instantiation
rather than an invention.

| | question | answer |
|---|---|---|
| Q1 | shared medium, published capacity | **yes** — lanelet occupancy ratio; capacity = lanes × length / (vehicle + IDM jam gap) |
| Q2 | `W` from structure, before any run | **yes** — lanelet–route incidence ÷ capacity, parsed straight from `road_traffic_cpm_lab.xml` |
| Q3 | proprioception | **yes** — realized vs free-flow traversal of the agent's own route |
| Q4 | exogenous driver named unprompted | **yes — rain**, and the σ=1 anchor is then the *same published HCM capacity adjustment factor*, 14% |
| Q5 | does the harm have an inverse | **no** — you cannot subtract seconds off a congested lanelet. **Identification and steering only** |
| Q6 | coordination gap | **37.3%** at the shipped fleet size — see below |
| Q7 | loaded enough | **yes**, `u ≈ 0.26`, the same regime as URB's `v/c ≈ 0.219` |

`sampling`, `navigation` and `discovery` — the obvious VMAS benchmarks — fail
Q1 and Q4 outright. They have no shared medium and no driver a practitioner
would name, so instantiating this form there means inventing both, which is the
self-designed-benchmark critique the spec exists to escape.

---

## The four objects (I.1)

```
u_i(t)  = max over lanelets a on i's route of  load_a / (capacity_a * g_a)
Op[a,p] = 1[lanelet a used by route p] / capacity_a
W[i,j]  = mean over a in E(i) of Op[a, route(j)]   for j != i,   W[i,i] = 0
A(t)    = sin^2( pi * min(phi/w, 1) ) if phi < w else 0      P = 100, w = 0.5
g_a     = clip( 1 - sigma * 0.14 * A(t) * s_a , 1e-3 , 1 )
harm_i  = f(u_i derated) / f(u_i nominal),      f(u) = 1 + alpha * u
```

The harm reaches the agent by **dividing the achievable speed**. VMAS's
`KinematicBicycle` takes `action.u[:,0]` as a direct velocity command, so a
congested, rain-derated lanelet simply caps how fast the vehicle can go. The
reward function is never touched: the agent is paid exactly what it was paid
before, for a journey the medium made slower (NS‑1.4).

**Measured on the shipped map** (104 lanelets, 3 element classes, 1548 routes of
3–7 hops):

```
operator     spread(std/mean) = 0.564    asymmetry = 0.093    zero diagonal
dial         sigma=1 removes 3.50% of capacity, peak/trough 1.347x
placebo      51 of 100 steps exactly dry, g == 1.0000 bit for bit at every sigma
loading      u_mean = 0.262   (URB reference: v/c = 0.219)
```

## The ceiling (I.5) — committed before any method code

```
              irreducible    own      PEER
 4/20 (20%)      51.2%      23.5%     25.3%
 8/20 (40%)      34.0%      28.7%     37.3%   <- URB at 40%: 42.6 / 16.5 / 40.8
12/20 (60%)      21.4%      31.5%     47.1%
16/20 (80%)      15.5%      30.6%     53.9%
20/20 (100%)      0.0%      30.2%     69.8%   <- URB at 100%: 18.9 / 81.1
```

NS‑4.2's prediction holds: the gap rises monotonically with controllable share,
because irreducible load disappears while peer load does not. Full table in
`runs/road_ns_ceiling.csv`.

---

## Two arithmetic errors this build already paid for

Recorded in the spirit of III.4, because both produced numbers that looked
entirely reasonable.

**Load and capacity must be the same kind of quantity.** Crediting a vehicle's
whole unit to every lanelet on its route — a *flow* — against a *spatial*
capacity reported `u_mean = 1.77`, a medium impossibly far over capacity, and a
coordination gap of 37.8% that was pure artefact. A vehicle occupies one lanelet
at a time; it contributes to each the fraction of its journey spent there.

**Then the own-share must follow.** Fixing the load but leaving `share_own`
crediting a full unit at the binding element overstated `own` roughly fivefold
and drove the peer share to **0.0%** — which reads exactly like "this domain has
no coordination gap" and would have disqualified the environment on an
arithmetic slip.

Both are why NS‑4.1 wants the decomposition computed and committed first: they
surfaced in an afternoon rather than after a training sweep.

---

## Status against the spec

### Implemented and verified offline

| requirement | where |
|---|---|
| NS‑1.1 loading ratio, **max** over the agent's elements | `dial.loading` |
| NS‑1.2 declared operator, zero-diagonal, spread, asymmetric | `structure.coupling` |
| NS‑1.3 exogenous driver, capacity only | `dial.driver_A` |
| NS‑1.4 harm through the performance function, reward untouched | `dial.harm` |
| NS‑2.1…2.5 identity / monotone / never generous / HCM anchor / placebo | `dial.dial_g` |
| NS‑4.1, 4.2 ceiling decomposition and fleet scan | `ceiling.py` |
| NS‑5.1 mean-preserving variant | `DialParams.mean_preserve` |
| I.6 mandatory report | `report.py` |
| I.7 all 14 conformance checks | `conformance.py` |
| P‑1.1, 1.2 reduction: `r` = 3 element classes, independent of N and of elements | `pact1/core.Basis` |
| P‑3.1…3.4 zero-diagonal basis, brute-force verification, geometric centring, pruning | `pact1/core.Basis` |
| II.4, P‑4.1, 4.2 RLS, decentralized, dead-row skip | `pact1/core.RLS` |
| P‑5.1, 5.2 inverted trust prior, **prediction** confidence gate | `pact1/core` |
| P‑6.1 dimensionless shift | `pact1/core.steer` |
| P‑7.1 floor property, bit-exact | `pact1/core.steer` |
| P‑8.1 herd index, logged never acted on | `pact1/core.herd_index` |

### Adapted, and the adaptation is load-bearing

**The channel (II.6).** URB steers over discrete route options and subtracts
`g·κ·z` from each option's logit. `road_traffic` assigns a reference path at
reset and its action is continuous, so there is no option set to rank. The
channel here is a **differential pace shift** — an agent whose route is
predicted more congested than the fleet average eases off, one on a clear route
presses on. This is the shift the spec itself names for traffic ("only a
differential one helps"), it is loop-coupled the same way, and it is
dimensionless by the same z-score. There is still no inverse, so this instance
sits in II.6's second row: **identification and steering only**.

**Route enumeration must produce variable-length routes.** `W[i,j]` is a mean
over *i*'s own elements, so fixed-length routes make it exactly symmetric and
fail NS‑1.2 for a reason that is an artefact of the enumeration rather than a
property of the road.

**Element classes** are `laneletType × (single/multi-lane)` — public
infrastructure, painted on the road, exactly as P‑1.2 requires. That gives r=3
on this map.

### Not implemented

**P‑6.2 — trust reached through the existing objective.** URB's shift sits
inside a softmax, so `log π(k|s)` depends on `w` and the ordinary policy
gradient flows into trust with no new sampled dimension. `road_traffic` has
continuous Gaussian actions and the pace shift is a deterministic transform of
the *sampled* action, so `∂ log π / ∂g = 0` and a trust head would receive **no
gradient at all**.

The faithful continuous analogue is to apply the shift to the policy's own
pre-squash mean rather than to the sampled action, which puts `g` inside the
distribution and restores the gradient. That requires wrapping the BenchMARL
actor and is not done here.

Consequence for II.11's arms: **`off` and `fixed` are implementable today;
`learned` is not.** Since P‑5.1 starts trust near full reliance, `fixed` is
close to the intended operating point and is the arm that answers "is trust
learned, or merely well-initialised?" — so the missing piece is the answer to
that question, not the method.

**The environment wrapper** (build-order step 6) and everything downstream of
it — the reward/record harm counters of NS‑3.2 and NS‑3.3, the maximum-excitation
probe of step 7, the sweep of step 8. Steps 2–5 are complete; step 6 needs the
simulator and has not been written or run.

`alpha = 2.28` in `DialParams` is inherited from URB and is a **placeholder**
until the probe measures it here. III.1's procedure — mean relative excess
divided by mean loading — is implemented as `dial.calibrate_alpha` and must be
run on this instance before any result is quoted.
