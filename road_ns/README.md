# Coupling Under Drift on the CPM lanelet medium

An instantiation of `PACT_NS_SPEC` on the CommonRoad map VMAS ships with its
`road_traffic` scenario. **Two hosts carry the identical medium**, and the only
difference between them is the vehicle model.

```bash
python road_ns/check_plumbing.py    # config consistency, both tasks   ~1 s
python road_ns/conformance.py       # I.7, 16 checks, offline          ~5 s
python road_ns/report.py --csv runs/road_ns_ceiling.csv   # I.6 + I.5  ~20 s
python pact1/selftest.py            # build-order step 5               ~20 s
python road_ns/probe.py             # build-order step 7 (needs vmas)  minutes
```

The first four run with **torch alone** — no vmas, no torchrl, no learning
framework. That is deliberate: the spec's build order puts the cheapest
disqualifying result first, and none of steps 2–5 should need a simulator.

If the map is not found, point at it:

```bash
export ROAD_NS_MAP=$(python -c "import vmas,pathlib;print(pathlib.Path(vmas.__file__).parent/'scenarios_data/road_traffic/road_traffic_cpm_lab.xml')")
```

---

## The two hosts

| task | fleet | envs | ms/frame | 1.2 M frames |
|---|---|---|---|---|
| `road_ns/lanelet_flow` blind | 16 + 24 bg | 600 | **0.271** | **5.4 min** |
| `road_ns/lanelet_flow` pact | 16 + 24 bg | 600 | 0.348 | 7.0 min |
| `road_ns/lanelet_flow` blind | 40 | 600 | 0.838 | 16.8 min |
| `road_ns/road_traffic` | 40 | 16 | 173 | **58 h** |
| `road_ns/road_traffic` | 16 | 64 | 29.8 | 10 h |

Measured on CPU through `vmas.make_env` with random actions. A full config
(5 seeds x 9 algorithms x 2 arms = 90 runs) is about **9 hours on one CPU core**.

Run the sweep on `lanelet_flow`; run `road_traffic` once at low N as a
provenance row (`scripts/cfg_provenance.sh`).

### Why `road_traffic` alone was not viable

Its cost is **not** the severity layer — that is 0.09 % of a `road_traffic`
step — and it is **not** the collision-reset storm. Overriding `done()` to
truncate instead of terminating on every collision changed the cost by nothing
(173 → 183 ms/frame), even though it took the fraction of env-steps terminating
from 94 % to 0.

The cost is the per-agent Python geometry: an O(N²) `interX` loop with a
`torch.nonzero` device sync per pair, five boundary distances per agent per
step, and the short-term reference-path resampling. That is most of its 4035
lines and no subclass reaches it.

### What `lanelet_flow` keeps and what it replaces

| kept | replaced |
|---|---|
| the CPM Lab CommonRoad map | collision meshes → vectorised centre-to-centre distances |
| `capacity = lanes × length / (vehicle + IDM jam gap)` | 5-point boundary distance → signed offset from the centre line |
| `W` = lanelet–route incidence ÷ capacity | short-term path resampling → a gather off the progress index |
| road_traffic's own seven reference loops | rejection-sampled per-env reset → one vectorised draw |
| the HCM anchor and the whole dial | terminate-on-collision → truncation + vectorised respawn |
| `KinematicBicycle` — the *same* vehicle model | — |
| the same reward coefficients | — |

**What this costs the story, stated plainly.** You can no longer say "we run
SigmaRL's published scenario". Every load-bearing claim survives: the medium is
a third-party map, the capacities come off it, the operator is written down
before any run, and σ = 1 is a published constant. Questions Q1, Q2 and Q4 of
III.1 are answered by the **map**, not by the vehicle model.

---

## Why this medium

The Part III.1 questionnaire, answered honestly.

| | question | answer |
|---|---|---|
| Q1 | shared medium, published capacity | **yes** — lanelet occupancy ratio; capacity = lanes × length / (vehicle + IDM jam gap) |
| Q2 | `W` from structure, before any run | **yes** — lanelet–route incidence ÷ capacity, parsed straight from `road_traffic_cpm_lab.xml` |
| Q3 | proprioception | **yes** — realized vs free-flow traversal of the agent's own route |
| Q4 | exogenous driver named unprompted | **yes — rain**, and the σ=1 anchor is then the *same published HCM capacity adjustment factor*, 14 % |
| Q5 | does the harm have an inverse | **no** — you cannot subtract seconds off a congested lanelet. **Identification and steering only** |
| Q6 | coordination gap | **26.7 %** at the committed 16-of-40 operating point; 82.9 % at 40 of 40 |
| Q7 | loaded enough | `u ≈ 0.17` at the operating point, the same regime as URB's `v/c ≈ 0.219` |

`sampling`, `navigation` and `discovery` — the obvious VMAS benchmarks — fail
Q1 and Q4 outright. They have no shared medium and no driver a practitioner
would name, so instantiating this form there means inventing both, which is the
self-designed-benchmark critique the spec exists to escape.

---

## The four objects (I.1)

```
u_i(t)  = max over lanelets a on i's route of  (load_a - 1[i is on a]) / (capacity_a * g_a)
Op[a,p] = 1[lanelet a used by route p] / capacity_a
W[i,j]  = mean over a in E(i) of Op[a, route(j)]   for j != i,   W[i,i] = 0
A(t)    = sin^2( pi * min(phi/w, 1) ) if phi < w else 0      P = 100, w = 0.5
g_a     = clip( 1 - sigma * 0.14 * A(t) * s_a , 1e-3 , 1 )
harm_i  = f(u_i derated) / f(u_i nominal),      f(u) = 1 + alpha * u
```

The harm reaches the agent by **dividing the achievable speed**. `KinematicBicycle`
takes `action.u[:,0]` as a direct velocity command, so a congested, rain-derated
lanelet simply caps how fast the vehicle can go. The reward function is never
touched: the agent is paid exactly what it was paid before, for a journey the
medium made slower (NS‑1.4).

**Measured on the shipped map** (104 lanelets, 3 element classes, 7 loops of
12–16 hops):

```
operator     spread(std/mean) = 1.331    asymmetry = 0.059    zero diagonal
dial         sigma=1 removes 3.50% of capacity, peak/trough 1.347x
placebo      51 of 100 steps exactly dry, g == 1.0000 bit for bit at every sigma
loading      u_mean = 0.170 at 16 of 40   (URB reference: v/c = 0.219)
```

### `-1[i is on a]` — the I.2 fix

The loading an agent experiences is **peer** loading. Its own vehicle does not
slow it down.

This was wrong until it was measured. With the agent's own unit left in the
element load, a **lone agent** read `harm = 1.0336` at σ=1 and `1.1388` at σ=3 —
it slowed *itself* down, which is a level shift and not a coupling, and it fails
I.2's own practical test ("an agent alone in the environment must read a harm of
exactly 1.0 in the worst storm you can dial"). The conformance suite missed it
because it only checked `structure.coupling` (the operator) and `harm(0, 0)`
(with `u` supplied by hand) — neither touches the sensor the run uses.

It also matters for identification: `Basis.channels` is strictly a sum over
`j ≠ i` (P‑3.1), so a target that still contains a self term asks the estimator
to explain something its regressor structurally cannot reach. That bias lands in
the intercept and inflates the irreducible share.

Gated by `test_lone_agent_reads_harm_exactly_one` and
`test_self_exclusion_is_the_zero_diagonal`. Set `ns_exclude_self: false` to
reproduce the old behaviour for the ablation table.

---

## The ceiling (I.5) — committed before any method code

At σ = 1, total fleet 40, `runs/road_ns_ceiling.csv`:

```
              irreducible    own      PEER
 8/40 (20%)      79.8%      17.3%      2.9%
16/40 (40%)      55.7%      17.6%     26.7%   <- the committed operating point
24/40 (60%)      32.6%      17.7%     49.6%      URB at 40%: 42.6 / 16.5 / 40.8
32/40 (80%)       9.4%      17.8%     72.8%
40/40 (100%)      0.0%      17.1%     82.9%   <- URB at 100%: 18.9 / 81.1
```

NS‑4.2's prediction holds: the gap rises monotonically with **controllable
share**, because irreducible load disappears while peer load does not.

**Background demand is what makes this testable.** `Δ_fixed` is "load from
participants no agent controls". `road_traffic` has none — every vehicle is an
agent — so the irreducible share is 0 at *every* fleet size and the mechanism
NS‑4.2 names cannot be exercised at all. `lanelet_flow` adds
`flow_n_background` vehicles that drive their route at a fixed pace and enter
the element load and nothing else. `scripts/cfg_fleet.sh` then sweeps the
controllable share holding the total at 40, which is the experiment the
prediction is about.

It used to sweep `task.n_agents` alone. That shrinks the whole fleet: every row
was 100 % controllable, irreducible was 0 throughout, and the numbers quoted in
the script's own comment belonged to a different experiment.

---

## Status against the spec

### Implemented and verified offline

| requirement | where |
|---|---|
| NS‑1.1 loading ratio, **max** over the agent's elements, peer-only | `dial.loading_by_route` |
| NS‑1.2 declared operator, zero-diagonal, spread, asymmetric | `structure.coupling` |
| NS‑1.3 exogenous driver, capacity only | `dial.driver_A` |
| NS‑1.4 harm through the performance function, reward untouched | `dial.harm`, `scenario.SeverityMixin.process_action` |
| NS‑2.1…2.5 identity / monotone / never generous / HCM anchor / placebo | `dial.dial_g` |
| NS‑3.1 the dial below the method, read from the task config | `scenario.SeverityMixin` |
| NS‑3.3 count harmed records, refuse to report an inert arm | `scenario.SeverityMixin.assert_layer_fired` |
| NS‑3.4 the driver's clock persists across episodes | `scenario.SeverityMixin.reset_world_at` |
| NS‑4.1, 4.2 ceiling decomposition and controllable-share scan | `ceiling.py`, `report.py` |
| NS‑5.1 mean-preserving variant | `DialParams.mean_preserve` |
| I.2 lone agent reads harm exactly 1.0 | `ns_exclude_self` |
| I.6 mandatory report | `report.py` |
| I.7 all conformance checks | `conformance.py` |
| P‑1.1, 1.2 reduction: `r` = 3 element classes, independent of N and of elements | `pact1/core.Basis` |
| P‑3.1…3.4 zero-diagonal basis, brute-force verification, geometric centring, pruning | `pact1/core.Basis` |
| II.4, P‑4.1, 4.2 RLS, decentralized, dead-row skip | `pact1/core.RLS` |
| P‑5.1, 5.2 inverted trust prior, **prediction** confidence gate | `pact1/core` |
| P‑6.1 dimensionless shift | `pact1/core.steer` |
| P‑7.1 floor property, bit-exact | `pact1/core.steer` |
| P‑8.1 herd index, logged never acted on | `pact1/core.herd_index` |
| P‑10.1 one entry point, severity from outside the method | `road_ns/run.py`, `scripts/_common.sh` |
| III.3 step 7 maximum-excitation probe | `probe.py` |

### Adapted, and the adaptation is load-bearing

**The channel (II.6).** URB steers over discrete route options and subtracts
`g·κ·z` from each option's logit. Both hosts here assign a route and take a
continuous action, so there is no option set to rank. The channel is a
**differential pace shift** — an agent whose route is predicted more congested
than the fleet average eases off, one on a clear route presses on. That is the
shift the spec itself names for traffic ("only a differential one helps"), it is
loop-coupled the same way, and there is still no inverse, so this instance sits
in II.6's second row: **identification and steering only**.

**`shift_mode = centred`, not URB's z-score.** This is a correction, not a
preference. `z = zscore(pred)` is exactly right when the shift enters a *logit*,
because only the ranking matters there. Multiplying a *physical velocity
command*, standardising throws away the one thing the channel needs to know —
how big the disturbance is. Measured: feeding predictions that differ by 1e‑6
still yields `|z| ≈ 1`, so at κ=1, trust=0.9 the z-score channel commanded **14 %
of agent-steps to reverse** and cut pace by more than half on 30 %, against a
real harm of 7–9 % at the storm peak. It applies the same compensation at
σ=0.01 as at σ=3.

`centred` subtracts the fleet mean and stops:
`shift_i = 1 − g·κ·(pred_i − mean_j pred_j)`. P‑6.1 asked for a *dimensionless*
shift and it already is — P‑2.1 defines the sensor as a **relative** excess
precisely so path-length scaling is out of it. Measured in simulation at σ=1:
shift ∈ [0.945, 1.023], fleet mean exactly 1.0000, 0 % reversed, 0 % clamped.
`shift_mode: zscore` is kept for the ablation.

`pact_shift_clip` is a guard rail, not a knob: it bounds `|shift − 1| ≤ 0.5` so
no diverged estimate can reverse a vehicle. Symmetric about 1, so P‑7.1 stays
bit-exact. At realistic spreads it never binds.

**Route turnover.** With the fleet layout frozen for an episode, every agent's
basis row is *constant* and there is nothing to identify — II.9 gate 5 would
fire. `flow_reroute_on_lap` gives a vehicle a new route when it completes its
loop, which is the honest reading of the event and the only source of excitation
in the episode.

### Not implemented

**P‑6.2 — trust reached through the existing objective.** URB's shift sits
inside a softmax, so `log π(k|s)` depends on `w` and the ordinary policy gradient
flows into trust with no new sampled dimension. Here the action is continuous
Gaussian and the pace shift is a deterministic transform of the *sampled*
action, so `∂ log π / ∂g = 0` and a trust head would receive **no gradient at
all**.

Consequence for II.11's arms: **`off` and `fixed` are implementable today;
`learned` is not.** Since P‑5.1 starts trust near full reliance, `fixed` is close
to the intended operating point.

The faithful fix is a **discrete option set** — a route choice at junctions, off
the map's own successor relation — which restores URB's logit channel verbatim,
makes κ genuinely a ranking constant, and puts `g` back inside the distribution
so the `learned` arm exists. That is v2.

**II.10's instrument panel is partial.** `fit_gain`, `pred_gain`, `cond_psi` and
`clip_frac` are computed by `probe.py` offline but are **not** logged per
iteration, so II.9 gates 5, 6 and 7 do not run during training. Gate 6 is the one
that saves compute.

**`alpha = 2.28` is still URB's figure and a declared placeholder.** III.1's
procedure — mean relative excess divided by mean loading — is implemented as
`dial.calibrate_alpha`, but it cannot be measured honestly from a scripted
driver on this host: there is no car-following model (agents are `collide=False`,
exactly as `road_traffic` has them), so at σ=0 congestion delays nobody except
through the proximity penalty, which only a trained policy responds to. Calibrate
from the first blind σ=0 checkpoint and re-run the ladder before quoting a
headline number. `road_ns/probe.py --calibrate` prints this.
