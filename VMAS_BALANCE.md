# `vmas/balance` under Coupling-Under-Drift

What the task is, what we inject into it, and requirement by requirement how the
URB spec (`PACT_NS_SPEC`) is covered — including what is **not** covered.

Written to be readable without the code. Config: `benchmarl/conf/task/simple_ns/balance.yaml`.

---

## 1. The task, in plain words

Four small robots stand under a beam. A package sits on the beam. Somewhere
above them is a target. The robots must **push the beam up to the target** while
**keeping it level** — tilt it and the package rolls off, the episode ends, and
they take a −10 penalty.

That is a **cooperative lift**: several actuators raising one shared load
together. The reward is dense — you are paid for closing the distance between
the package and the target — plus that one big penalty for dropping it.

```
n_agents          4          the supports
package_mass      5          the load
line (beam) mass  5          rotatable; tilting it is what loses the package
max_steps         100
fall penalty     -10         once, and the episode ends
```

The hard part is not lifting. It is that the four supports have to **agree**.
If one pushes harder than the others the beam rotates, and once it rotates
enough the package is gone.

---

## 2. What goes wrong in real life

This is a real industrial operation: a **synchronised lift**. Several hydraulic
jacks raise one load together — bridge jacking, ship lifts, installing heavy
machinery, jacking an aircraft.

And it has a famous failure mode:

> The jacks all run off **one power pack** — one pump, one accumulator, one
> pressure rail. Each jack's controller asks the rail for flow. When several
> jacks draw at once, the rail **sags**, and every jack delivers less force than
> it asked for.
>
> How far the rail sags for a given draw is its **stiffness**, and that is not
> constant. Over a shift the fluid warms up. Viscosity falls. Leakage past pump
> and valve clearances rises. The accumulator's gas pre-charge bleeds down. **The
> rail gets softer as the day goes on.**
>
> This is why synchronised-lift rigs are commissioned cold and then drift. The
> jacks that were matched at 08:00 are not matched at 14:00. Operators know it,
> and the standard fix is a per-jack **pressure feed-forward**: work out how much
> you are about to lose, and ask for that much extra.

Everything we inject is that paragraph. Nothing else.

---

## 3. What we actually inject

**In one sentence:** each support delivers less force than it commanded, by an
amount that depends on *how hard its neighbours are pulling* and on *how warm the
fluid is*.

### The pieces

**The driver `A(t)` — the fluid warming up.**
A smooth bump over a 100-step cycle that starts and ends at **exactly** zero.
Half of every cycle is exactly quiet — that is the placebo, and it is free.
Average over a cycle: `A = 0.25`.

**The draw — what your neighbours are pulling.**
A draw on a pressure rail is a *scalar*: what costs you pressure is **how much**
your neighbours are pulling, not which direction they are pushing.

```
x_m,i(t) = rho * x_m,i(t-1)  +  (1-rho) * sum over j != i of type m:  W_ij * ||u_j(t-1)||
```

`rho = 0.9` is the rail's own lag — pressure does not respond instantly.

**`W` — the plumbing, written down before any run.**
`W_ij = recv_i / (1 + (distance_ij / lambda)^2)`, and `W_ii = 0`.
Jacks closer on the manifold couple harder. The zero diagonal is the whole point:
**your own draw is not something that happens to you.**

**The classes — jacks are not identical.**
Three declared classes (`r = 3`), fixed before training, independent of how many
agents there are:

| | class 0 | class 1 | class 2 |
|---|---|---|---|
| `recv` — how much droop you **feel** | 1.6 | 1.0 | 0.4 | **public** |
| `send` — how much your draw **costs others** | 0.2 | 1.0 | 1.8 | **unknown** |

A jack on a long thin hose feels more droop and causes less. You know your own
plumbing; you do **not** know what a neighbour's draw costs you *today*, because
that depends on the fluid state. That split is the whole method.

**The droop, and how it reaches the robot.**

```
droop_i(t) = sigma * L * A(t) * sum_m send_m * x_m,i(t)      clamped to [0, 0.9]
delivered  = commanded * (1 - droop_i)
```

The reward function is **never touched**. The robot is paid exactly what it was
paid before; it simply achieves less, because its actuator delivered less.

Measured, with a scripted lift controller:

| σ | mean droop over a cycle |
|---|---|
| 0 | **exactly 0.0000** |
| 1 | 0.057 |
| 2 | 0.105 |
| 4 | 0.167 |

**Why this disturbance and not another.** A derate on a *lift* is what drops the
load. You ask for the force that holds station, you get less, and the beam tips
fastest on whichever support is drooping worst. It attacks the **levelling
loop** — the loop this task actually fails at. A sideways shove would be a
disturbance `balance` does not care about.

---

## 4. What the method does

The robot knows the flow it asked for and can measure the force it got, so the
**fractional shortfall is directly observable**:

```
y_i = 1 - ||delivered|| / ||commanded||        ( = droop_i )
```

Nothing privileged. It never sees another robot's residual.

It regresses that on the public channels, `psi = [1, x_1, x_2, x_3]`, with a
recursive least-squares estimator that forgets slowly (`mu = 0.95`). Four numbers
to learn, **no matter how many robots there are**.

Then it does what the rig operator does — asks for extra:

```
commanded_sent = commanded / (1 - trust * predicted_droop)
```

With a correct estimate the two cancel exactly. This is why the channel is called
**invertible**, and it is why `balance` can show a return falling a long way and
coming back — unlike `road_ns`, where a congested road cannot be un-congested and
the method can only steer around it.

It stops working at the **relief valve** (`droop_max = 0.9`) and at the robot's
own action limit. That is where σ\* comes from: physics, not a number we chose.

---

## 5. The compensation commons — measured, and unlogged

Asking for extra flow **draws more from the rail**, which sags it further, for
everyone. The fix feeds the problem. Measured at σ=2 over 300 steps:

| | mean rail draw | mean droop |
|---|---|---|
| blind | 0.609 | 0.1098 |
| PACT | 0.693 | **0.1196** |

Compensating raised the draw 13.7 % and the disturbance 8.9 %. This is II.8's
compensation commons — and here it is **literal**, not an analogy: a hydraulic
rig really does behave this way, which is why real installations put flow limits
on individual jacks.

**This is currently not logged.** P-8.1 says log the externality and do **not**
act on it. Acting on it would make the method a mechanism rather than a per-agent
estimator and break the decentralisation claim. Logging it is a gap — see §8.

---

## 6. How the spec is covered — Part I (the non-stationarity)

| req | what it demands | here | status |
|---|---|---|---|
| **NS‑1.1** | a shared medium with a loading ratio, max over the agent's own elements | the medium is the **pressure rail**. It has exactly one element, so "max over elements" is that element, and the loading is the draw on it. | met, degenerately — one element, so the max is trivial. `road_ns` is where NS‑1.1 bites. |
| **NS‑1.2** | operator `W` written from structure, never fitted; zero diagonal, spread, asymmetric | `W_ij = recv_i / (1 + (d/λ)²)`, from declared plumbing + geometry | **met.** Measured: diagonal exactly 0, spread 0.592, asymmetry 0.348 |
| **NS‑1.3** | exogenous driver, a function of observable time, reaching agents only by shrinking capacity | fluid temperature over a shift; it only scales what the *neighbours* cost you | **met** |
| **NS‑1.4** | harm through the environment's own performance function; **must not** subtract a penalty | delivered force is derated. Reward untouched, inherited from stock `balance` | **met** |
| **NS‑1.5** | declare whether a compensation channel with a known inverse exists | it does: divide by `(1 − est)`, the feed-forward real rigs use | **met, and declared** |
| **I.2** | category‑C signature: a lone agent reads harm **exactly** 1.0 at any severity | every sum is strictly `j ≠ i`, so one jack on the rail droops exactly 0 | **met, and tested in simulation** at σ=20 |
| **NS‑2.1** | identity at zero, exactly, over the whole driver domain | σ=0 → droop exactly 0.0000; both arms bit-identical | **met, tested** |
| **NS‑2.2** | monotone in σ at every driver value | linear in σ with a non-negative coefficient | **met, tested** |
| **NS‑2.3** | never generous — the disturbance may not help | droop clamped to `[0, 0.9]`; delivered force only ever falls | **met** |
| **NS‑2.4** | σ=1 anchored to a **published constant** | `loss_at_sigma1 = 0.14`, carried from the HCM figure so the two families share a scale. A *stated procedure*, not an anchor. | **NOT met** — see §8 |
| **NS‑2.5** | a placebo regime where the dial is provably inert | `A(t)` is exactly 0 for 51 of 100 steps; `wet_fraction=0` makes it inert at every σ | **met, tested** |
| **NS‑3.1** | the dial sits **below** the method and is read from the task config | `ExertionMixin` is below `PactMixin`; every key comes from the yaml | **met** |
| **NS‑3.2** | harm the **records**, not just the rewards | the derate is applied in `process_action`, below the action interface, so every logged trajectory carries it | **met, tested** |
| **NS‑3.3** | fail loudly when the layer is inert | counters + `assert_layer_fired()`; `InertLayerError` raised per batch during training | **met** |
| **NS‑3.4** | the driver's clock persists across episodes | `_step` is not reset — the fluid does not un-warm because an episode ended | **met** |
| **NS‑4.1 / 4.2** | ceiling decomposition, and the gap vs controllable share | **degenerate here.** With no uncontrolled participants and a strictly `j ≠ i` sum, `Δ_fixed = 0` and `Δ_own = 0`, so the coordination gap is **100 % by construction**. | **not informative** — see §8 |
| **I.6** | report capacity removed, swing, placebo days, loading, gap, records harmed | the debug CSV carries load, driver, clipped fraction, live fraction per iteration | **partly** — no capacity-removed-over-cycle table in `road_ns`'s form |
| **I.7** | the conformance suite, ported | 11 offline checks, torch only, including the (B)-vs-(C) and (A)-vs-(C) decision procedures as *measurements* | **met** |

---

## 7. How the spec is covered — Part II (the method)

| req | what it demands | here | status |
|---|---|---|---|
| **P‑1.1** | `r` independent of N and of elements | `r = 3` classes at N = 2, 5, 11, 40 | **met, tested** |
| **P‑1.2** | know the model class, estimate the parameters | classes declared from hardware before training; only `send` is estimated | **met** |
| **P‑2.1** | sensor is the agent's own relative excess, nothing privileged; clip declared | `y = 1 − delivered/commanded`, exactly observable | **met** |
| **P‑3.1** | zero-diagonal basis; a lone agent reads exactly zero on every channel | strictly `j ≠ i` | **met, gated at startup** |
| **P‑3.2** | verify the vectorised basis against brute force at startup, abort on mismatch | gate 1 prints `channels == definition to 0.00e+00` on every run | **met** |
| **P‑3.3** | centre and scale on a geometric reference | reference drawn from `balance`'s **own spawn distribution**; condition number 594 → 7 | **met** (and see §9) |
| **P‑3.4** | prune degenerate channels; verify option ordering | not implemented — `r = 3` and all three classes are populated at N=4 | **NOT met** |
| **P‑4.1** | decentralized — never another agent's residual | only peers' broadcast actions | **met** |
| **P‑4.2** | skip rows whose regressor is numerically zero | liveness tested on the **channel** columns, not on `psi` | **met** (this was broken; see §9) |
| **P‑4.3** | μ declared, swept, never a swept value presented as a default | swept 0.999 → 0.90, table in the yaml | **met** |
| **P‑5.1** | invert the prior — start near full reliance | `pact_trust = 0.9` | **met** |
| **P‑5.2** | gate on **prediction** uncertainty, never on `tr(P)` | `confidence()` uses `psi'Pψ` | **met** |
| **P‑5.3** | report policy-set and applied trust as separate columns | `pact/trust_policy` and `pact/trust_applied` | **met** |
| **P‑6.1** | make the shift dimensionless | the correction is a *ratio*, inherently dimensionless | **met** |
| **P‑6.2** | reach trust through the existing objective, no new sampled dimension | the correction is a deterministic transform of the **sampled** action, so `∂log π/∂g = 0` | **NOT met** — `learned` arm impossible |
| **P‑7.1** | at `g=0` return the untouched policy, bit for bit, for any estimate | tested in simulation; also guarded against non-finite estimates | **met, tested** |
| **P‑8.1** | log the externality, do not act on it | measured (§5) and **not logged** | **NOT met** — see §8 |
| **II.9** | seven gates | 1, 2, 3 at startup (abort); 5, 6, 7 offline only | **partly** |
| **II.10** | the instrument panel, one row per step | one row per *iteration* in `pact_debug.csv`, including β-against-truth | **met** |
| **II.11** | three arms through the identical wrapper | `blind`, `pact`, `pactoff` (trust=0, bit-identical), plus `oracle` (ceiling) and `intercept` (peer channels deleted). `learned` impossible — see P‑6.2 | **met except `learned`** |

---

## 8. What is **not** covered, and what it costs

1. **NS‑2.4 — no published anchor.** σ=1 delivers 0.14 of the action range
   because that is the HCM figure `road_ns` uses, so the two ladders are readable
   against each other. It is a *stated calibration procedure*, not an anchor.
   **The droop story makes a real anchor reachable** — hydraulic fluid viscosity
   against temperature is standardised (ISO VG, viscosity index) and pump
   volumetric efficiency against temperature is a documented curve — but we have
   not done it. Say "stated procedure" in the paper, not "anchored".

2. **P‑8.1 — the commons is real and unlogged.** §5 measures it: compensating
   raises the rail draw 13.7 % and the droop 8.9 %. Log it (total rail draw, and
   droop with vs without compensation) and **do not act on it**.

3. **NS‑4.1/4.2 — the ceiling decomposition is degenerate.** 100 % peer by
   construction, so it is not a useful diagnostic here. To make it informative,
   add *uncontrolled* jacks drawing on the same rail — background demand — which
   would restore `Δ_fixed` and make the controllable-share sweep meaningful, as it
   is in `road_ns`.

4. **P‑6.2 — the `learned` arm does not exist.** Trust is fixed at 0.9. Since
   P‑5.1 starts near full reliance, `fixed` is close to the intended operating
   point, so what is missing is the answer to "is trust learned or merely
   well-initialised", not the method.

5. **P‑3.4 — no channel pruning.**

6. **II.9 gates 5–7 do not run during training** — `fit_gain` and `cond_psi` are
   in the debug CSV but nothing aborts on them. Gate 6 is the one that saves
   compute.

7. **σ cannot be calibrated offline on `balance`.** The spec's procedure needs a
   *competent* controller, and `balance`'s shipped heuristic is on the ground
   **61 % of the time at σ=0** — with no disturbance at all. A hand-written PD
   lift controller gets that to 46 %; still not competent. So the ladder must be
   read against a **trained** B0 rather than a scripted one. Run B0 first; if it
   also sits at the fall floor, this host cannot carry the claim and that is a
   result about the host.

---

## 9. Failures already paid for (III.4, this instance)

Recorded so a port does not rediscover them at full price.

| shortcut | what it cost |
|---|---|
| Normalising the dial against a uniform-over-arena reference | `balance` spaces its supports ~0.25 apart, a uniform draw ~1.0; at λ=0.35 that is **14×** in transmitted force. σ=1 delivered 0.50 of the action range instead of 0.14 and the ladder stopped being monotone. |
| Centring `psi` on that same wrong reference | regressor came out at **11.8** instead of O(1); β uncorrelated with truth while `fit_gain` still read 0.86 |
| Computing the oracle outside the environment | one step stale, so PACT appeared to **beat** the ceiling (78.9 % vs 65.0 %) — not a possible result |
| Leaving the correction unbounded | a **diverged** estimator (error 1e34) scored **better** than a correct one, 96.2 % vs 84.7 %, because the action box turns garbage into a saturating kick |
| No covariance bound in the RLS | a real 3 M-frame run logged **5.7 M non-finite predictions** out of ~12 M agent-steps; the applied correction was 4 % of the disturbance. The method never ran. |
| Testing P‑4.2 liveness on `psi` | `psi` carries an intercept of exactly 1, so every row tested live and the dead-row skip **never fired** |
| Pairing `y(t−1)` with `psi(t)` | the target regressed on a near-independent row |
| Inheriting μ = 0.999 from URB | a 1000-step memory against a 100-step driver cycle — it averages the drift away |
| Judging `balance` by its shipped heuristic | 61 % on the ground at σ=0 made the host look "non-monotone" when there was simply no performance left to disturb |

---

## 10. Running it

```bash
python simple_ns/check_plumbing.py                 # config consistency
python simple_ns/conformance.py                    # 11 offline checks
python simple_ns/smoke.py --host balance           # 10 in-simulator checks
```

Then B0 first, because on this host it is the gate:

```bash
python simple_ns/run.py algorithm=mappo task=simple_ns/balance seed=0 \
  task.ns_severity=0 task.pact_enabled=false \
  experiment.max_n_frames=3000000 experiment.save_folder=runs/bal/b0
```

and the pair at the operating point, `task.ns_severity=2.0` with
`task.pact_enabled=false` / `true`. Each run writes `pact_debug.csv` beside
itself. Read it in the order the columns are documented: did the dial fire, how
big was it, does the reduction hold, is β being recovered, is the estimator
healthy, is trust armed, how much was cancelled — and only then, did it win.
