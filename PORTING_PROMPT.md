# Port the Coupling-Under-Drift form to a new environment

Attach this together with `PACT_NS_SPEC.html`. The spec is the contract; this is
what we learned building three instances of it, and the direction we settled on.

---

## The direction

The paper is **not** "replicate URB elsewhere". It is a **classification of
injected non-stationarity**, with PACT held fixed across the cells so that what
varies is the physics and not the method.

Two axes:

| | invertible channel | no inverse |
|---|---|---|
| **(C) interaction-mediated** — a lone agent feels *nothing* | the demonstration cell: a correct estimate **cancels** the disturbance | the bounded cell: the method may only **steer**, and the margin is capped by the coordination gap |
| **(B) exogenous** — a lone agent feels it | the control | — |

Each environment is one cell-instance. Your job is to add one, honestly labelled.

Existing instances, for reference:
`road_ns/` = (C, no inverse), a real road map, HCM-anchored.
`simple_ns/` = (C, invertible), stock VMAS hosts, two channels.

---

## What to build

1. **driver.py** — `A(t)`, the severity dial, the declared class constants.
   torch only, no simulator. `A(t)` must reach **exactly** zero over part of its
   cycle; that is the placebo and it is free.
2. **coupling.py** — the declared operator `W` and the basis. `r` classes,
   independent of N and of the number of elements. Every sum strictly `j != i`.
3. **layer.py** — a **mixin** that sits *below* the method and *above* the host,
   hooked at the single place every host reads an action. The host's `reward`,
   `done` and `observation` are inherited **untouched**.
4. **PACT** — import from `pact1/core.py`. Do not reimplement it. A cross-cell
   comparison run on two implementations measures the implementations.
5. **Arms** — `blind`, `pact`, `pactoff` (trust=0, must be bit-identical to
   blind), `oracle` (handed the true disturbance = the ceiling), and
   `intercept` (peer channels deleted, everything else identical).

---

## Pick the channel from the physics, then claim accordingly

| the disturbance is… | the channel | you may claim |
|---|---|---|
| additive in the agent's action space | subtract the estimate | identification **and** compensation |
| a multiplicative derate on delivered effort | divide by `(1 − est)` | identification **and** compensation, up to saturation |
| a property of a shared medium with no inverse | shift a ranking | identification **and steering only** |

Conflating these is the one thing the spec says is not publishable.

---

## The story test

Write the mechanism as prose in the task yaml, and make every clause a
requirement. It passes if a practitioner in that domain would say *"oh, that
actually happens"* — and if you can point, clause by clause, at:

- **interaction-mediated**: one agent alone is untouched, *structurally*, because
  the sum runs over `j != i` — not because the number is small;
- **exogenous**: a function of the clock that nobody controls, and the answer a
  practitioner gives unprompted to "what makes this harder some days?";
- **never a reward term**: it removes capability; the reward function is untouched;
- **the classes are real**: a public property of the hardware, and the split
  between what is public (your own plumbing/geometry) and what is unknown (what a
  neighbour's action costs you *today*).

Worked example, `balance` — N hydraulic jacks raising one load off one power
pack; when several draw at once the pressure rail sags; how far it sags drifts as
the fluid warms over a shift. Lone jack on the rail = full pressure, at any
temperature. The inverse is the feed-forward real rigs already use, and it dies
at the relief valve, so σ\* comes from physics rather than from a number we chose.

---

## Traps — each of these cost real time here

**Calibration**

1. Calibrate σ against a **competent** controller, never random actions. On
   `sampling` a random policy's return *rose* with severity — being shoved around
   spread the agents out and improved coverage.
2. **First check the host's own controller can do the task at σ=0.** `balance`'s
   shipped heuristic is on the ground 61 % of the time with no disturbance at
   all, so its ladder read −6.16 / −6.00 / −6.07 and looked "non-monotone". There
   was no performance left to disturb. If no competent scripted controller
   exists, you cannot calibrate offline — train B0 first and say so.
3. **Normalise the dial to the host's own spawn geometry.** A uniform-over-arena
   reference was off by 14× on `balance`, and σ=1 delivered 0.50 of the action
   range instead of 0.14.
4. **Same for P-3.3's centring reference.** Getting this wrong gave `psi = 11.8`
   instead of O(1); the design matrix was wrecked and β came out uncorrelated
   with truth while `fit_gain` still read 0.86.
5. **The oracle is an arm inside the layer, not a wrapper.** Computed outside the
   env it is one step stale, which understated the ceiling enough that PACT
   appeared to *beat* it (78.9 % vs 65.0 %) — not a possible result.

**The estimator**

6. **Pair `y(t−1)` with `psi(t−1)`**, not `psi(t)`. The wrong pairing regresses
   the target on a near-independent row.
7. **Bound covariance windup.** Without it a real run logged 5.7 M non-finite
   predictions out of 12 M agent-steps and the applied correction was 4 % of the
   disturbance. PACT was not beaten by the baseline; it never ran.
8. **P-4.2's liveness must test the *channel* columns.** `psi` carries an
   intercept of exactly 1, so `psi.abs().sum() > 0` is true for every row ever
   built and the dead-row skip silently never fires.
9. **μ is not inheritable.** URB's 0.999 is a 1000-step memory; if your driver
   cycles every 100 steps it averages the drift away. Sweep it.
10. **Select μ on prediction error, never on return.** A *diverged* estimator
    scored better than a correct one (96.2 % vs 84.7 % of B0) because a bounded
    but arbitrary correction is a perturbation some controllers tolerate.
11. **Bound the correction** (`corr_clip`), which is what closes that route.

**Reporting**

12. Log `fit_gain` **and** β-against-truth. High `fit_gain` with `beta_cos ≈ 0`
    means it predicts without identifying — claim compensation, **not**
    per-channel decomposition. Without the β column you will claim the wrong one.
13. If σ=1 is not anchored to a published constant, say so. A *stated calibration
    procedure* is honest; presenting it as an anchor is not.

**Engineering**

14. **Assert every scripted edit applied.** Two silent `str.replace` skips here:
    one left a task enum listed but never imported (`NameError` on first launch),
    one left the debug-CSV path unset inside the launcher meant to set it.
15. Write a `check_plumbing.py`: yaml keys ↔ dataclass fields ↔ the kwargs the
    layer pops, all three directions, plus every name in the task registry
    actually imported. torch only, ~1 s, run before queueing anything.

---

## Order of work

```
check_plumbing → conformance (offline, torch only)
              → smoke (in-simulator: σ=0 bit-identical, placebo inert,
                       trust=0 == blind, N=1 exactly zero, channel invertible)
              → calibrate σ against a competent controller
              → B0 training run
              → the blind/pact pair
```

Do not start the sweep on a red gate. Each one corresponds to a requirement whose
violation produces *plausible* numbers.

---

## Done means

- `conformance.py` — offline, torch only, includes the (B)-vs-(C) and (A)-vs-(C)
  decision procedures as **measurements**.
- `smoke.py` — in-simulator, every identity the spec gates.
- `calibrate.py` — the σ ladder with blind / oracle / PACT columns.
- `check_plumbing.py` — config consistency.
- One debug CSV row per iteration carrying the whole II.10 panel plus the domain
  metric, ordered so the first column that answers "no" is the one to fix.
- The task yaml carries the story, the committed operating point with the table
  it was chosen from, and every declared constant with *why*.
- A README stating plainly which cell this is, what is **not** implemented, and
  what that costs the claim.
