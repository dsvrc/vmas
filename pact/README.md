# Navigation-PCW — a category-C non-stationarity for VMAS, and PACT on top of it

`vmas_ns/navigation_pcw` is stock VMAS `navigation` under a **propwash–circulation
wake (PCW)** non-stationarity, plus a complete implementation of the PACT
pipeline (Phase 1 certification, Phase 2 method, ceiling and ablation arms).

Everything runs through the normal BenchMARL entry point:

```bash
python benchmarl/run.py algorithm=ippo task=vmas_ns/navigation_pcw experiment.render=false
```

---

## 1. The non-stationarity

### 1.1 The story

*N* holonomic vehicles operate in a **confined arena** — an indoor flight cage, a
test tank. Their thrusters do not push against nothing: every command feeds
momentum into the enclosed medium, and because the volume is closed that
momentum cannot escape. It accumulates as a slowly-decaying **bulk circulation**
of the whole arena. A hull sitting in that circulation is dragged around its yaw
axis; VMAS agents have no yaw actuation and no heading reference (indoors and
underwater a magnetometer is unusable — pose comes from motion capture or an
acoustic beacon, but the **actuator frame does not**). Thrust is produced in the
body frame, so a yawed vehicle pushes in the wrong direction and does not know it.

How hard a given circulation drags on a hull is set by the **coupling strength of
the medium** — density, stratification, extraction state — which cycles slowly
over the operating day. That cycle is the exogenous driver `A(t)`.

A domain expert recognises this immediately: uncompensated heading drift is
exactly what ruins indoor multirotor flight, and rotor-wash interaction in a
confined volume is a real and well-documented nuisance.

### 1.2 The equations

```
m_j(t)     = ( p_j(t) × u_j(t) )_z                    # j's angular impulse: ONE scalar
Φ_i(t)     = mean_{j ≠ i} m_j(t)                      # OTHERS only  ← category-C signature
x2_i(t+1)  = ρ·x2_i(t) + (1−ρ)·G·Φ_i(t)               # bulk circulation, driver-free
c(t)       = A(t)·σ                                   # exogenous driver × severity
θ_i(t)     = c(t) · x2_i(t)                           # yaw offset  (the liability)

delivered_i(t) = R(θ_i(t)) · u_i(t)                   # harm lives in the TRANSITION
reward         = the original navigation reward, byte for byte
```

Note the factorisation `θ = c · x2`: the *circulation* accumulates independently
of the medium's coupling strength, and the coupling strength only scales the
force it exerts. That is both better physics and what makes theorem T1 **exact**
for any driver path rather than exact to `O((1−ρ)|dc/dt|·window)` — the per-step
cosine gate reads 1.0, not 0.999-something.

### 1.3 The guide's hard constraints, and where each is enforced

| # | Constraint | How this design satisfies it | Enforced at |
|---|---|---|---|
| 1 | Natural / reviewer-plausible | Confined-volume rotor wash spinning up the medium; unobservable yaw drift with no magnetometer. Both are real failure modes. | — |
| 2 | Strictly category C | `A(t)` **multiplies** `Σ_{j≠i}` and is never an additive term. At `N=1` the sum is empty ⇒ `x2 ≡ 0` ⇒ `θ ≡ 0` **for any driver value**. | [`pcw_core.peer_mean`](../benchmarl/environments/vmas_ns/pcw_core.py) — one function, one early return |
| 3 | Dynamics-only, never reward shaping | The scenario subclasses VMAS `navigation` and **does not override `reward`, `observation` or `done`**. Only `process_action` (the transition) and `info` (privileged read-out) are touched. | [`scenario.py`](../benchmarl/environments/vmas_ns/scenario.py) — by inheritance |
| 4 | Guaranteed performance collapse | Blind falls to 17% of B0 at the default severity (peak). | §2 |
| 5 | One unifying idea across environments | Hidden liability ← leaky accumulator ← others' exertion × exogenous driver; throttles effectiveness inside the transition. | — |
| 6 | Minimal knobs | Exactly one severity dial, `ns_severity`. `ns_gain`, `ns_rho`, `ns_driver_period`, `ns_phase_spread` are fixed structural constants; `ns_freeze_driver` is a Phase-1 instrument. | [`navigation_pcw.yaml`](../benchmarl/conf/task/vmas_ns/navigation_pcw.yaml) |
| 7 | Information-recoverable | The channel is a **rotation** — norm-preserving and exactly invertible. Knowing `θ` *is* the entire solution; the compensation costs no thrust. Nothing is physically destroyed, so nothing is unrecoverable. | [`pcw_core.channel_inverse`](../benchmarl/environments/vmas_ns/pcw_core.py) |

**The two litmus tests.**

- *Not (A) — learning-induced?* The driver runs off a global step counter that is
  never reset by an episode boundary and never reads any agent. Freeze every
  teammate at an optimal policy and the difficulty still drifts.
- *Not (B) — exogenous-reducible?* Set `n_agents=1`. The effect does not shrink;
  it **vanishes**, and the environment is byte-identical to stock VMAS
  navigation. Verified numerically (`|θ| max = 0.000000`) and by unit test
  (`test_n1_reduces_the_environment_exactly`, `test_peer_mean_is_identically_zero_at_n1`).

**Tragedy of the commons.** Thrust earns reward and thrust is exactly what stirs
the medium — but no agent appears in its own exertion sum, so charging the
medium only ever hurts *other* agents. With `shared_rew: False` the two
degenerate escapes (fly radially so you inject no circulation; exert less) are
public goods that cost the individual and benefit the team, so independent
learners have no gradient toward them. That is the intended structure, and it is
why `shared_rew` must stay `False`.

### 1.4 Why a *periodic* driver, and why this shape

A periodic driver gives collapse-**and-recover**: the team must re-coordinate in
both directions, and the compensation gain must be driven *down* as well as up.
That is what makes the trough the hard part of the method (§5.2), and it is a
sharper research problem than a monotonic ramp. The shape is a smooth raised
cosine, which suits a thermal/diurnal cycle; period 2000 global steps ≈ 33
episodes, so `c` is near-constant *within* an episode but drifts *across* them —
exactly the regime PACT is designed for.

`ns_phase_spread: True` gives each vectorised world its own fixed offset in the
cycle: a fleet operating around the clock. Every collection batch and every
evaluation then spans the full cycle, which makes metrics cycle-averages with low
variance and independent of *when* the evaluation happened. Set it `False` for
the single-global-phase ablation, where the whole population collapses and
recovers together and the training curve oscillates visibly.

---

## 2. Calibration (Phase 0) — and the two design decisions it forced

Guide pitfall #4 says to size severity against the policy's *measured* operating
scale, not an assumed one. [`pact/calibrate.py`](calibrate.py) mirrors VMAS
navigation's dynamics exactly (double integrator, per-step drag 0.25, two
substeps, `dt=0.1`, unit mass, action box `[-1,1]²`) and drives it with a
**saturating velocity servo** — deliberately a *strong* stand-in for a blind
policy, since high-gain velocity feedback is the controller class most able to
reject a rotational actuator disturbance. Results are therefore optimistic for
blind, i.e. conservative for the design.

```bash
python pact/calibrate.py            # ~1 min, torch only, no simulator needed
```

Measured operating scale at the shipped constants (`G=20`, `ρ=0.8`, `N=3`,
horizon 60): `|Φ| rms = 0.195`, `|x2| rms = 3.48 rad per unit c`.

| σ | blind (% B0) | scripted β=c (% B0) | saturation | mean \|θ\| | \|θ\| > 90° |
|---|---|---|---|---|---|
| 0.2 | 88.9 | 100.0 | 2.5% | 0.66 rad | 12% |
| 0.4 | 33.2 | 100.0 | 2.4% | 1.43 rad | 44% |
| 0.6 | 22.4 | 100.0 | 2.5% | 1.53 rad | 49% |
| **0.8** | **17.4** | **100.0** | 2.4% | 1.55 rad | 49% |
| 1.0 | 12.3 | 100.0 | 2.4% | 1.56 rad | 50% |

B0 = 1.0295. Blind clears the "≤ 30%" requirement from σ ≥ 0.6; scripted
compensation with the true gain holds 100% everywhere.

**Decision 1 — `max_steps: 60`, not the stock 100.** At the stock horizon the
task carries ~3× travel slack, and a blind team can recover to **34.8%** of B0
simply by moving slowly: less thrust, less circulation, still arrives in time.
That escape needs no knowledge of the driver, so it would defeat *any*
exertion-driven non-stationarity and break guide condition 7. At 60 steps the
best escape is **23.0%**, while B0 barely moves (1.030 vs 1.046) because agents
still comfortably reach their goals. This is the one place the base task is
altered. Reproduce with:

```bash
python pact/calibrate.py --severities 0.0 0.8 --horizons 100 80 60 50 \
       --speed-scales 1.0 0.7 0.5 0.35 0.25 0.15
```

|  horizon | B0 | best blind over speed scales (% B0) |
|---|---|---|
| 100 (stock) | 1.046 | **34.8** ← escapes |
| 80 | 1.046 | 28.8 |
| **60** | **1.030** | **23.0** |
| 50 | 0.984 | 20.6 |

**Decision 2 — `shared_rew: False` (the stock default) is load-bearing**, per §1.3.

**The frontier.** For a norm-preserving *transform* channel the actuator is
almost never the binding constraint — saturation stays at ~2.5% at every severity
and the scripted controller never loses. What binds instead is the **conditioning
of the inverse**: how wrong `β` may be and still clear the bar.

| σ | β=0 (blind) | 0.5·c | 0.8·c | 1.0·c | 1.2·c | 1.5·c |
|---|---|---|---|---|---|---|
| 0.6 | 22.4 | 93.4 | 100.0 | 100.0 | 100.0 | 99.6 |
| 0.8 | 17.4 | 77.3 | 100.0 | 100.0 | 100.0 | 97.8 |
| 1.0 | 12.3 | 54.9 | 100.1 | 100.0 | 100.0 | 91.2 |

So `σ* ≥ 1.0` on both readings, and `ns_severity: 0.8` sits below it with margin.
`pact/phase1_certify.py` reports **both** definitions against a real trained
policy and states which one binds.

---

## 3. The three per-environment declarations

| # | Declaration | Navigation-PCW |
|---|---|---|
| 1 | **Exertion `Φ`** | `mean_{j≠i} (p_j × u_j)_z` — the angular impulse each peer feeds the medium. The PACT message is literally **one scalar per agent per step**, computed by each agent from its own position and its own executed command. |
| 2 | **Leak / coupling** | Leaky accumulator, `ρ = 0.8` (≈ 5-step eddy memory), global (well-mixed medium), gain `G = 20` rad per unit circulation. |
| 3 | **Harm channel `g` + inverse** | Transform: `delivered = R(θ)·u`. Inverse: `u = R(−β·x2)·a`, clipped to the action box. Bounded resource: the action box corners (rarely binding — see §2). |

---

## 4. File map

| Path | What it is |
|---|---|
| `benchmarl/environments/vmas_ns/pcw_core.py` | The arithmetic, **torch only** — no torchrl, no vmas. Both the environment and the method import from here, so they cannot drift apart. |
| `benchmarl/environments/vmas_ns/scenario.py` | The VMAS scenario. Subclasses `navigation`; overrides only `make_world`, `reset_world_at`, `process_action`, `info`. |
| `benchmarl/environments/vmas_ns/pact.py` | `PactTransform` (the method), `Phase1ProbeTransform` (the scripted probe), `CtdePayloadTransform` (critic-only driver). |
| `benchmarl/environments/vmas_ns/common.py` | `VmasNsTask` / `VmasNsClass`: env construction, transform wiring, and all diagnostics + the hard gate via `log_info`. |
| `benchmarl/conf/task/vmas_ns/navigation_pcw.yaml` | Every knob, with the calibration behind each. |
| `benchmarl/algorithms/mappo_ctde.py` | MAPPO whose **critic only** sees the true driver. |
| `pact/calibrate.py` | Phase 0: the simulator-free calibration battery above. |
| `pact/smoke_test.py` | 9 integration checks against the real simulator — run once before spending training compute. |
| `pact/phase1_certify.py` | Phase 1: the σ* sweep with both confounding checks. |
| `pact/evaluate_arms.py` | Phase 2: the phase-profile / peak / cycle-average arm table. |
| `pact/rollout_utils.py` | Shared loading + rollout accounting for the two scripts. |
| `test/test_pact_pcw.py` | 25 tests of the arithmetic in pure torch — no simulator. |

---

## 5. Running the pipeline

### 5.0 Sanity, before any GPU time

```bash
python test/test_pact_pcw.py     # 25 arithmetic tests, torch only, no simulator
python pact/calibrate.py         # reproduces the tables in §2, ~1 min
python pact/smoke_test.py        # 9 integration checks, needs vmas + torchrl
```

`smoke_test.py` is the one to run on the training machine. It asserts the claims
the unit tests cannot reach because they are about wiring rather than arithmetic:

1. `ns_severity=0` reproduces stock `vmas/navigation` step for step — the
   dynamics-only / reward-untouched constraint, verified rather than asserted;
2. `n_agents=1` at severity 5.0 is byte-identical to severity 0 — the
   irreducibility certificate;
3. the driver ignores what the agents do and is *not* rewound by an episode reset;
4. the NS actually fires (i.e. this VMAS build really does route through
   `BaseScenario.process_action` — the one load-bearing integration assumption);
5. PACT widens the interface by exactly one action dim and three observation features;
6. the per-step cosine gate reads 1.0 while compensation is live;
7. `β = 0` reproduces the blind environment bit for bit;
8. a partial reset clears exactly the worlds that reset, on both sides;
9. the CTDE payload reaches the critic and is absent from the actor's spec.

### 5.1 Phase 1 — certify

```bash
# B0: the NS off.  Severity-independent, so this ONE checkpoint serves the whole sweep.
python benchmarl/run.py algorithm=ippo task=vmas_ns/navigation_pcw \
  task.ns_severity=0 experiment.render=false \
  experiment.checkpoint_at_end=true experiment.max_n_frames=3_000_000

python pact/phase1_certify.py <outputs/.../checkpoints/checkpoint_3000000.pt>
```

Phase 1 refuses to be believed until both confounding checks pass: the probe at
driver 0 must reproduce B0 exactly (it does so *by construction* — `R(0)` is the
identity bit for bit), and compensation must actually recover at low severity.
It re-optimises β **per severity**, which is the single most common way to
understate σ*.

### 5.2 Phase 2 — the ladder

```bash
COMMON="task=vmas_ns/navigation_pcw experiment.render=false experiment.checkpoint_at_end=true"

# reference (upper bound, not the target)
python benchmarl/run.py algorithm=ippo  $COMMON task.ns_severity=0

# blind baselines
python benchmarl/run.py algorithm=ippo  $COMMON
python benchmarl/run.py algorithm=mappo $COMMON
python benchmarl/run.py algorithm=ippo  $COMMON model=layers/gru      # memory alone does not fix it

# PACT, fully decentralised (no privileged info anywhere)
python benchmarl/run.py algorithm=ippo $COMMON task.pact_enabled=true model=layers/gru

# PACT + CTDE critic (driver in the critic only; execution stays decentralised)
python benchmarl/run.py algorithm=mappo_ctde $COMMON task.pact_enabled=true model=layers/gru

# O1: the compensation ceiling (privileged at execution time — a reference, not a method)
python benchmarl/run.py algorithm=ippo $COMMON \
  task.pact_enabled=true task.pact_oracle=true model=layers/gru

# irreducibility certificate: must equal the stationary task exactly
python benchmarl/run.py algorithm=ippo $COMMON task.n_agents=1
python benchmarl/run.py algorithm=ippo $COMMON task.n_agents=1 task.ns_severity=0
```

`model=layers/gru` is **required** for the PACT arms: a memoryless policy cannot
sense the hidden phase and leaves β at a constant compromise. The host algorithm
and all its hyperparameters are otherwise untouched.

### 5.3 The report

```bash
python pact/evaluate_arms.py \
  --b0  runs/b0/checkpoints/checkpoint_3000000.pt \
  --arm blind_ippo=runs/blind_ippo/checkpoints/checkpoint_3000000.pt \
  --arm blind_mappo=runs/blind_mappo/checkpoints/checkpoint_3000000.pt \
  --arm pact=runs/pact/checkpoints/checkpoint_3000000.pt \
  --arm pact_ctde=runs/pact_ctde/checkpoints/checkpoint_3000000.pt \
  --arm ceiling=runs/oracle/checkpoints/checkpoint_3000000.pt
```

---

## 6. Metric protocol: report PEAK **and** CYCLE

A single scalar hides the point of a periodic driver. In the trough the NS is
switched *off* (`A → 0`), so every arm scores ~100% there and a cycle-average
understates every gap. The primary number is therefore the **peak** (`A = 1`) —
the phase σ* is defined at, and the phase a controller is judged by — with the
cycle-average reported alongside. `evaluate_arms.py` prints the full phase
profile plus both summaries so neither can be cherry-picked.

---

## 7. Diagnostics — what appears in wandb/csv automatically

`VmasNsClass.log_info` runs on every collection through the stock `run.py`; no
callback wiring is needed.

| Metric | Read it for |
|---|---|
| `ns/abs_theta_{mean,p95,max}`, `ns/abs_x2_rms` | **Calibration.** These are the numbers that say whether `ns_gain` is sized to the policy's real operating scale. If `abs_theta_mean` is far from ~1.5 rad at σ=0.8, recalibrate. |
| `pact/gate_cosine_{mean,min}`, `pact/gate_rel_err_max` | **The one hard gate.** Per-step cosine between the computed waveform and the environment's accumulator. Exact arithmetic ⇒ a value below `pact_gate_tol` aborts the run with `PactGateError`, because it is a wiring bug, not something to tune. |
| `pact/beta_{peak,trough}`, `pact/c_{peak,trough}`, `pact/beta_abs_err_mean` | **The ballgame.** Success is `β_peak → c` and `β_trough → 0`. A β that is flat across phases is phase-blind and wants recurrence / the CTDE critic. |
| `pact/residual_abs_mean` vs `pact/uncompensated_abs_mean` | How much of the disturbance is actually being cancelled. |
| `pact/sat_frac` | The bounded resource. Expect ~2%; a sharp rise means the action box has become the binding constraint. |

The gate is a **per-step cosine over the agent vector**, deliberately not a
correlation pooled across the driver's range — a pooled correlation reads ~0.95
off the varying-`c` fan even when every point is exact, which is how a correct
pipeline gets misdiagnosed as broken. `test_per_step_cosine_beats_a_pooled_correlation`
demonstrates exactly that.

### Why PACT cannot crater below blind

With `β = 0` the compensation is `R(0)·a = a` bit for bit — the blind policy.
There is no estimator anywhere in the control path: the waveform is *computed*
from shared messages by the same recursion the environment runs, and the only
learned quantity is one bounded scalar multiplying it. An evaluation below blind
means a wiring or normalisation bug, not a failure of the method.

---

## 8. What the surrogate predicts, and what still has to be measured

Predicted (Phase 0 surrogate, peak-frozen, N=3, σ=0.8):

| Arm | % of B0 |
|---|---|
| stationary reference (σ=0) | 100 |
| O1 compensation ceiling | ~100 |
| blind (strong hand-tuned servo) | 17 |
| blind, best "exert less" escape | 18–23 |

The surrogate settles the *environment* questions — operating scale, collapse
depth, the escape routes, the σ* frontier, the N=1 certificate. It cannot settle
the *learning* questions, and these are the ones to watch on the server:

1. **Does trained blind IPPO/MAPPO actually land near 17%?** The surrogate's
   servo is stronger than a trained MLP at rejecting a rotation, so trained blind
   should be *at or below* this. If it comes in much higher, check
   `ns/abs_theta_mean` first — the most likely cause is that the learned policy
   thrusts less than the servo, so `ns_gain` needs raising.
2. **Does β track the phase?** This is the open research question, not a
   configuration detail. Expect the ladder the PACT reference describes: a good
   *constant* β is an easy robust win worth roughly half the gap; full phase
   tracking is the frontier and is what the recurrent policy and the CTDE critic
   are for. Read `pact/beta_peak` vs `pact/beta_trough`.
3. **Local unobservability** is a structural property of the class, not a tuning
   failure: at `β = c` the residual vanishes, so `c` becomes unobservable exactly
   when it is being tracked well. It bounds the last stretch of β-tracking.

Report both tiers honestly: fully-decentralised PACT (no privileged information
anywhere) and PACT + CTDE critic (privileged driver in the critic only, training
only). The gap between decentralised PACT and the O1 ceiling *is* the
phase-tracking frontier for this environment.
