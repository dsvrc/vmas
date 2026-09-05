# Shared Link Contention — the NS, and PACT on top of it

`vmas_slc` plants the category-C non-stationarity of `NS_FORM_SPEC` into VMAS in
the form most natural to a robot swarm, and `pact2` implements the compensator
of `PACT_PIPELINE_SPEC` against it.

Everything runs through the normal BenchMARL entry point:

```bash
python pact2/run.py algorithm=ippo task=vmas_slc/sampling experiment.render=false
```

> **Use `pact2/run.py`, not `benchmarl/run.py`.** `python benchmarl/run.py` puts
> `benchmarl/` on `sys.path[0]` rather than the repo root, so `import benchmarl`
> resolves to whatever is installed in site-packages while hydra reads yaml from
> *this* checkout. When those differ, the task yaml is found but its ConfigStore
> schema is not, and hydra fails with `Could not load
> 'task/vmas_slc_sampling_config'` — an import-path bug that reads like a config
> bug. `pact2/run.py` pins the repo root and asserts the two agree before starting.

### Nothing in the installed `vmas` is modified

`VmasSlcClass.get_env_fun` hands `VmasEnv` a scenario **instance**, not a name:

```python
VmasEnv(scenario=make_slc_scenario("sampling", pact=True), ...)
```

`vmas.make_env` only falls back to `scenarios.load(<name>.py)` when given a
string, so passing an object bypasses the package lookup entirely. Two
consequences worth having: upgrading `vmas` cannot silently revert the
non-stationarity, and `vmas/sampling` and `vmas_slc/sampling` coexist — which is
what lets `smoke_test.py` diff them step for step.

---

## 1. The non-stationarity

### 1.1 The story

A VMAS world is a **robot swarm**, and every real swarm shares one finite medium
that no scenario models: the **radio link**. Each robot's controller runs on
state that arrives over a channel it shares with the fleet — telemetry to the
supervisor, ranging to the localization anchors. A robot that moves faster must
report its pose more often to keep the fleet's dead-reckoning error bounded
(event-triggered communication), so **motion is airtime**. When the channel is
congested the loop gets its setpoints through only in the airtime the channel is
idle, and a sampled-data loop's achievable gain scales with its update rate — the
robot delivers less force than it commanded.

An exogenous emitter in the facility (the site's own WLAN, a neighbouring cell)
cycles slowly over the operating day and raises the noise floor, which **shrinks
channel capacity**. Because loading is a *ratio* of demanded airtime to available
capacity, shrinking the denominator multiplies every agent's contribution to
every other agent's loading. Reward is untouched; the fleet achieves less because
the medium delivers less.

Every piece of that is ordinary to anyone who has deployed multiple robots.

### 1.2 The equations

```
Phi_j(t)  = phi_floor + phi_slope * ||v_j(t)||      exertion: airtime demanded
L_k(t)    = sum_j E[k,j] Phi_j(t) + Lfix_k          channel load
K_k(t)    = K0_k * g(A(t))                          capacity, shrunk by the driver
u_i(t)    = max_{k in links(i)} L_k(t) / K_k(t)     loading -- the BINDING link
f_exec_i  = (1 - harm_gain * u_i(t)) * f_cmd_i      the harm; reward untouched
```

with `W[i,j] = mean_{k in links(i)} E[k,j]` for `j != i` and `W[i,i] = 0`.

### 1.3 The four required objects (`NS_FORM_SPEC` A.2)

| object | SLC |
|---|---|
| medium | the deployment's radio channels |
| `L_k` | airtime demanded on channel *k* |
| `K_k` | channel capacity, `K0 * g(A)` |
| `W` | channel plan × adjacent-channel overlap × per-robot duty |
| driver `A(t)` | an exogenous emitter's slow duty cycle, off a global clock |
| `g(A)` | Shannon capacity ratio at the raised noise floor, clipped at 1 |
| harm | `(1 - c u) * f`, multiplicative — a **gain**, so heading is preserved |
| inverse | `a / (1 - c u)`, exact until the rail |
| sensor | the agent's own `u_i`, **one step stale** |
| `Phi` | `phi_floor + phi_slope*||v||` — a magnitude, uncancellable |
| loop? | **yes** by default; `slc_phi_reads_executed=false` is the contrast arm |

### 1.4 Three design decisions, and why

**`W` is position-independent.** A distance-based interference operator looks
more physical but hands each agent local authority over its own loading — move
away from the crowd — and that collapses the coordination gap, the exact failure
Part C warns about. Deployment channel plans genuinely are fixed at install time.
The operator is therefore static, which also means PACT's "computed once at
construction" holds verbatim.

**`aclr = 0.25`, not −30 dB.** A clean orthogonal channel plan would leave each
agent coupled to one co-channel peer and nobody else. `0.25` is the ordinary
*partially overlapping* 2.4 GHz picture — the unplanned channel map every
warehouse actually has — and it is what gives `W` five live peers spanning 16×
before the duty spread. Measured spread `std/mean = 0.87` against POWER's 1.35,
asymmetry 0.55.

**`harm_gain` is pinned to exactly 1.0 by an anchor, not tuned.** A control loop
gets its setpoints through only in the channel's idle fraction, so the delivered
loop gain *is* `1 - u`. Setting `slc_harm_at_nominal = slc_u_nominal` says that
and nothing more. The single remaining knob, `slc_u_nominal`, is chosen by
measurement in `calibrate.py` §0.

### 1.5 What differs from the POWER instantiation — state these

1. **`sigma = 0` is not stock VMAS.** Contention exists at every severity; the
   dial controls how the *capacity* moves. `g == 1` exactly at `sigma = 0` over
   the whole driver domain (verified bit-exactly), which is what B.1.1 actually
   requires. Stock VMAS is recovered byte for byte by `slc_harm_enabled=false`,
   and that is the separate `b0` arm.
2. **The coordination gap is essentially flat in `sigma`** (64.1% → 63.8% →
   63.5% at σ = 0.5/1.0/1.5), where POWER's grew 6.3 → 9.5 → 12.7. This is
   structural, not a defect: `1/g` is a *common factor* on `u_i`, so it scales
   every contributor equally and cannot shift their shares. The knobs that widen
   the gap here are **N** and the **frequency plan**, and both behave as C.4
   predicts. It also makes the two experimental axes orthogonal — σ sets
   difficulty, N sets the coordination share.
3. **N=1 is not byte-identical to the stationary task**, only *coordination-free*:
   the agent still feels the dial through its own term. The certificate is
   `Delta_peer / Delta_total == 0` exactly, computed from the operator, which is
   what A.3 claims.

---

## 2. The measured ceiling — run this first

```bash
python pact2/ceiling.py            # ~40 s, torch only, no simulator, no training
```

Part C decomposes each agent's excess loading `Delta_i = u_i (1-g)` by **who can
move it**. At the shipped defaults (N=6, 3 channels, σ=1):

```
             irreducible      own (free)      PEER (coordination)
SLC              14.2%           22.0%              63.8%
POWER            13.6%           76.9%               9.5%
```

**The coordination gap is 6.7× POWER's**, and the reason is structural rather
than tuned: contention on a shared channel is caused by other people's traffic,
whereas grid loading is dominated by the agent's own injection.

C.4's falsifiable prediction — the gap grows with N — measured with no training:

```
N          1       3       6       9      12
PEER    0.0%   49.0%   63.8%   71.8%   75.1%
```

and from the other side, more channels means fewer co-channel peers and a smaller
gap (`n_chan` 2 → 6 takes it 71.3% → 54.2%). No competing credit-assignment
method predicts either direction.

---

## 3. Which VMAS tasks, and why

`sampling`, `discovery` and `navigation`, chosen on the criteria that actually
bind:

| | `sampling` | `discovery` | `navigation` |
|---|---|---|---|
| `Phi` uncancellable | **best** — no "arrived" state, coverage needs sustained motion | **best** — `targets_respawn` keeps it alive | weakest — `Phi` decays once agents park on goals |
| pre-existing coupling | almost none with `shared_rew=False` → **cleanest attribution** | strong: reward needs *simultaneous* presence | collision avoidance only |
| N free / N=1 runnable | yes / yes | yes / yes | yes / yes |
| role | **headline** | **highest-gap arm** | familiarity + N-scaling |

`balance`, `transport` and `wheel` were considered and dropped as leads: their
mechanical coupling already dominates, which muddies attribution, and neither
runs at N=1.

---

## 4. PACT

### 4.1 What each agent knows

**Knows:** its own loading `u_i(t-1)` (the sensor, one step stale); the fleet's
broadcast exertion `Phi_j(t-1)`, one scalar per robot per step; its own current
exertion `Phi_i(t)`; the declared operator and `g(t)`, a function of observable
time.

**Does not know:** peers' *current* exertion `Phi_j(t)` — which is exactly what
the harm applied at step *t* is built from.

> **The coordination broadcast is paid for by every arm.** It is one float per
> robot per step, and it rides in `slc_phi_floor` — the heartbeat frame every
> robot already sends, which already loads the medium whether or not anybody
> listens to it. The irony that the medium is comms and the method needs comms is
> deliberate and it is charged for.

### 4.2 The control law

```
psi_now  = peer_basis(Phi(t-1))            declared basis, exact arithmetic
own_now  = own_basis(Phi_i(t))             CURRENT -- it knows its own velocity

RLS update on [1, own_col(t-1), psi(t-2)] -> y = u_i(t-1)     one-step-ahead
null model on [1, own_col(t-1)]           -> the mandatory lift baseline

ff   = u_prev * (g(t-1)/g(t) - 1)          LOCAL, closed form, no estimator
own  = beta_own * (own_now - own_prev)     LOCAL, exact
peer = trust * (ell_now - ell_prev)        COORDINATION  <- the claim
u_hat = u_prev + ff_gain*ff + own_gain*own + peer
delta = a / (1 - harm_gain*u_hat) - a      the channel inverse, railed
```

The regressor is **asymmetric in time**: own column current, peer columns
previous. Regressing `y(t)` on all of `psi(t-1)` would lag the own-gain column
and corrupt the coefficient the inverse depends on.

`mode="delta"` makes every term a correction to the *measured* stale loading, so
§6.4's pedestal problem disappears by construction rather than needing a slow
EMA. `mode="level"` implements §6.4 as written and is in the ablation table; it
measured **6.6× worse** loading error on the surrogate.

### 4.3 The floor property

When the gates say inadmissible, `trust` is exactly `0.0` and the executed action
is **byte-identical to the information-matched `ff` arm** — verified bit for bit
in `selfcheck.py` (250 steps) and again against the real simulator in
`smoke_test.py`. The floor that matters is *never worse than its own baseline*,
which is stronger than *never worse than blind*.

---

## 5. Phase 0 — what the surrogate settled, and what it did not

```bash
python pact2/selfcheck.py     # 21 arithmetic checks, ~90 s
python pact2/ceiling.py       # Part C
python pact2/calibrate.py     # the sweeps
```

**Settled:**

| question | measured |
|---|---|
| does the dial hurt (G3)? | blind reaches its waypoint **22.0% less often** than stock VMAS |
| does the `ff` arm need peers? | no — it recovers **72.5%** of that with local information alone |
| does the peer term add anything? | it cuts the residual loading error a further **12.6%** at N=6 |
| does that margin widen with N? | **yes: 12.6% at N=6, 17.4% at N=12** — the direction the ceiling predicts |
| `mu`? | **0.99**, not POWER's 0.9995. The peer signal here is velocity, which moves fast; the optimum follows the drift rate, exactly as the spec says to expect |
| `max_trust`? | interior optimum at **0.8** in loading error, degrading by 1.5 — this is the shipped default and the Phase-1 sweep's initialisation |
| is the inverse usable inside the action set? | at `u_nominal=0.30`: `sat_frac` 0.28, `delta_clip_frac` **0.00**. At 0.5 it is 0.71/0.34 and the correction becomes a constant bias |
| capacity cost (D.2) | σ=1 removes **10.9%** of mean capacity with a 1.687× swing; `slc_mean_preserve=true` leaves −2.1% and measurably outperforms |

**Not settled, and do not quote otherwise:**

- **The task metric is flat between `ff` and `pact` on the surrogate.** It runs a
  fixed high-gain servo, so a better loading estimate barely changes the command
  it issues. The *mechanism* is measurable here; whether it moves **return** is a
  learning question and needs the real environment.
- **The `max_trust` optimum above is an estimation optimum, not the T4 return
  inverted-U.** It is a defensible initialisation for the Phase-1 sweep, which
  still has to be run on the real env and validated on **held-out seeds**.
- **The A.6 contrast is not clean on the surrogate**: the two `Phi` definitions
  differ by a small level shift as well as by the loop. Read it on the real env,
  where the policy adapts to each.
- **The trace-gate and covariance-windup ablations are excitation-death
  failures.** They bite only once a policy has converged, and a fixed-controller
  surrogate would show them as no-ops and silently bless the bug. They are
  reproduced with excitation forced *off* in `selfcheck.py` (trace(P) reaches
  **7.5e20** without the bound) and must be re-read on the real training curve
  late in the run.

---

## 6. The arms

| arm | command | isolates |
|---|---|---|
| `b0` | `task.slc_harm_enabled=false` | stock VMAS, byte for byte |
| `blind` | *(defaults)* | the baseline under contention |
| **`ff`** | `pact_enabled=true pact_mode=ff` | **the information-matched baseline — everything PACT has except peers** |
| `pact` | `pact_enabled=true` | the method |
| `peer_only` | `pact_ff_gain=0 pact_own_gain=0` | the coordination term alone |
| `noloop` | `slc_phi_reads_executed=false` | A.6: T4 predicted **not** to apply |
| `placebo` | `slc_p_quiet=1.0` | G6 — must be byte-identical across σ |
| `n1` | `n_agents=1` | G1 — the coordination gap is exactly 0 |
| `level` | `pact_mode=level` | §6.4 as written |
| `trace` | `pact_gate=trace` | reproduces the silent-disarm failure |
| `nowindup` | `pact_p_max_mult=1e30` | reproduces the Q4 windup collapse |
| `sweep-trust` | | the T4 inverted-U — **evidence, not tuning** |
| `sweep-n` | | the C.4 N-scaling prediction, `pact` vs `ff` at each N |

```bash
bash pact2/run_pipeline.sh check       # plumbing + arithmetic + Phase 0, no GPU
bash pact2/run_pipeline.sh smoke       # wiring, needs vmas
bash pact2/run_pipeline.sh certify     # the gate output -- COMMIT IT before methods
bash pact2/run_pipeline.sh arms
bash pact2/run_pipeline.sh sweep-n
```

---

## 7. Diagnostics — read these two first

`VmasSlcClass.log_info` runs on every collection through the stock `run.py`.

| metric | question | healthy |
|---|---|---|
| `pact/applied_trust` | **was the method ON AT ALL?** | not ~0 |
| `pact/delta_nonzero_frac` | is it acting? | high |
| `pact/delta_clip_frac` | rail-pinned = constant bias, not compensation | ~0 |
| `pact/base_abs`,`ff_abs`,`own_abs`,`peer_abs` | the four-way local/coordination split | report **all four** |
| `pact/fit_gain_now` | do the peer channels beat the null **now**? | positive, rising |
| `pact/u_err_abs` | the mechanism measurement: `ff` vs `pact` | pact lower |
| `pact/trP`, `clamp_frac` | covariance windup | flat; rising late is fine |
| `pact/state`, `frac_alive` | INERT / ASLEEP / ALIVE | ALIVE |
| `slc/dial_ratio`, `dial_skip` | is the severity live? | `<1` / `~0` |
| `slc/phi_std_over_mean` | A.5's counter-check | `> 0.05` (measured 0.21) |
| `slc/sat_frac` | correction lost to the action rail | low |

`log_info` raises `SlcDialError` if `slc_severity > 0` but the dial read exactly
1.0 on essentially every step with quiet shifts ruled out. A silently discarded
dial produced five rows of pure scenario noise on POWER and *nothing looked
wrong*; this is a wiring bug, not something to tune, so the run stops.

---

## 8. File map

| path | what it is |
|---|---|
| `benchmarl/environments/vmas_slc/slc_core.py` | the NS arithmetic, **torch only** |
| `benchmarl/environments/vmas_slc/pact_core.py` | the compensator, **torch only** |
| `benchmarl/environments/vmas_slc/scenario.py` | `SlcMixin` → `PactMixin`, over the stock scenario |
| `benchmarl/environments/vmas_slc/common.py` | BenchMARL wiring + diagnostics + the dial gate |
| `benchmarl/conf/task/vmas_slc/*.yaml` | every knob, with the reasoning beside it |
| `pact2/selfcheck.py` | 21 arithmetic checks, no simulator |
| `pact2/ceiling.py` | Part C, no simulator |
| `pact2/calibrate.py` | Phase 0 sweeps, no simulator |
| `pact2/check_plumbing.py` | every config key reaches its dataclass field |
| `pact2/smoke_test.py` | 13 wiring checks against the real simulator |
| `test/test_slc_pact.py` | the invariants, under pytest or standalone |

The environment and the method both import from `slc_core`, so they cannot drift
apart. Both core modules are torch-only by design, which is why `vmas_slc/__init__`
exposes the torchrl-dependent wiring lazily.

---

## 9. Honest limits — state all of these

1. **The compensator is classical.** Project onto a known basis, run RLS, invert.
   An adaptive-control reviewer will say so and be right about the *mechanism*.
   Lead with the reduction that makes it applicable at N agents, the commons
   theory, and the recoverable-fraction ceiling — not with the estimator.
2. **The operator is declared.** Not privileged — every operator has a channel
   plan — but the `ff` arm is **mandatory**, or the gap is information rather
   than mechanism.
3. **The correction is dominated by the agent's own stale sensor reading**, which
   is local. The `ff` arm gets that too, so the ablation is clean; but quote the
   four-way split, never `delta_abs` alone.
4. **First-order.** The sampled-data gain loss is a linearisation; second-order
   contention effects (backoff dynamics, capture) are not modelled.
5. **Trust is not learned**, only well-initialised and gated.
6. **`theta` may be predictable without being decomposable.** Measured
   `cond = 98.5`; report it before claiming to identify anything.
7. **Excitation dies** in converging domains. `cond` rising over training is the
   default, not an edge case — and it is what arms the two failure modes the
   surrogate cannot reproduce.
