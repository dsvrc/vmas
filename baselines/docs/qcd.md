# QCD+ / RR — X1, prior-free black-box non-stationary RL

## Source

| what | where |
|---|---|
| paper | Gerogiannis, Huang, Veeravalli, *Is Prior-Free Black-Box Non-Stationary Reinforcement Learning Feasible?*, [arXiv 2410.13772](https://arxiv.org/abs/2410.13772) |
| detector | Besson, Kaufmann, Maillard, Seznec, *Efficient Change-Point Detection for Tackling Piecewise-Stationary Bandits*, JMLR 2022 — the Bernoulli GLR that 2410.13772's Algorithm 3 plugs in, with the threshold `beta(n, delta) = log(4 n sqrt(n) / delta)` that paper names |
| also cited | Wei & Luo, *Non-stationary Reinforcement Learning without Prior Knowledge* (MASTER), COLT 2021 — the algorithm 2410.13772 analyses |
| implementation | [`benchmarl/algorithms/qcd.py`](../../benchmarl/algorithms/qcd.py), [`benchmarl/conf/algorithm/qcd.yaml`](../../benchmarl/conf/algorithm/qcd.yaml), the `X1` block of [`_baseline_math.py`](../../benchmarl/algorithms/_baseline_math.py) |

The paper has no released code. It is three algorithms written out as
pseudocode (Algorithms 1, 2 and 3) plus one theorem, and all four are short
enough to transcribe exactly; the detector it plugs in is the one Besson et al.
published, and that is where the threshold comes from.

## What the paper actually says, in three steps

1. **MASTER** is the state of the art in *black-box* non-stationary RL: it
   assumes nothing about when or how often the environment changes.
2. **Theorem 4** says MASTER's two non-stationarity tests compare a quantity
   bounded by 1 against a threshold of order `54 (log2 T + 1) log(T/delta)`, so
   they cannot fire until `T >= 1.24e9` rounds. Below that — i.e. in every
   experiment anyone runs — MASTER *is* random restarting.
3. So the baseline worth running is the one that does fire: **quickest change
   detection plus a full restart**. In their 5-armed piecewise-stationary
   bandits, MASTER declares 0 changes on every problem and the QCD methods
   declare 8–150.

`verify.py` checks step 2 against the number the paper quotes: the smallest `T`
at which `sqrt(T) > 54 (log2 T + 1) log T` is **1.2463e9**, and at this repo's
horizon (a 3 M-frame run at 300 workers is 10 000 detector rounds) it cannot
fire by seven orders of magnitude. **That is why the arm that runs here is
QCD+ and not MASTER**, and the algorithm prints it at construction so the
reason is in the log rather than in a footnote.

## Why this baseline belongs in the table

Every other non-stationarity row here is *told* something. LCPO (B6) observes
the context. RMA (B8) is given a privileged teacher. DR (B9) is given the
severity range. PACT is given a declared basis. **QCD+ is told nothing.** It
watches the reward stream, decides the world changed, and throws the learner
away. If it wins, every method in this paper that models the disturbance is
priced against a method that models nothing.

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | a **black-box wrapper**: the base learner runs untouched between restarts | Alg. 2 and 3 both take "base bandit algorithm B" as input and never look inside it | `QcdLoss` is `ClipPPOLoss` with diagnostics added and **nothing** changed in the objective | **met** |
| 2 | Alg. 3 (QCD+): pull, observe `R_t`, run the detector on the history, restart on an alarm | `if D(H_B, A_t) = True: H_B <- {}` | `Qcd.observe_batch` feeds the per-step reward stream in time order; `apply_restart` performs the restart | **met** |
| 3 | the detector is a **Bernoulli GLR** | "Detector: Generalized Likelihood Ratio test on arm history" | `glr_statistic`: `max_{1<=s<n} [ s kl(mu_{1:s}, mu_{1:n}) + (n-s) kl(mu_{s+1:n}, mu_{1:n}) ]`, computed from prefix sums | **met, tested** (`verify.py`: fires on a step change, silent on a flat stream) |
| 4 | threshold `beta(n, delta) = log(4 n sqrt(n) / delta)` | quoted verbatim in the paper | `glr_threshold` | **met, tested** against its definition |
| 5 | `delta` of order `1/poly(T)` | "delta in (0,1) with order 1/poly(T)" | `delta: 0` resolves to `1/sqrt(T)` with `T` the number of detector samples in the run; the resolved value is printed | **met** |
| 6 | Alg. 2 (RR): restart at i.i.d. `Geometric(eta_r)` times | "Initialize: restart intervals ~ i.i.d. Geo(eta / sqrt(polylog T))" | `RandomRestartSchedule`, selected by `detector=random` | **met, tested** (mean gap is `1/eta` over 200 k rounds) |
| 7 | `eta_r` is set from the (unknown) change rate | `eta_r = sqrt(eta / poly(log T))` | **adapted**: there is no `eta` here to derive it from, so `rr_interval` is a declared sweep point and the yaml says so | **adapted** — see (A) |
| 8 | a restart is `H_B <- {}`: forget everything this instance learned | Alg. 2 and 3, both | the policy (and, by default, the critic) is restored to its **initial parameters** and every optimiser's Adam moments are cleared | **adapted** — see (B) |
| 9 | the restart takes effect before the next round | the loop restarts and then pulls | the restart runs in `on_train_end`, so the *next* collection uses the restarted weights | **met** — see (C) |
| 10 | Theorem 4: MASTER's tests cannot fire below `T ~ 1.24e9` | Thm 4 | `master_can_fire` / `master_min_horizon`, evaluated at this run's horizon and **printed at construction** | **met, tested** |
| 11 | MASTER itself (Alg. 1), as a running arm | Alg. 1 | **not implemented** | **NOT met** — see (D) |
| 12 | the detector is **per arm** | `D(H_B, A_t)` is applied per-arm after each pull | **adapted**: there are no arms; one detector per agent *group*, on the team reward | **adapted** — see (E) |
| 13 | observations in `[0, 1]` | a bandit's rewards are, by assumption | **adapted**: a declared affine map with a clip, and `qcd_clip_frac` reports how much saturated | **adapted** — see (F) |
| 14 | the experiments are 5-armed piecewise-stationary bandits | §5 | not reproduced; this is a VMAS row, and the bandit experiments are the paper's evidence for its own claims, not a baseline for ours | **out of scope, deliberate** |

### (A) `rr_interval`

The paper's order-optimality result fixes `eta_r` only up to the unknown change
rate `eta` (`eta_r = sqrt(eta / polylog(T))`, Theorem 11). On this instance the
"change rate" is not a well-defined quantity — the driver varies *continuously*
within an episode rather than jumping between pieces — so there is nothing to
substitute. `rr_interval = 1000` detector samples is a declared choice giving
~10 restarts over a 3 M-frame run at 300 workers, and it should be swept if the
row is close.

### (B) What a restart means for a parametric learner

`H_B <- {}` is unambiguous for a bandit: it is the empirical means. For a
policy network the same sentence has to be turned into something, and the
choice made here is **the parameter vector it started from, plus the optimiser
state**. Two notes:

* the optimiser state is not an afterthought. Leaving Adam's first and second
  moments in place would push the freshly re-initialised parameters along the
  *old* gradient for hundreds of steps, which is not a restart.
* restoring the *initial* parameters rather than drawing a *fresh* random
  initialisation is a reading, not a theorem. Both are "a new instance of the
  base learner"; this one is reproducible from a seed and needs no assumptions
  about how torchrl's stacked per-agent parameters are initialised.

`restart_critic: false` restarts only the policy, and exists to price which
half of the learner the restart is actually throwing away.

### (C) Why the restart happens *after* the iteration

The batch that triggered the alarm was collected by the pre-restart policy and
carries its log-probabilities. Restarting before training on it would leave
PPO's importance ratio comparing a freshly initialised network against a
trained one — an enormous, meaningless update. Restarting after means the batch
gets one ordinary PPO update and the *next* collection runs on the restarted
weights, which is exactly Algorithm 3's order.

### (D) MASTER is not run

Algorithm 1's scheduling machinery — a recursive tree of `ALG` instances
scheduled at every dyadic block length with probability `rho(2^n)/rho(2^m)` —
would be a substantial piece of work, and Theorem 4 says the result is already
known: **at any horizon this repo can reach, it never fires, and it reduces to
restarting at random with memory.** That reduction is what `detector=random`
runs. The theorem is checked arithmetically in `verify.py` and printed at
construction, so the claim is not taken on trust; what is not done is spending
a queue slot to watch a detector not fire.

### (E) One detector per group, on the team reward

There are no arms, so "per-arm detection" has no counterpart. What the detector
watches here is `R_t`, the reward averaged over the parallel worlds (which are
i.i.d. replicas of the same process) and over the agents (whose reward is the
team's on every `simple_ns` host), **one sample per environment step, in time
order**. That is the stream Algorithm 3 observes, at the environment's own time
scale — which matters, because the driver varies *within* an episode. A
detector fed one sample per training iteration would see learning progress and
almost nothing else.

### (F) The map onto `[0, 1]` — the one adaptation that can go wrong

The Bernoulli GLR is defined on a `[0, 1]`-valued stream. A bandit's rewards
already are; a VMAS per-step team reward is not. The stream is therefore
`(reward - reward_low) / (reward_high - reward_low)`, clipped, with
`reward_low` and `reward_high` **declared in the yaml and printed at
construction**.

This is the same class of problem as LCPO's `ood_threshold`, which had to be
rescaled off the paper's value because the paper's value was on a different
environment's scale and **could never fire here**. Watch `qcd_clip_frac`:

* near 0 → the range is too wide, the stream crowds the middle, and the
  detector is deaf;
* large → the range is too narrow, the stream saturates, and the detector is
  blind.

The defaults bracket the per-step team reward on `simple_ns/balance`. **Check
them for another host.**

## Running it

```bash
GROUP=x1 bash scripts/run_extra_baselines.sh      # qcd_glr, qcd_rr, qcd_none
ONLY=qcd_glr bash scripts/run_extra_baselines.sh
```

Three rows, differing by one flag, which is the point: `qcd_glr` is Algorithm
3, `qcd_rr` is Algorithm 2, and `qcd_none` is the same file with restarts off —
so any difference between `qcd_none` and `mappo_blind` is a bug in this
wrapper and not a result.

## What to watch

| column | what it means |
|---|---|
| `qcd_restarts` | how many times the learner was thrown away. **0 means the row is `mappo_blind` with extra steps**, and the detector needs a narrower `reward_low`/`reward_high` or a larger `delta`. |
| `qcd_stat` / `qcd_threshold` | the GLR statistic and the threshold it is being compared against. If the statistic never gets within an order of magnitude of the threshold, nothing is being detected. |
| `qcd_clip_frac` | the fraction of the reward stream that hit the clip. See (F). |
| `qcd_steps_since_restart` | how long the current instance has been learning. |

And read the construction banner: it prints the resolved `delta`, the number of
detector samples in the run, and whether MASTER's own tests could fire at that
horizon.
