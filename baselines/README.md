# The BASELINES.md baselines, on VMAS

Everything in `../paper/BASELINES.md` that can run on a VMAS host, implemented
against the paper **and** the authors' own code, one baseline per algorithm.

```bash
python baselines/verify.py          # ~2 s, torch only. Run this first.
python simple_ns/check_plumbing.py  # ~1 s, torch only.
LIST=1 bash scripts/run_baselines.sh   # print every launch line, run nothing

# SMOKE FIRST -- one iteration of every row, into a throwaway OUT_ROOT:
FRAMES=12000 BATCH=6000 ENVS=60 SEEDS=0 OUT_ROOT=runs/smoke \
  EXTRA="experiment.off_policy_n_optimizer_steps=20 experiment.evaluation=false" \
  KEEP_GOING=1 bash scripts/run_baselines.sh

bash scripts/run_baselines.sh          # then the real thing
```

`verify.py` parses every source file on **the interpreter you are running it
with**, before anything else, because the cluster's python is not the laptop's:
a backslash inside an f-string expression is Python 3.12 syntax and older
interpreters reject the file outright.

---

## 1. How each baseline was built

Every one of them went through the same four steps, and the result of step 2 is
the file you are looking for when you want to check the work:

1. **Get the authors' code.** Cloned from the repository BASELINES.md names, at
   the commit that was on `main` on 2026-09-19. What was read is listed under
   "Source" in each doc, by file and by function.
2. **Make a checklist from the paper and the code, before writing anything.**
   That is the table in each `docs/<name>.md`: one row per idea, with the
   reference's own line next to it.
3. **Implement it, row by row**, marking each row `met` / `adapted` / **`NOT
   met`**. Nothing is marked `met` unless the code does the thing the row says.
4. **Check what can be checked offline.** `baselines/verify.py` exercises the
   published formulas against their definitions — HAPPO's factor, the ESO's pole
   placement, LIAM's "everyone but me" index, the mean-field average, LCPO's
   conjugate gradients and its out-of-distribution test — plus every config↔code
   consistency check. It found a wrong observer gain in the first draft of the
   ESO arm; that is what it is for.

**Nothing here is a reimplementation "in spirit".** Where a published idea does
not fit this environment, it is either adapted with the adaptation named and
justified, or left out and marked `NOT met`. Both kinds are listed in §4 below,
so a reviewer never has to read the code to find out what is missing.

---

## 2. What runs, and what it answers

Reviewer objection → baseline, in BASELINES.md §C's order.

| # | class | baseline | how it runs | status |
|---|---|---|---|---|
| B1 | trust region, on-policy | **HAPPO** | `algorithm=happo` | new algorithm — [docs](docs/happo.md) |
| B1 | trust region, off-policy | **HASAC** | `algorithm=hasac` | new algorithm — [docs](docs/hasac.md) |
| B2 | memory | **GRU-MAPPO / R-MAPPO** | `algorithm=mappo model=layers/gru` | config — [docs](docs/rnn.md) |
| B3 | graph / communication | **GNN-MAPPO** | `algorithm=mappo model=layers/gnn` | config — [docs](docs/gnn.md) |
| B4 | mean field | **MF-AC** | `algorithm=mfac` | new algorithm — [docs](docs/mfac.md) |
| B5 | agent modelling | **LIAM** | `algorithm=liam` | new algorithm — [docs](docs/liam.md) |
| B6 | NS RL, observed context | **LCPO** | `algorithm=lcpo` | new algorithm — [docs](docs/lcpo.md) |
| B7 | NS RL, latent | **LILAC** | — | **not implemented** — [docs](docs/lilac.md) |
| B8 | meta-RL / online sys-id | **RMA / UP-OSI** | `algorithm=rma` | new algorithm — [docs](docs/rma_osi.md) |
| B9 | domain randomisation | **DR over σ** | `task.ns_dr_enabled=true` | env flag — [docs](docs/dr_sigma.md) |
| B9 | robust MARL | **ERNIE** | `algorithm=ernie` | new algorithm — [docs](docs/ernie.md) |
| B10 | classical control | **ESO / DOB** | `task.ns_baseline=eso` | env arm — [docs](docs/eso_dob.md) |
| B10 | learn the coupling | **unstructured RLS** | `task.ns_baseline=rls_raw` | env arm — [docs](docs/rls_raw.md) |
| D.4 | information grant | **oracle-driver blind** | `task.ns_observe_driver=true` | env flag — [docs](docs/dr_sigma.md) |
| B11, B12, Tier 3 | — | TPA, DGN, DCG, MAT, … | — | **skipped, with reasons** — [docs](docs/skipped.md) |

Classes for the launcher: `reference b1 b2 b3 b4 b5 b6 b8 b9 b10 grants`.

---

## 3. Where the code is

```
benchmarl/algorithms/
  happo.py  hasac.py  mfac.py  liam.py  lcpo.py  rma.py  ernie.py
  _baseline_math.py      the published formulas, torch only, so verify.py can
                         check them without torchrl
  _history.py            the observation window RMA and LIAM both need
  _compat.py             the torchrl internals the losses reach into, in one
                         place, so a torchrl bump reports by name; also
                         callback_base(), which exists because of the rule below
benchmarl/conf/algorithm/
  happo.yaml hasac.yaml mfac.yaml liam.yaml lcpo.yaml rma.yaml ernie.yaml
simple_ns/
  baselines.py           B10's two non-learning compensators, as ARMS
  observer.py            the discrete ESO, torch only
  layer.py               the new dial keys: ns_observe_driver,
                         ns_observe_prev_action, ns_dr_*, ns_baseline
scripts/
  baselines_common.sh    the table: one function per row
  run_baselines.sh       the entry point
baselines/
  verify.py              the offline checks -- syntax, plumbing, arithmetic
  docs/                  one checklist per baseline
```

### One rule for anything added under `benchmarl/algorithms/`

`benchmarl/__init__.py` imports `benchmarl.algorithms` **first**, and both
`benchmarl.experiment` and `benchmarl.environments` import names back out of
it. So an algorithm module that imports either of them at import time closes a
cycle and `import benchmarl` fails outright:

```
ImportError: cannot import name 'IppoConfig' from partially initialized
module 'benchmarl.algorithms'
```

Import them **inside the function that needs them** instead. `mappo_ctde.py`
already did this for `benchmarl.environments`; HAPPO and RMA need
`experiment.callback.Callback` as a base class, so they fetch it through
`_compat.callback_base()` and build the callback class on first use.
`verify.py` checks the rule structurally — it is invisible to a syntax check
and to anything that cannot import torchrl.

### Every new environment knob is OFF by default

`ns_observe_driver`, `ns_observe_prev_action`, `ns_dr_enabled` and
`ns_baseline` all default to the inert value in all four task yamls, and
`verify.py` checks that they do. With the defaults, `simple_ns` is bit-for-bit
the environment the existing PACT rows were run on: no arm already measured
changes because these exist.

---

## 4. Everything that is adapted or missing, in one place

This is the list a reviewer should be handed. Each item links to the doc that
explains it.

### Adapted

| baseline | adaptation | why |
|---|---|---|
| HAPPO | the optimiser budget is spent agent by agent inside BenchMARL's fixed call count, rather than HARL's explicit per-agent loop | BenchMARL fixes the number of optimiser calls per iteration; the launcher multiplies it by `n_agents` so each agent gets the same epochs MAPPO's policy gets ([happo](docs/happo.md)) |
| HAPPO | agents outside the current block are written back after every step | torchrl stores the non-shared policies as ONE stacked tensor, so per-agent optimisers are not expressible and Adam's momentum would keep moving them ([happo](docs/happo.md)) |
| HASAC | one actor update per optimiser call instead of a full sweep inside one | keeps the sequential property at one backward pass per call; the effect is HARL's `policy_freq = n_agents` ([hasac](docs/hasac.md)) |
| LIAM | PPO in place of A2C; a truncated history window in place of the whole episode; a squared-error action head in place of the softmax one | BenchMARL ships no A2C; the window is how a history reaches the policy at collection time; the action space is continuous ([liam](docs/liam.md)) |
| LCPO | a joint trust region over the product policy, one KL averaged over agents | LCPO is single-agent; torchrl's stacked parameters cannot express a per-agent conjugate gradient ([lcpo](docs/lcpo.md)) |
| LCPO | the reservoir is bounded and subsampled | LCPO keeps every state ever seen; at a 3 M-frame budget that is 3 M joint states ([lcpo](docs/lcpo.md)) |
| RMA | the history encoder is an MLP over a fixed window, not a 1-D CNN | same architecture class, same input; the window is what the environment can provide ([rma_osi](docs/rma_osi.md)) |
| MF-AC | the mean action is read off the observation | `ns_observe_prev_action` already puts each agent's last action there, so the mean needs no new environment state ([mfac](docs/mfac.md)) |
| ESO/DOB | the observer runs on the directly measured residual | on this channel the disturbance IS observable one step late, so the plant-inversion half of a DOB is the identity ([eso_dob](docs/eso_dob.md)) |

### Not implemented

| baseline | what is missing | why |
|---|---|---|
| **LILAC (B7)** | the whole method | needs a per-episode latent, an episode index carried through the replay buffer, and an LSTM prior over the sequence of all past episode latents. BenchMARL's off-policy buffer has no episode identity, and the three pieces cannot be checked offline. BASELINES.md B7 itself allows "skip and cite". [Full design, and what it would take](docs/lilac.md) |
| ERNIE | the Stackelberg / leader-follower gradient correction | needs the Jacobian of the perturbation w.r.t. every policy parameter; the authors build it only for a small traffic network, and torchrl's stacked per-agent parameters make the required in-place `param.grad +=` splice unavailable. The adversarial regulariser — the part the README calls "the simplest version of ERNIE" — is implemented ([ernie](docs/ernie.md)) |
| LCPO | `--auto_target_entropy`, LCPO's automatic entropy tuning | the decay schedule is implemented; the automatic one is a second mechanism, and the paper's runs use both ([lcpo](docs/lcpo.md)) |
| HASAC | discrete actions | HARL's discrete HASAC swaps the squashed Gaussian for a Gumbel-softmax policy — a different actor, not a flag ([hasac](docs/hasac.md)) |
| B11 TPA | — | built on MAPDN's power-grid state; there is no VMAS instantiation of it ([skipped](docs/skipped.md)) |
| B12, Tier 3 | MAT, DGN, DCG, QPLEX, M2TD3, RARL, VariBAD, PEARL, AMAGO, MAMBA, MAMBPO, MBCD, LOLA, M-FOS, Meta-MAPG | BASELINES.md Tier 3: cite, run only on request ([skipped](docs/skipped.md)) |

---

## 5. Two things to check before trusting any row

1. **Did the dial fire?** Every run writes `pact_debug.csv` beside itself. If
   `ns_live_frac` is 0 the disturbance never happened and the row is about
   nothing. Read the columns in the order documented in
   `benchmarl/environments/simple_ns/common.py`.
2. **Is the baseline actually running its own algorithm?** Each new algorithm
   prints a banner at construction saying what it resolved — HAPPO its block
   sizes, LCPO its context slice, RMA its phase-1 budget, MF-AC which mean it is
   taking. If a banner says something you did not intend, the row is measuring
   something you did not intend.

---

## 6. Status of this work

Written but **not executed**: this machine has `torch` but no `torchrl`,
`tensordict` or `vmas` installed, so nothing here has been run end to end.
`baselines/verify.py` and `simple_ns/check_plumbing.py` are the parts that
could be checked, and they pass. The first live run of each row should be a
short one — `FRAMES=60000 SEEDS=0` — read for the banner and for
`ns_live_frac`, before any sweep.
