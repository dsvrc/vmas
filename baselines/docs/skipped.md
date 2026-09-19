# Skipped, and why

BASELINES.md's own Tier 3 is "cite, run only on request", and §E is "methods to
cite and argue rather than run". This file records everything not implemented,
with the reason, so nothing is missing by accident.

## Skipped because the host is wrong

| method | BASELINES.md | why not on VMAS |
|---|---|---|
| **TPA for AVC** | B11, Tier 1 | Temporal Prototype-Aware learning for *active voltage control*. Its inputs are the power network's operating state — bus voltages, load and PV profiles — and its prototypes are over the daily pattern of those. Its code targets MAPDN. There is no VMAS instantiation: porting it would mean inventing what a "prototype" is on a beam-balancing task, which is inventing a method and calling it TPA. **Run it on MAPDN**, where BASELINES.md puts it and where the port is the launcher only. |
| **Safety-constrained MARL for AVC** (IJCAI 2024) | B11 | no public code, and the same host problem |
| **DGN** | B3 alternative | value-based (Q-learning with graph attention), discrete actions. This host is continuous. BASELINES.md chooses GNN-MAPPO as B3's representative and marks DGN "optional"; [gnn.md](gnn.md) is the row that runs. |
| **DCG** | B3 alternative | value factorised on a coordination graph, PyMARL-based, discrete actions → BASELINES.md itself assigns it to URB |
| **QPLEX** | B1 alternative | value factorisation for discrete actions; BASELINES.md lists it as an alternative to cite |

## Skipped because the setting does not apply

| method | BASELINES.md | why |
|---|---|---|
| **LOLA**, **M-FOS**, **Meta-MAPG**, POLA | B5 alternatives, §E | opponent *shaping*: defined for two-player general-sum games with white-box or meta-game access to the opponent's learning. Not applicable to 4 cooperative supports on one beam, and BASELINES.md §E says so. The (B)-vs-(C) intercept experiment already shows the drift is not learning-induced. |
| **Multi-timescale decentralised learning** (CoLLAs 2023) | §E | addresses *learning-induced* non-stationarity by staggering learning rates; no public code |
| **Provably-optimal non-stationary RL** (sliding-window / restart bandits; the ICLR 2024 black-box approach) | §E | theory for tabular or linear settings; cite as the source of the "re-learn" paradigm |
| **FANS-RL** | B7 alternative | no public code |
| **Decision Adapter** | B8 alternative | no public code |

## Skipped as Tier 3 — cite, run only on request

BASELINES.md §D: *"Tier 3 — cite, run only on request: MAT, DGN, DCG, QPLEX,
M2TD3, RARL, VariBAD, PEARL, AMAGO, MAMBA, MAMBPO, MBCD."*

Each one has a representative of its class that **is** implemented:

| Tier 3 | class | the row that covers it |
|---|---|---|
| MAT | sequential-update / trust-region MARL | **HAPPO** ([happo.md](happo.md)) |
| DGN, DCG | graph-structured policies | **GNN-MAPPO** ([gnn.md](gnn.md)) |
| QPLEX | value factorisation | QMIX / VDN, already runnable on the stock host |
| M2TD3, RARL | robust RL | **DR over σ** and **ERNIE** ([dr_sigma.md](dr_sigma.md), [ernie.md](ernie.md)) |
| VariBAD, PEARL, AMAGO | meta-RL / in-context adaptation | **RMA / UP-OSI** ([rma_osi.md](rma_osi.md)) |
| MAMBA, MAMBPO | model-based MARL (B12) | none — see below |
| MBCD | change-point detection | none — see [lilac.md](lilac.md) |

## B12: model-based MARL

BASELINES.md B12 is already "cite; run only if asked". Its prediction is that
*a world model absorbs the drift into its latent but has no separate,
invertible object for the coupling gain; sample cost is large.* Neither MAMBA
nor MAMBPO has a VMAS adapter, and the sample cost of a world model on a
3 M-frame budget makes it a separate project rather than a row.

## Not skipped, but incomplete

Two things are implemented in part, and both are listed in
`baselines/README.md` §4:

* **ERNIE's Stackelberg / leader-follower correction** — the adversarial
  regulariser is implemented; the optimisation-stability refinement is not. See
  [ernie.md](ernie.md) (C).
* **LCPO's `--auto_target_entropy`** — the entropy *decay* schedule is
  implemented; the automatic tuning is a second mechanism and is not. See
  [lcpo.md](lcpo.md), row 21.

And one is not implemented at all:

* **LILAC (B7)** — see [lilac.md](lilac.md) for the full design and the exact
  reason.
