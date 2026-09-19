# ESO / DOB — BASELINES.md B10, the classical non-learning compensator

## Source

| what | where |
|---|---|
| method | Active Disturbance Rejection Control's extended state observer. Han, *From PID to ADRC*, IEEE TIE 2009; Gao, *Scaling and bandwidth-parameterization based controller tuning*, ACC 2003; the discrete form in Miklosovic, Radke & Gao, *Discrete implementation and generalization of the extended state observer*, ACC 2006 |
| references BASELINES.md names | [pyadrc](https://pyadrc.readthedocs.io/en/latest/), [MRGilak/ADRC](https://github.com/MRGilak/Active-Disturbance-Rejection-Controller), the [ADRC toolbox paper](https://arxiv.org/pdf/2112.01614) |
| implementation | [`simple_ns/observer.py`](../../simple_ns/observer.py) (the recursion, torch only), [`simple_ns/baselines.py`](../../simple_ns/baselines.py) (`EsoMixin`) |

## What the baseline is for

BASELINES.md B10 calls this **the single most important non-RL baseline**:

> Control engineers will ask why an estimator on a declared basis beats a
> disturbance observer that needs no structure at all.
>
> Prediction: it cancels the *slow* component and lags the fast one; on the
> feeder it cannot break the loop because its estimate is one step behind the
> peers' reaction, whereas PACT's channels are computed from the peers'
> broadcast actions *before* the step.

## What it is

Per agent, one gain, no peer information, no basis:

```
e  = y(t-1) - z1
z1 <- z1 + z2 + beta1 * e        beta1 = 2w
z2 <- z2 + beta2 * e             beta2 = w^2
pred(t) = z1 + z2
```

`y` is the agent's **own** residual — on the droop channel the fractional
shortfall `1 - delivered/commanded`, which it measures directly. `z1` is the
observed disturbance, `z2` its rate, and `z1 + z2` the one-step-ahead
extrapolation. The correction is then the identical feed-forward PACT uses:
`commanded_sent = commanded / (1 - trust · pred)`.

## The checklist

| # | what B10 asks for | here | status |
|---|---|---|---|
| 1 | estimate the lumped disturbance from **the agent's own residual** | `y = self._y_prev`, the same sensor PACT's P-2.1 declares | **met** |
| 2 | **no peer information** | the arm reads no peer action, no channel, no operator | **met, structurally** — `EsoMixin` has no access to `self._x` |
| 3 | **no basis** | no design matrix, no classes | **met** |
| 4 | **one gain** | one constant, the observer bandwidth `w` | **met** |
| 5 | a **first-order** observer | a second-order (two-state) ESO | **adapted** — see (A) |
| 6 | cancel it next step | the same droop feed-forward, the same `pact_trust`, the same `pact_warmup`, the same `pact_corr_clip` | **met** — see (B) |
| 7 | the estimate is one step behind | `y(t-1)` is what the observer is fed, and there is nothing else it could be fed | **met — this is the point of the row** |

### (A) Why a two-state observer, and why the gains are what they are

B10 says "a first-order observer". A one-state observer on a directly measured
signal is a first-order low-pass, and it has a **standing error on a ramp** —
the disturbance here is a smooth bump, so it is ramping most of the time, and a
one-state observer would be handicapped by its own structure rather than by the
problem.

The standard linear-ADRC ESO carries a second state for the disturbance's rate,
which removes exactly that error and gives the one-step-ahead extrapolation the
feed-forward needs. That is the strongest honest version of the baseline, and
it is what "extended state observer" means in the ADRC literature B10 cites.
The *identification* is still first-order in the sense B10 means: one scalar
signal, one bandwidth, no structure.

The gains follow from the pole placement and nothing else. The estimation error
obeys

```
e1 <- (1 - beta1) e1 + e2
e2 <- -beta2 e1 + e2
```

with characteristic polynomial `z^2 - (2 - beta1) z + (1 - beta1 + beta2)`.
Setting that equal to `(z - (1-w))^2`:

```
2 - beta1         = 2(1 - w)    ->  beta1 = 2w
1 - beta1 + beta2 = (1 - w)^2   ->  beta2 = w^2
```

**`baselines/verify.py` checks this directly** — that both roots of the
observer's polynomial really are at `1-w`, for several `w`; that it converges to
a constant disturbance with zero steady-state error; and that the second state
removes the lag on a ramp. The first draft of this file used
`beta1 = 1 - (1-w)^2`, which is `2w - w^2`, and that check is what found it. A
wrong observer gain does not crash — it produces an observer that simply tracks
badly, and no training curve tells you which.

### (B) Everything except the estimator is PACT's

The arm reads `pact_trust`, `pact_warmup` and `pact_corr_clip` from the PACT
block on purpose. The row exists to price the **estimator**, so every other
knob is held equal:

| | PACT | ESO arm |
|---|---|---|
| sensor | own residual, one step stale | same |
| channel | `u / (1 - g·pred)` | same |
| trust | `pact_trust`, after `pact_warmup` | same |
| bound | `pact_corr_clip` | same |
| **estimate** | RLS on `[1, x_1..x_r]`, peer channels | **a 2-state observer on `y` alone** |

`pact_enabled=true` together with `ns_baseline=eso` is an **error**, not a
double feed-forward: they are alternative compensators for the same channel.

### (C) The observer state is not reset per episode

`_on_reset` clears the correction but leaves `z1`, `z2` alone: the rig does not
forget what it learned about the rail because a lift finished. This matches the
dial's own NS-3.4 (the driver's clock persists across episodes) and matches
PACT, whose RLS is also not reset.

## Running it

```bash
GROUP=b10 ONLY=eso bash scripts/run_baselines.sh
```

`eso_bandwidth: 0.3` is the default. It is a **declared** constant, not a swept
one; if it is swept, say so, and sweep it on prediction error rather than on
return — the same rule `pact_mu` is under, and for the same reason
(VMAS_BALANCE.md §9: a diverged estimator once scored better than a correct
one).

## What to watch

The arm writes the same diagnostic columns PACT does, so the two are directly
comparable:

| column | what it tells you |
|---|---|
| `pact_pred` | the observer's estimate. Compare against `ns_load`, the truth. |
| `pact_corr_vs_d` | correction over disturbance. Approaching 1 means it is cancelling; well under 1 with a live disturbance is the lag B10 predicts. |
| `pact_trust_applied` | 0 before `pact_warmup`, `pact_trust` after. The ESO has no confidence gate — it has no covariance to gate on — so applied trust is the constant. That asymmetry with PACT is real and should be stated. |
| `ns_clipped_frac` | if the feed-forward saturates the action box, the row is about the box. |
