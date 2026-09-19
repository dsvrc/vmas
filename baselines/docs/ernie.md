# ERNIE — BASELINES.md B9, robust MARL by adversarial regularization

## Source

| what | where |
|---|---|
| paper | Bukharin et al., *Robust Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical Foundation and Stable Algorithms*, NeurIPS 2023 ([arXiv 2310.10810](https://arxiv.org/abs/2310.10810)) |
| code read | [abukharin3/ERNIE](https://github.com/abukharin3/ERNIE): the README's own "simplest version of ERNIE" snippet, and `Algorithms/coma.py` — the `perturb_actor` branch, the `perturb_critic` branch, `get_adv_reg_loss`, and `Algorithms/configs/config_coma.py` for the defaults |
| implementation | [`benchmarl/algorithms/ernie.py`](../../benchmarl/algorithms/ernie.py), [`benchmarl/conf/algorithm/ernie.yaml`](../../benchmarl/conf/algorithm/ernie.yaml) |

## What the baseline is for

BASELINES.md B9: the robust-RL answer — *hedge against the drift instead of
identifying it*. The prediction: *robustness costs σ=0 performance and still
does not cancel; the asymptote lies between blind and PACT.*

ERNIE is the MARL-specific representative, chosen because it claims robustness
to **changing transition dynamics**, which is what the dial does.

## What the method is

One extra term on the policy loss: a penalty on how far the policy output moves
when the observation is pushed in the direction that moves it most. That is a
local Lipschitz penalty on π.

```
s~ <- s + N(0, 1e-3)
repeat perturb_num_steps times:
    d  <- || f(s) - f(s~) ||_F
    g  <- d(d)/d(s~)                      (clamped to +-perturb_radius)
    s~ <- s~ + perturb_alpha * g * |s|
loss <- loss + lam * || f(s) - f(s~.detach()) ||_F
```

## The checklist

| # | idea | the reference's line | here | status |
|---|---|---|---|---|
| 1 | the perturbation starts from Gaussian noise of std 1e-3 | `torch.normal(old_obs, ones_like(old_obs) * 1e-3)` | `perturb_init_std: 0.001` | **met** |
| 2 | the perturbation ascends the **policy-distance** gradient | README: `distance_loss = norm(net(obs) - net(perturbed), p="fro")` then `grad(...)` | `ErnieLoss.adv_reg_loss`, the ascent loop | **met** |
| 3 | the step is scaled by `|obs|` | README: `+ perturb_alpha * grad * torch.abs(obs.detach())` | `scale_by_obs: true` | **met** |
| 4 | the gradient is projected onto a ball before stepping | `coma.py`: `obs_grad = torch.clamp(obs_grad, -perturb_radius, perturb_radius)` | `perturb_radius: 1.0`; `0` disables it, which is the README's version | **met, both variants available** |
| 5 | `perturb_num_steps` ascent steps | `for i in range(config.alg.perturb_num_steps)` | `perturb_num_steps: 1`, the repo's default | **met** |
| 6 | the regulariser is `\|\| f(s) - f(s~) \|\|`, with **s~ detached** | `get_adv_reg_loss`: `perturbed_obs = perturbed_obs.detach()` | identical | **met — and it matters**: without the detach the regulariser would push the perturbation as well as the policy |
| 7 | **both** forwards carry the parameter gradient | `normal = network(obs)`, `perturbed = network(perturbed_obs)` | identical | **met** |
| 8 | the norm is a **Frobenius norm over the whole minibatch**, not a mean | `torch.norm(normal - perturbed, p="fro")` | identical, reproduced as published | **met, with a warning** — see (A) |
| 9 | it is added to the policy loss with weight `lam` | `actor_loss = actor_loss + config.alg.lam * adv_reg_loss` | `td_out["loss_objective"] += lam * reg` | **met** |
| 10 | `f` is the policy network's output head | `self.actor(obs)`, which ends in a softmax | discrete: the probability vector. Continuous: `[loc, scale]`, which is what BenchMARL's actor emits before the distribution is built | **adapted** — see (B) |
| 11 | `perturb_critic`: the same regulariser on the critic instead | `config.alg.perturb_critic = False` in the repo's config | not implemented | **not run**: the repo's own default is the actor branch, and B9 is about the policy |
| 12 | the **Stackelberg / leader-follower** gradient correction | `coma.py`, the `leader_follower` branch: `d_delta_d_theta`, `smooth_partial`, `param.grad = param.grad + lam * leader_follower` | not implemented | **NOT met** — see (C) |
| 13 | the RL half | COMA in the repo; the paper also reports MAPPO variants | MAPPO | **met in kind** |

### (A) The Frobenius norm scales with the minibatch

`torch.norm(x, p="fro")` on a batched tensor is the square root of the sum of
squares over **every element**, not a mean. So the regulariser's magnitude
scales as `sqrt(minibatch_size)` and `lam` is not minibatch-invariant.

This is a property of the published code and it is reproduced rather than
quietly fixed. What it means in practice: **if you change
`experiment.on_policy_minibatch_size`, rescale `lam` by the square root of the
ratio**, or the regulariser silently changes strength. The yaml says so next to
the key.

### (B) What "the policy output" is for a continuous policy

ERNIE's `self.actor(obs)` returns a softmax over a discrete action set. The
direct analogue here is the network's output head, which for a continuous
policy is the distribution parameters `[loc, scale]` — the same tensor, one
step before the distribution is constructed. `policy_output()` reads `probs`
when the distribution has them and `[loc, scale]` otherwise, and raises on
anything else rather than guessing.

The alternative — a KL between the two distributions — would be a different
regulariser from the published one, so it is not used.

### (C) The Stackelberg correction is not implemented

The paper's "stable algorithms" section reformulates the adversarial
regularisation as a Stackelberg game and adds a leader-follower correction to
the policy gradient. The repo builds it as

```python
d_delta_d_theta[k] = torch.zeros([obs_dim] + list(param.shape))
...
leader_follower = matmul(d_delta_d_theta[k].T, smooth_partial.unsqueeze(1)) * follower_lr / lr
param.grad = param.grad + lam * leader_follower
```

i.e. the Jacobian of the perturbation with respect to **every policy
parameter**, one slab per observation dimension. Two things stop it here:

1. **Cost.** It is `obs_dim` backward passes per optimiser step and
   `obs_dim × |θ|` of storage. The authors build it only for a small traffic
   network; at a 256×256 MLP and a ~20-wide observation it is 20 extra
   double-backwards on every one of 675 optimiser calls per iteration.
2. **Expressibility.** It ends in an in-place splice into `param.grad` per
   parameter tensor. torchrl's per-agent parameters are stacked, and the
   correction is derived per agent, so the splice has no place to go.

The **adversarial regulariser** — which the authors' README calls "the simplest
version of ERNIE" and presents as the method — is implemented in full. The
missing piece is the optimisation-stability refinement, not the robustness
mechanism. This is listed in `baselines/README.md` §4 as a gap.

## Running it

```bash
GROUP=b9 ONLY=ernie bash scripts/run_baselines.sh
```

## What to watch

| column | what it tells you |
|---|---|
| `ernie_adv_reg` | the regulariser's value. It should fall as the policy smooths. **If it is zero from the start**, the ascent found no direction — check `perturb_alpha` and `perturb_radius`. If it dominates `loss_objective`, `lam` is too large for this minibatch size (see (A)). |
| σ=0 return | B9's prediction is that robustness **costs** σ=0 performance. Run `ernie` at `SIGMA=0` as well as at the committed σ, or the "it costs something" half of the claim is unmeasured. |
