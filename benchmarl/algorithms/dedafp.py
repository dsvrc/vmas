#  DEDA-FP -- deep fictitious play for CONTINUOUS, NON-STATIONARY mean field
#  games.
#
#      Magnino, Shao, Wu, Shen, Lauriere, "Solving Continuous Mean Field Games:
#      Deep Reinforcement Learning for Non-Stationary Dynamics", arXiv
#      2510.22158 (NeurIPS 2025).
#
#  Algorithm 3, in the paper's own three lines:
#
#      for k = 1..K:
#          1. BEST RESPONSE   pi*_k = argmax_pi J(pi, Gbar_{k-1})     [deep RL]
#          2. AVERAGE         pi_k  = argmin_theta L_NLL over M_SL    [supervised]
#          3. DISTRIBUTION    Gbar_k = argmax_phi  log q_phi(x | t)   [cond. flow]
#      return pi_K, Gbar_K
#
#  and the three pieces are three different kinds of learning, which is the
#  point: the best response is RL, the average policy is a maximum-likelihood
#  fit to the actions every past best response took (this is what makes it
#  fictitious play rather than an average of weights), and the population
#  distribution is a CONDITIONAL NORMALIZING FLOW over states given the time
#  index, because a non-stationary mean field game's equilibrium measure is a
#  function of t.
#
#  Why it belongs in this paper's table.  Every other non-stationarity baseline
#  here treats the disturbance as exogenous -- something to observe (LCPO),
#  identify (RMA), average over (DR) or detect (QCD+).  A mean field game says
#  the thing that varies IS the population, and the right object to learn is
#  the population's distribution.  On this instance the disturbance an agent
#  feels is literally a functional of what the other agents are doing, so "the
#  mean field is the non-stationarity" is a live hypothesis and this row is
#  what tests it.
#
#  REQUIRES `task.ns_observe_time=true`: both the average policy and the flow
#  are conditioned on t, and a finite-horizon mean field equilibrium is not
#  defined without it.
#
#  See `baselines/docs/dedafp.md` for the clause-by-clause checklist, including
#  the two things that are NOT met -- the density does not enter the REWARD
#  (that would change the task, which every arm shares), and the population is
#  not literally made to play pibar (the mean field abstraction is what the
#  frozen Gbar_{k-1} stands for).

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, List, Optional, Tuple, Type

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import IndependentNormal, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms import _compat
from benchmarl.algorithms._baseline_math import (
    affine_coupling,
    gaussian_nll,
    standard_normal_log_prob,
)
from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.ippo import Ippo, IppoConfig
from benchmarl.models.common import ModelConfig


POLICY_INPUT_KEY = "fp_policy_input"
BR_LOGITS_KEY = "fp_logits_br"
AVG_LOGITS_KEY = "fp_logits_avg"


# ===========================================================================
#  The conditional normalizing flow:  Gbar_k(. | t)
# ===========================================================================


class ConditionalFlow(nn.Module):
    """A time-conditioned normalizing flow over the population's states.

    ``n_layers`` affine coupling layers with alternating masks; each layer's
    shift and log-scale are produced by a small MLP that reads the masked-in
    half of ``x`` TOGETHER WITH the conditioning variable ``t``, which is what
    makes the density time-dependent.  ``log_prob`` is exact -- base
    log-density plus the accumulated log-determinant -- so the fit is maximum
    likelihood and not a bound.

    The reference builds autoregressive neural spline flows with 16 layers and
    spline parameters that are functions of ``t``.  An affine coupling flow is
    the same object (an exactly invertible map with a tractable
    log-determinant, trained by MLE, conditioned on ``t``) with a simpler
    elementwise transform; see baselines/docs/dedafp.md, "adapted".
    """

    def __init__(self, dim: int, n_layers: int, hidden: int, cond_dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError(
                f"a coupling flow needs at least 2 dimensions to split; got "
                f"{dim}. Widen flow_size."
            )
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        masks = []
        for layer in range(self.n_layers):
            mask = torch.zeros(self.dim)
            mask[layer % 2 :: 2] = 1.0
            masks.append(mask)
        self.register_buffer("masks", torch.stack(masks))
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim + cond_dim, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, 2 * self.dim),
                )
                for _ in range(self.n_layers)
            ]
        )
        for net in self.nets:
            #  Start at the identity map, so an untrained flow is the standard
            #  normal rather than an arbitrary warp -- otherwise the density
            #  channel is noise for the first few thousand frames and the
            #  policy learns to ignore it.
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = x
        log_det = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for layer in range(self.n_layers):
            mask = self.masks[layer]
            params = self.nets[layer](torch.cat([z * mask, cond], dim=-1))
            shift, log_scale = params.chunk(2, dim=-1)
            #  Bounded log-scale: an unbounded one makes the log-determinant
            #  and hence the loss diverge on the first badly scaled batch.
            log_scale = torch.tanh(log_scale) * 3.0
            z, step_det = affine_coupling(z, shift, log_scale, mask)
            log_det = log_det + step_det
        return standard_normal_log_prob(z) + log_det


# ===========================================================================
#  The actor: best response, average policy, and the density channel
# ===========================================================================


class FpDensityInput(nn.Module):
    """``[observation, muhat(x | t)]`` -- the policy's input.

    The mean field enters the agent through the density the flow reports at the
    agent's OWN state at the current time.  That is the query the paper uses
    the flow for ("the learned normalizing flow provides direct density queries
    mu(x)"); what differs here is where the answer goes.  In the paper it goes
    into the reward, because the MFG's reward is defined to depend on the
    population density.  A VMAS task's reward is the task's, shared by every
    arm in this repo, and changing it for one baseline would make that arm
    incomparable -- so the density is supplied to the POLICY instead.  It is
    the same channel carrying the same information; it is marked as an
    adaptation in baselines/docs/dedafp.md and it is the one place this row
    departs from Algorithm 3.
    """

    def __init__(
        self,
        flow: ConditionalFlow,
        state_slice: slice,
        time_slice: slice,
        density_clip: float,
    ):
        super().__init__()
        self.flow = flow
        self.state_slice = state_slice
        self.time_slice = time_slice
        self.density_clip = float(density_clip)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = observation[..., self.state_slice]
        cond = observation[..., self.time_slice]
        #  Detached: the flow is fitted by maximum likelihood between
        #  fictitious-play iterations and is FROZEN during the best response,
        #  which is what "argmax_pi J(pi, Gbar_{k-1})" means -- the mean field
        #  the agent best-responds to does not move while it does so.
        with torch.no_grad():
            density = self.flow.log_prob(x, cond).exp()
        density = density.clamp(0.0, self.density_clip).unsqueeze(-1)
        return torch.cat([observation, density], dim=-1)


class FpSelect(nn.Module):
    """Which policy is ACTING: the best response, or the average.

    Algorithm 3 returns ``pibar_K``, not ``pi*_K``: the equilibrium object in
    fictitious play is the average, and the best responses are the means of
    computing it.  BenchMARL reports the return of the policy that collects, so
    the last fictitious-play iteration switches the acting policy to ``pibar``
    and the best-response objective is zeroed for the duration -- that
    iteration measures ``pibar_K`` and trains nothing but the critic.  Both the
    collection policy and the loss policy are this one module, so the switch
    reaches both at once.
    """

    def __init__(self):
        super().__init__()
        self.use_average = False

    def forward(self, logits_br: torch.Tensor, logits_avg: torch.Tensor):
        return logits_avg if self.use_average else logits_br


# ===========================================================================
#  The supervised-learning reservoir:  M_SL
# ===========================================================================


class SlReservoir:
    """``M_SL``: every ``(t, s, a)`` any past best response produced.

    The reference appends ``N_sa`` fresh samples per iteration and never
    forgets.  Bounded here by reservoir sampling, which keeps a UNIFORM sample
    of everything ever added -- so the average policy is still fitted to the
    uniform mixture over iterations that fictitious play calls for, rather than
    to the most recent iterations.  Stated as an adaptation in
    baselines/docs/dedafp.md.
    """

    def __init__(self, capacity: int, generator: torch.Generator):
        self.capacity = int(capacity)
        self.generator = generator
        self.inputs: Optional[torch.Tensor] = None
        self.actions: Optional[torch.Tensor] = None
        self.n_seen = 0
        self.n_stored = 0

    def add(self, inputs: torch.Tensor, actions: torch.Tensor) -> None:
        if self.inputs is None:
            self.inputs = torch.zeros(
                self.capacity, *inputs.shape[1:], device=inputs.device
            )
            self.actions = torch.zeros(
                self.capacity, *actions.shape[1:], device=actions.device
            )
        n = inputs.shape[0]
        free = self.capacity - self.n_stored
        take = min(free, n)
        if take > 0:
            self.inputs[self.n_stored : self.n_stored + take] = inputs[:take]
            self.actions[self.n_stored : self.n_stored + take] = actions[:take]
            self.n_stored += take
        if take < n:
            #  Vectorised reservoir sampling for the overflow: element j (the
            #  (n_seen + take + j)-th ever seen) replaces a uniformly chosen
            #  slot with probability capacity / (n_seen + take + j + 1).
            rest = n - take
            positions = torch.arange(
                rest, device=inputs.device, dtype=torch.float64
            ) + float(self.n_seen + take + 1)
            keep = (
                torch.rand(rest, generator=self.generator).to(inputs.device).double()
                < (self.capacity / positions)
            )
            idx = torch.randint(
                self.capacity, (rest,), generator=self.generator
            ).to(inputs.device)
            chosen = keep.nonzero(as_tuple=True)[0]
            if chosen.numel():
                self.inputs[idx[chosen]] = inputs[take:][chosen]
                self.actions[idx[chosen]] = actions[take:][chosen]
        self.n_seen += n

    def sample(self, batch_size: int):
        if self.n_stored == 0:
            return None
        idx = torch.randint(
            self.n_stored, (batch_size,), generator=self.generator
        ).to(self.inputs.device)
        return self.inputs[idx], self.actions[idx]


# ===========================================================================
#  The loss
# ===========================================================================


class DedaFpLoss(ClipPPOLoss):
    """The best-response step, plus the fictitious-play diagnostics.

    The best response is ordinary PPO against a FROZEN mean field -- Algorithm
    3 line 1 is "solve the MDP induced by ``Gbar_{k-1}``" and imposes nothing
    on how.  The averaging and the flow are fitted between iterations by
    :meth:`DedaFp.fictitious_play_step`, each with its own optimiser, so
    neither gradient ever reaches the best response.
    """

    #  Redeclared so torchrl's convert_to_functional does not warn: it checks
    #  the SUBCLASS's own __annotations__.  Same list torchrl's own losses
    #  carry.
    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(self, *args, state: "FpState", **kwargs):
        super().__init__(*args, **kwargs)
        self.fp_state = state

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_out = super().forward(tensordict)
        state = self.fp_state
        if state.selector.use_average:
            #  The measuring iteration: pibar is acting, so the stored
            #  log-probabilities are pibar's and the importance ratio carries
            #  no information about the best response. Zeroed rather than
            #  skipped so the optimiser still runs and the tensordict keeps the
            #  same keys every call, which is what BenchMARL's stack needs.
            td_out.set("loss_objective", td_out.get("loss_objective") * 0.0)
        device = td_out.device

        def _log(name, value):
            td_out.set(
                name,
                torch.as_tensor(float(value), device=device, dtype=torch.float32),
            )

        _log("dedafp_iter", state.iteration)
        _log("dedafp_mode", 1.0 if state.selector.use_average else 0.0)
        _log("dedafp_sl_nll", state.last_sl_nll)
        _log("dedafp_flow_nll", state.last_flow_nll)
        _log("dedafp_buffer", state.reservoir.n_stored)
        _log("dedafp_density", state.last_density)
        return td_out


class FpState:
    """What the fictitious-play outer loop remembers, per group."""

    def __init__(
        self,
        flow: ConditionalFlow,
        selector: FpSelect,
        reservoir: SlReservoir,
        state_slice: slice,
        time_slice: slice,
    ):
        self.flow = flow
        self.selector = selector
        self.reservoir = reservoir
        self.state_slice = state_slice
        self.time_slice = time_slice
        self.iteration = 0
        self.last_sl_nll = 0.0
        self.last_flow_nll = 0.0
        self.last_density = 0.0
        self.avg_optimizer: Optional[torch.optim.Optimizer] = None
        self.flow_optimizer: Optional[torch.optim.Optimizer] = None


# ===========================================================================


class DedaFp(Ippo):
    """DEDA-FP on BenchMARL's IPPO host.

    Args:
        fp_iters (int): ``K``, the number of fictitious-play iterations. The
            frame budget is split evenly between them.
        state_start, state_size (int): the slice of the observation the
            population distribution is modelled over. ``0``/``0`` means
            "everything before the time column".
        time_start, time_size (int): where ``t`` lives. ``task.ns_observe_time``
            appends it LAST, so ``-1``/``1`` is right for every simple_ns host.
        flow_layers, flow_hidden (int): the conditional flow's depth and width.
        flow_epochs, flow_batch (int): maximum-likelihood steps per
            fictitious-play iteration, and their minibatch size.
        flow_lr (float): the flow's own learning rate.
        sl_capacity (int): how many ``(t, s, a)`` triples ``M_SL`` keeps.
        sl_epochs, sl_batch (int): supervised steps per fictitious-play
            iteration for the average policy, and their minibatch size.
        sl_lr (float): the average policy's own learning rate.
        sl_subsample (int): keep one transition in ``n`` from each iteration.
        density_clip (float): declared bound on the density channel.
        deploy_average_last (bool): make ``pibar`` the acting policy for the
            last fictitious-play iteration, so the number the run reports is
            the equilibrium policy's and not the last best response's.

    All other arguments are :class:`~benchmarl.algorithms.Ippo`'s.
    """

    def __init__(
        self,
        fp_iters: int,
        state_start: int,
        state_size: int,
        time_start: int,
        time_size: int,
        flow_layers: int,
        flow_hidden: int,
        flow_epochs: int,
        flow_batch: int,
        flow_lr: float,
        sl_capacity: int,
        sl_epochs: int,
        sl_batch: int,
        sl_lr: float,
        sl_subsample: int,
        density_clip: float,
        deploy_average_last: bool,
        **kwargs,
    ):
        self.fp_iters = int(fp_iters)
        self.state_start = int(state_start)
        self.state_size = int(state_size)
        self.time_start = int(time_start)
        self.time_size = int(time_size)
        self.flow_layers = int(flow_layers)
        self.flow_hidden = int(flow_hidden)
        self.flow_epochs = int(flow_epochs)
        self.flow_batch = int(flow_batch)
        self.flow_lr = float(flow_lr)
        self.sl_capacity = int(sl_capacity)
        self.sl_epochs = int(sl_epochs)
        self.sl_batch = int(sl_batch)
        self.sl_lr = float(sl_lr)
        self.sl_subsample = max(int(sl_subsample), 1)
        self.density_clip = float(density_clip)
        self.deploy_average_last = bool(deploy_average_last)
        super().__init__(**kwargs)

        if self.fp_iters < 2:
            raise ValueError(
                f"fp_iters={self.fp_iters}: fictitious play with fewer than "
                "two iterations has nothing to average."
            )
        if self.has_rnn:
            raise NotImplementedError(
                "DEDA-FP here does not support recurrent models: the average "
                "policy is fitted to stored (t, s, a) triples, which a "
                "recurrent policy cannot be evaluated on without its hidden "
                "state."
            )
        self._generator = torch.Generator().manual_seed(int(self.experiment.seed))
        self._states: Dict[str, FpState] = {}
        self._modules: Dict[str, Dict] = {}
        self._losses: Dict[str, DedaFpLoss] = {}

    # ------------------------------------------------------------------
    #  slicing
    # ------------------------------------------------------------------

    def _obs_dim(self, group: str) -> int:
        return int(self.observation_spec[group, "observation"].shape[-1])

    def _time_slice(self, group: str) -> slice:
        obs_dim = self._obs_dim(group)
        start = self.time_start
        if start < 0:
            start += obs_dim
        stop = start + self.time_size
        if not (0 <= start < stop <= obs_dim):
            raise ValueError(
                f"DEDA-FP time slice [{start}, {stop}) does not fit an "
                f"observation of width {obs_dim} for group {group!r}. Both the "
                "average policy and the conditional flow are functions of t: "
                "launch with `task.ns_observe_time=true`."
            )
        return slice(start, stop)

    def _state_slice(self, group: str, time_slice: slice) -> slice:
        obs_dim = self._obs_dim(group)
        start = self.state_start
        if start < 0:
            start += obs_dim
        size = self.state_size or (time_slice.start - start)
        stop = start + size
        if not (0 <= start < stop <= obs_dim):
            raise ValueError(
                f"DEDA-FP state slice [{start}, {stop}) does not fit an "
                f"observation of width {obs_dim} for group {group!r}."
            )
        if start < time_slice.stop and stop > time_slice.start:
            raise ValueError(
                f"DEDA-FP state slice [{start}, {stop}) overlaps the time "
                f"slice [{time_slice.start}, {time_slice.stop}). The flow is "
                "the density of the state GIVEN t; t cannot also be one of "
                "the modelled coordinates."
            )
        return slice(start, stop)

    # ------------------------------------------------------------------
    #  the actor
    # ------------------------------------------------------------------

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        obs_dim = self._obs_dim(group)
        time_slice = self._time_slice(group)
        state_slice = self._state_slice(group, time_slice)

        flow = ConditionalFlow(
            dim=state_slice.stop - state_slice.start,
            n_layers=self.flow_layers,
            hidden=self.flow_hidden,
            cond_dim=time_slice.stop - time_slice.start,
        ).to(self.device)
        selector = FpSelect()

        augment = TensorDictModule(
            FpDensityInput(
                flow=flow,
                state_slice=state_slice,
                time_slice=time_slice,
                density_clip=self.density_clip,
            ),
            in_keys=[(group, "observation")],
            out_keys=[(group, POLICY_INPUT_KEY)],
        )

        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = Composite(
            {
                group: Composite(
                    {
                        POLICY_INPUT_KEY: Unbounded(
                            shape=(n_agents, obs_dim + 1), device=self.device
                        )
                    },
                    shape=(n_agents,),
                )
            }
        )

        def _make(out_key):
            return model_config.get_model(
                input_spec=actor_input_spec,
                output_spec=Composite(
                    {
                        group: Composite(
                            {out_key: Unbounded(shape=logits_shape)},
                            shape=(n_agents,),
                        )
                    }
                ),
                agent_group=group,
                input_has_agent_dim=True,
                n_agents=n_agents,
                centralised=False,
                share_params=self.experiment_config.share_policy_params,
                device=self.device,
                action_spec=self.action_spec,
            )

        br_model = _make(BR_LOGITS_KEY)
        avg_model = _make(AVG_LOGITS_KEY)
        select = TensorDictModule(
            selector,
            in_keys=[(group, BR_LOGITS_KEY), (group, AVG_LOGITS_KEY)],
            out_keys=[(group, "logits")],
        )

        extractor = NormalParamExtractor(scale_mapping=self.scale_mapping)
        self._modules[group] = {
            "flow": flow,
            "br": br_model,
            "avg": avg_model,
            "selector": selector,
            "augment": augment,
            "extractor": extractor,
            "n_agents": n_agents,
        }
        self._states[group] = FpState(
            flow=flow,
            selector=selector,
            reservoir=SlReservoir(self.sl_capacity, self._generator),
            state_slice=state_slice,
            time_slice=time_slice,
        )

        sequence = TensorDictSequential(augment, br_model, avg_model, select)
        if continuous:
            extractor = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            return ProbabilisticActor(
                module=TensorDictSequential(sequence, extractor),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )
        return ProbabilisticActor(
            module=sequence,
            spec=self.action_spec[group, "action"],
            in_keys=[(group, "logits")],
            out_keys=[(group, "action")],
            distribution_class=Categorical,
            return_log_prob=True,
            log_prob_key=(group, "log_prob"),
        )

    # ------------------------------------------------------------------
    #  parameter partition
    # ------------------------------------------------------------------

    def _partition(self, group: str, loss: LossModule):
        """Split the actor's leaves into best response / average / flow.

        ``convert_to_functional`` holds the SAME Parameter objects the modules
        do -- the identity test is exact -- and the key-path test is the
        fallback for a future torchrl that clones them.  The split has to be
        exact, because it is the only thing keeping the supervised gradient out
        of the best response and the policy gradient out of the flow.
        """
        modules = self._modules[group]
        ids = {
            name: {id(p) for p in modules[name].parameters()}
            for name in ("br", "avg", "flow")
        }
        buckets: Dict[str, List[torch.Tensor]] = {"br": [], "avg": [], "flow": []}
        for key, value in loss.actor_network_params.items(True, True):
            flat = key if isinstance(key, str) else ".".join(str(k) for k in key)
            flat = f".{flat}."
            for name in ("flow", "avg", "br"):
                if id(value) in ids[name] or f".{name}." in flat:
                    buckets[name].append(value)
                    break
            else:
                buckets["br"].append(value)
        return buckets

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        buckets = self._partition(group, loss)
        expected = len(list(self._modules[group]["avg"].parameters()))
        if len(buckets["avg"]) != expected:
            raise RuntimeError(
                "DEDA-FP could not separate the average policy's parameters "
                f"from the best response's: expected {expected} leaves, found "
                f"{len(buckets['avg'])}. The supervised loss and the policy "
                "gradient would then share an optimiser, and the average "
                "policy would stop being an average."
            )
        #  The flow and the average policy are DELIBERATELY absent: each is
        #  fitted by its own maximum-likelihood optimiser between
        #  fictitious-play iterations (see `fictitious_play_step`), and giving
        #  either of them to BenchMARL's optimiser would train it with the
        #  policy gradient.
        return {
            "loss_objective": buckets["br"],
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    # ------------------------------------------------------------------
    #  the loss
    # ------------------------------------------------------------------

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if not continuous:
            raise NotImplementedError(
                "DEDA-FP here covers continuous actions only: the averaged "
                "policy is a Gaussian fitted by maximum likelihood, which is "
                "the paper's L_NLL, and a discrete average policy is a "
                "different supervised head."
            )
        state = self._states[group]
        iter_frames = self._iter_frames()
        print(
            f"DEDA-FP {group}: K={self.fp_iters} fictitious-play iterations of "
            f"~{iter_frames} frames each; the population distribution is a "
            f"{self.flow_layers}-layer conditional affine-coupling flow over "
            f"observation[..., {state.state_slice.start}:"
            f"{state.state_slice.stop}] given t = observation[..., "
            f"{state.time_slice.start}:{state.time_slice.stop}]; the density "
            f"muhat(x|t) is appended to the POLICY's input (clipped at "
            f"{self.density_clip}); M_SL keeps {self.sl_capacity} (t, s, a) "
            f"triples and the average policy takes {self.sl_epochs} NLL steps "
            f"per iteration; "
            + (
                "the LAST iteration deploys pibar and zeroes the best-response "
                "objective, so the reported return is the equilibrium "
                "policy's"
                if self.deploy_average_last
                else "the reported return is the LAST BEST RESPONSE's, not "
                "pibar's -- see baselines/docs/dedafp.md"
            )
        )
        loss_module = DedaFpLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            **_compat.coefficient_kwargs(
                DedaFpLoss,
                entropy_coef=self.entropy_coef,
                critic_coef=self.critic_coef,
            ),
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            state=state,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        self._losses[group] = loss_module
        return loss_module, False

    # ------------------------------------------------------------------
    #  the fictitious-play outer loop
    # ------------------------------------------------------------------

    def _iter_frames(self) -> int:
        total = int(self.experiment_config.get_max_n_frames(self.on_policy))
        return max(total // self.fp_iters, 1)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        batch = super().process_batch(group, batch)
        state = self._states[group]

        #  Algorithm 3 line 2: every (t, s, a) the CURRENT best response
        #  produced joins M_SL.  Added every iteration, not only at the
        #  boundary, because the buffer is the union over best responses and
        #  the boundary is only when the fit is re-run.  Nothing is added
        #  during the measuring iteration: those actions are pibar's, and
        #  refitting pibar to its own output is not fictitious play.
        #
        #  The policy input is RECOMPUTED from the observation rather than read
        #  back off the batch.  It is the same number -- the flow is frozen for
        #  the whole iteration, by construction -- and it does not depend on
        #  which intermediate keys the collector happens to carry.
        if not state.selector.use_average:
            n_agents = self._modules[group]["n_agents"]
            observation = batch.get((group, "observation"))
            action = batch.get((group, "action"))
            flat_obs = observation.reshape(-1, n_agents, observation.shape[-1])
            flat_act = action.reshape(-1, n_agents, action.shape[-1])
            with torch.no_grad():
                flat_in = self._modules[group]["augment"].module(flat_obs)
            state.reservoir.add(
                flat_in[:: self.sl_subsample].detach(),
                flat_act[:: self.sl_subsample].detach(),
            )
            state.last_density = float(flat_in[..., -1].mean())

        iteration = min(
            self.experiment.total_frames // self._iter_frames(), self.fp_iters - 1
        )
        if iteration != state.iteration:
            state.iteration = int(iteration)
            self.fictitious_play_step(group)
            if self.deploy_average_last and state.iteration == self.fp_iters - 1:
                state.selector.use_average = True
                print(
                    f"[dedafp] {group}: fictitious-play iteration "
                    f"{state.iteration} (the last) -- pibar is now the ACTING "
                    "policy and the best-response objective is zeroed. The "
                    "return from here on is the equilibrium policy's."
                )
        return batch

    def fictitious_play_step(self, group: str) -> None:
        """Algorithm 3 lines 3 and 4: refit ``pibar``, then refit ``Gbar``."""
        state = self._states[group]
        modules = self._modules[group]
        loss = self._losses[group]
        buckets = self._partition(group, loss)

        if state.avg_optimizer is None:
            state.avg_optimizer = torch.optim.Adam(buckets["avg"], lr=self.sl_lr)
            state.flow_optimizer = torch.optim.Adam(
                buckets["flow"], lr=self.flow_lr
            )

        # --- pibar_k: maximum likelihood over M_SL ------------------------
        nll_total, nll_steps = 0.0, 0
        for _ in range(self.sl_epochs):
            drawn = state.reservoir.sample(self.sl_batch)
            if drawn is None:
                break
            inputs, targets = drawn
            #  The average policy is a function of (t, s) alone.  It is
            #  evaluated here through the SAME module the actor uses -- the
            #  parameters torchrl's functional copy holds ARE this module's,
            #  which is what lets one optimiser here reach the network that
            #  acts.  Same identity RMA's phi/base split rests on.
            td = TensorDict(
                {
                    group: TensorDict(
                        {POLICY_INPUT_KEY: inputs},
                        batch_size=[inputs.shape[0], inputs.shape[1]],
                    )
                },
                batch_size=[inputs.shape[0]],
            )
            logits = modules["avg"](td).get((group, AVG_LOGITS_KEY))
            loc, scale = modules["extractor"](logits)
            nll = gaussian_nll(loc, scale, targets).mean()
            state.avg_optimizer.zero_grad()
            nll.backward()
            state.avg_optimizer.step()
            nll_total += float(nll.detach())
            nll_steps += 1
        if nll_steps:
            state.last_sl_nll = nll_total / nll_steps

        # --- Gbar_k: maximum likelihood of the population's states --------
        #  Fitted on the states M_SL holds, which is the union over every best
        #  response so far -- the empirical occupancy of the fictitious-play
        #  AVERAGE, which is the measure the reference draws from pibar_k.  See
        #  baselines/docs/dedafp.md, "adapted".
        flow_total, flow_steps = 0.0, 0
        offset = state.state_slice
        cond = state.time_slice
        for _ in range(self.flow_epochs):
            drawn = state.reservoir.sample(self.flow_batch)
            if drawn is None:
                break
            inputs, _ = drawn
            x = inputs[..., offset]
            t = inputs[..., cond]
            nll = -modules["flow"].log_prob(x, t).mean()
            state.flow_optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(buckets["flow"], 10.0)
            state.flow_optimizer.step()
            flow_total += float(nll.detach())
            flow_steps += 1
        if flow_steps:
            state.last_flow_nll = flow_total / flow_steps

        print(
            f"[dedafp] {group}: fictitious-play iteration {state.iteration} at "
            f"{self.experiment.total_frames} frames -- M_SL holds "
            f"{state.reservoir.n_stored} of {state.reservoir.n_seen} triples; "
            f"pibar NLL {state.last_sl_nll:.4f} over {nll_steps} steps; "
            f"Gbar NLL {state.last_flow_nll:.4f} over {flow_steps} steps"
        )


@dataclass
class DedaFpConfig(IppoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.DedaFp`."""

    fp_iters: int = MISSING
    state_start: int = MISSING
    state_size: int = MISSING
    time_start: int = MISSING
    time_size: int = MISSING
    flow_layers: int = MISSING
    flow_hidden: int = MISSING
    flow_epochs: int = MISSING
    flow_batch: int = MISSING
    flow_lr: float = MISSING
    sl_capacity: int = MISSING
    sl_epochs: int = MISSING
    sl_batch: int = MISSING
    sl_lr: float = MISSING
    sl_subsample: int = MISSING
    density_clip: float = MISSING
    deploy_average_last: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return DedaFp

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False
