#!/usr/bin/env bash
#  Shared launcher for the BASELINES.md baselines on the simple_ns hosts.
#
#  One entry point, one host configuration, one severity.  Each baseline is one
#  line in the table at the bottom of this file, and the ONLY thing that differs
#  between two lines is the flags named in it -- which is what makes the rows
#  comparable at all.  See baselines/README.md for the table in prose and
#  baselines/docs/<name>.md for what each one is.
#
#  Overridable from the environment:
#      HOST=balance SIGMA=2.0 SEEDS="0 1 2" DEVICE=cuda bash scripts/run_baselines.sh
#      GROUP=b1 bash scripts/run_baselines.sh          # one class only
#      ONLY="happo hasac" bash scripts/run_baselines.sh
#      LIST=1 bash scripts/run_baselines.sh            # print the plan, run nothing

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SAMPLING_DEVICE="${SAMPLING_DEVICE:-$DEVICE}"
TRAIN_DEVICE="${TRAIN_DEVICE:-$DEVICE}"

#  balance is the host VMAS_BALANCE.md documents end to end, so it is the
#  default here.  Every baseline runs unchanged on transport / sampling /
#  navigation -- the dial and the method are the same objects in all four.
HOST="${HOST:-balance}"

#  The severity the row is about.  Supplied from OUTSIDE the method (P-10.1),
#  identically to every arm including the baselines.
SIGMA="${SIGMA:-2.0}"

FRAMES="${FRAMES:-3000000}"
BATCH="${BATCH:-30000}"
ENVS="${ENVS:-300}"
SEEDS="${SEEDS:-0 1 2}"
MINIBATCH="${MINIBATCH:-4096}"
LOGGERS="${LOGGERS:-[csv]}"
EXTRA="${EXTRA:-}"
OUT_ROOT="${OUT_ROOT:-runs/baselines}"
LIST="${LIST:-0}"

#  Fail fast by default, which is what you want on a first run: the first row
#  that dies stops the sweep and you read one traceback instead of forty.
#  KEEP_GOING=1 for an overnight sweep, where one bad row should not cost the
#  other thirty-nine; the failures are collected and printed at the end, and
#  the script still exits non-zero.
KEEP_GOING="${KEEP_GOING:-0}"
BASELINE_FAILURES=""

#  Number of agents in the host, read from the task config rather than assumed:
#  HAPPO's optimiser budget is split agent by agent, so this number changes the
#  launch line.
NAGENTS="$(sed -n 's/^n_agents:[[:space:]]*\([0-9]\+\).*/\1/p' \
  "benchmarl/conf/task/simple_ns/${HOST}.yaml" | head -1)"
: "${NAGENTS:?could not read n_agents from benchmarl/conf/task/simple_ns/${HOST}.yaml}"

#  HAPPO gives each agent this many PPO epochs, so the on-policy optimiser
#  budget becomes n_agents * HAPPO_EPOCHS.  HARL's ppo_epoch default is 5;
#  BenchMARL's MAPPO default is on_policy_n_minibatch_iters = 45, so leaving
#  this at 45 would give every HAPPO agent the same number of epochs MAPPO's
#  shared policy gets -- at n_agents times the compute.  See
#  baselines/docs/happo.md, "budget".
HAPPO_EPOCHS="${HAPPO_EPOCHS:-45}"

COMMON=(
  "task=simple_ns/${HOST}"
  "task.ns_severity=${SIGMA}"
  "experiment.render=false"
  "experiment.checkpoint_at_end=true"
  "experiment.max_n_frames=${FRAMES}"
  "experiment.sampling_device=${SAMPLING_DEVICE}"
  "experiment.train_device=${TRAIN_DEVICE}"
  "experiment.loggers=${LOGGERS}"
  "experiment.on_policy_n_envs_per_worker=${ENVS}"
  "experiment.off_policy_n_envs_per_worker=${ENVS}"
  "experiment.on_policy_collected_frames_per_batch=${BATCH}"
  "experiment.off_policy_collected_frames_per_batch=${BATCH}"
  "experiment.on_policy_minibatch_size=${MINIBATCH}"
)

# $1 row name, rest: overrides
run_one () {
  local name="$1"; shift
  for seed in ${SEEDS}; do
    local dir="${OUT_ROOT}/${HOST}/sigma${SIGMA}/s${seed}/${name}"
    if [ "${LIST}" = "1" ]; then
      echo "python simple_ns/run.py ${COMMON[*]} seed=${seed} experiment.save_folder=${dir} $* ${EXTRA}"
      continue
    fi
    # simple_ns/run.py nests a seed_<N> level under save_folder and BenchMARL
    # nests a TIMESTAMPED folder under that, so the finished marker is
    # <dir>/seed_*/*/checkpoints.  The second glob is the pre-seed_<N> layout,
    # so a sweep that ran before that change still counts as finished.
    if compgen -G "${dir}/seed_*/*/checkpoints" > /dev/null || compgen -G "${dir}/*/checkpoints" > /dev/null; then
      echo "== skip ${name} seed ${seed}"
      continue
    fi
    echo "== ${HOST} | sigma ${SIGMA} | seed ${seed} | ${name}"
    mkdir -p "${dir}"
    if [ "${KEEP_GOING}" = "1" ]; then
      # shellcheck disable=SC2086
      if ! python simple_ns/run.py \
        "${COMMON[@]}" "seed=${seed}" "experiment.save_folder=${dir}" "$@" ${EXTRA}
      then
        echo "!! FAILED: ${name} seed ${seed} -- continuing (KEEP_GOING=1)"
        BASELINE_FAILURES="${BASELINE_FAILURES} ${name}/s${seed}"
      fi
    else
      # shellcheck disable=SC2086
      python simple_ns/run.py \
        "${COMMON[@]}" "seed=${seed}" "experiment.save_folder=${dir}" "$@" ${EXTRA}
    fi
  done
}

# ===========================================================================
#  The table.  One function per baseline; the name of the function is the name
#  of the row.  GROUP picks a class, ONLY picks rows by name.
# ===========================================================================

# --- reference rows -------------------------------------------------------
#  Stock MAPPO, not baselines: the numbers every baseline row is read against.
#  NOT in ALL_GROUPS, because the PACT ladder (scripts/ns_*.sh) already runs
#  them -- re-running them here would burn a queue slot to reproduce a number
#  you have.  Ask for them explicitly if you want them in this OUT_ROOT:
#
#      GROUP=reference bash scripts/run_baselines.sh
#      ONLY=mappo_b0   bash scripts/run_baselines.sh
GROUP_reference="mappo_blind mappo_pact mappo_b0"

mappo_b0 () {  # the no-NS reference the percentages are of
  run_one mappo_b0 algorithm=mappo task.ns_severity=0 task.pact_enabled=false
}
mappo_blind () {
  run_one mappo_blind algorithm=mappo task.pact_enabled=false
}
mappo_pact () {
  run_one mappo_pact algorithm=mappo task.pact_enabled=true
}

# --- B1  trust region / monotone improvement -------------------------------
GROUP_b1="happo hasac"

happo () {
  #  share_policy_params=false is REQUIRED: with one shared policy there is no
  #  sequential decomposition left to measure, and the algorithm raises.
  #  The optimiser budget is multiplied by n_agents so each agent gets the same
  #  number of epochs MAPPO's policy gets.
  run_one happo algorithm=happo \
    experiment.share_policy_params=false \
    "experiment.on_policy_n_minibatch_iters=$((NAGENTS * HAPPO_EPOCHS))"
}
hasac () {
  run_one hasac algorithm=hasac experiment.share_policy_params=false
}

# --- B2  memory: "just add an RNN" -----------------------------------------
#  ONE ARM.  ippo_gru is the same baseline on an independent critic; dropped.
#      ONLY=ippo_gru bash scripts/run_baselines.sh
GROUP_b2="mappo_gru"

mappo_gru () {
  #  BenchMARL's own GRU model as the POLICY; the critic stays an MLP, so the
  #  row prices memory in the actor. This is R-MAPPO's construction.
  run_one mappo_gru algorithm=mappo model=layers/gru task.pact_enabled=false
}
ippo_gru () {
  run_one ippo_gru algorithm=ippo model=layers/gru task.pact_enabled=false
}

# --- B3  graph / communication ---------------------------------------------
#  NOT IN THE DEFAULT SWEEP.  Removed from ALL_GROUPS on request: this row is
#  not wanted in the standard run.  It is still reachable explicitly --
#      GROUP=b3 bash scripts/run_baselines.sh
#      ONLY=mappo_gnn bash scripts/run_baselines.sh
#  -- exactly like the `reference` class, so nothing is deleted and the row can
#  be produced later without editing anything.
GROUP_b3="mappo_gnn"

mappo_gnn () {
  #  ns_observe_prev_action puts each agent's own last action in its node
  #  features, so the graph carries the NEIGHBOURS' ACTIONS -- which is what
  #  BASELINES.md B3 asks for, and what an observation-only GNN would not do.
  run_one mappo_gnn algorithm=mappo model=layers/gnn \
    task.ns_observe_prev_action=true task.pact_enabled=false
}

# --- B4  mean field ---------------------------------------------------------
#  ONE ARM.  mfac_team is the reference implementation's team mean (self
#  included) rather than the paper's mean over N(j); dropped.
#      ONLY=mfac_team bash scripts/run_baselines.sh
GROUP_b4="mfac"

mfac () {
  run_one mfac algorithm=mfac task.ns_observe_prev_action=true
}
mfac_team () {
  #  the reference implementation's team mean (self included) rather than the
  #  paper's mean over N(j)
  run_one mfac_team algorithm=mfac task.ns_observe_prev_action=true \
    algorithm.include_self=true
}

# --- B5  agent modelling ----------------------------------------------------
#  NOT IN THE DEFAULT SWEEP.  Measured on balance at 300 workers: ~1340 s per
#  iteration, i.e. ~37 h for one seed of one row.  LIAM stores a history window
#  PER TRANSITION, so the on-policy buffer grows by
#  batch * n_agents * history_len * obs_dim and every optimiser call moves it.
#  Still reachable by name:
#      GROUP=b5 bash scripts/run_baselines.sh
#      ONLY=liam bash scripts/run_baselines.sh
#  If you do run it, cut the cost at the window rather than the batch:
#      ONLY=liam EXTRA="algorithm.history_len=10" bash scripts/run_baselines.sh
GROUP_b5="liam"

#  NOTE, for liam and for rma: the history window is stored per transition, so
#  the on-policy buffer grows by BATCH * n_agents * history_len * obs_dim.  At
#  the defaults that is a few hundred MB on top of everything else.  If it does
#  not fit, lower BATCH for these rows -- the window length is the published
#  value and the batch size is not.
liam () {
  run_one liam algorithm=liam task.ns_observe_prev_action=true
}

# --- B6  non-stationary RL with an observed context -------------------------
GROUP_b6="lcpo"

lcpo () {
  #  LCPO takes ONE policy step and ONE value step per rollout, on the whole
  #  rollout.  minibatch == batch and one iteration reproduce that; leaving
  #  BenchMARL's 45 x N optimiser calls in place would train the critic
  #  hundreds of times per rollout against LCPO's once.
  run_one lcpo algorithm=lcpo \
    task.ns_observe_driver=true task.pact_enabled=false \
    "experiment.on_policy_minibatch_size=${BATCH}" \
    "experiment.on_policy_n_minibatch_iters=1" \
    "algorithm.ood_window=${BATCH}"
}

# --- B8  meta-RL / online system identification -----------------------------
#  ONE ARM.  uposi is the same file with predict_latent=false (identify the
#  parameters instead of a latent); dropped.  RMA is the one B8 names first and
#  the closer methodological neighbour of PACT.
#      ONLY=uposi bash scripts/run_baselines.sh
#  NOT IN THE DEFAULT SWEEP -- TOO SLOW.  RMA stores a 50-frame history window PER TRANSITION, so the
#  on-policy buffer grows by batch * n_agents * 50 * obs_dim and every
#  optimiser call moves it -- the same cost that made liam unusable.
#  Reachable by name:
#      GROUP=b8 bash scripts/run_baselines.sh
#      ONLY=rma bash scripts/run_baselines.sh
#  Cut the cost at the WINDOW, not the batch, if you run it:
#      ONLY=rma EXTRA="algorithm.history_len=10" bash scripts/run_baselines.sh
GROUP_b8="rma"

rma () {
  run_one rma algorithm=rma \
    task.ns_observe_driver=true task.ns_observe_prev_action=true \
    task.pact_enabled=false \
    "algorithm.phase1_frames=$((FRAMES / 3))"
}
uposi () {
  #  the same machinery with UP-OSI's choice: no latent, identify the model
  #  parameters themselves
  run_one uposi algorithm=rma algorithm.predict_latent=false \
    task.ns_observe_driver=true task.ns_observe_prev_action=true \
    task.pact_enabled=false \
    "algorithm.phase1_frames=$((FRAMES / 3))"
}

# --- B9  robustness ---------------------------------------------------------
GROUP_b9="dr_sigma ernie"

dr_sigma () {
  #  BASELINES.md B9's MUST row.  sigma is redrawn per episode from [0, 3]; the
  #  committed severity is IGNORED during training, which is the point.
  #  Evaluate the checkpoint at the committed sigma afterwards -- see
  #  baselines/docs/dr_sigma.md, which gives the one-line evaluation command.
  run_one dr_sigma algorithm=mappo task.pact_enabled=false \
    task.ns_dr_enabled=true task.ns_dr_low=0.0 task.ns_dr_high=3.0
}
ernie () {
  run_one ernie algorithm=ernie task.pact_enabled=false
}

# --- B10  the non-learning compensators -------------------------------------
#  TWO ARMS, because eso and rls_raw are DIFFERENT compensators, not two
#  settings of one.  rls_raw_w (the same regression handed the declared
#  operator W) is dropped.
#      ONLY=rls_raw_w bash scripts/run_baselines.sh
GROUP_b10="eso rls_raw"

eso () {
  run_one eso algorithm=mappo task.ns_baseline=eso task.pact_enabled=false
}
rls_raw () {
  run_one rls_raw algorithm=mappo task.ns_baseline=rls_raw task.pact_enabled=false
}
rls_raw_w () {
  #  the stronger version: the same raw per-peer regression, but handed the
  #  declared operator W, which is public
  run_one rls_raw_w algorithm=mappo task.ns_baseline=rls_raw \
    task.rls_raw_use_operator=true task.pact_enabled=false
}

# --- D.4  the information grants -------------------------------------------
GROUP_grants="oracle_driver_blind"

oracle_driver_blind () {
  #  BASELINES.md D.4: a stock learner handed the true driver.  Not a published
  #  method -- an upper bound on what observing the context alone can buy.
  run_one oracle_driver_blind algorithm=mappo \
    task.ns_observe_driver=true task.pact_enabled=false
}

# ===========================================================================
#  THE EXTRA BASELINES (X1 .. X6).  See baselines/README_EXTRA.md and
#  baselines/docs/<name>.md.  Run them with scripts/run_extra_baselines.sh.
# ===========================================================================

# --- X1  prior-free black-box NS-RL: detect and restart ---------------------
#  Gerogiannis, Huang, Veeravalli, arXiv 2410.13772.
#  ONE ARM.  qcd_glr is Algorithm 3, the method the paper recommends.
#  qcd_rr (Algorithm 2) and qcd_none (the wrapper's control) are dropped.
#      ONLY="qcd_rr qcd_none" bash scripts/run_extra_baselines.sh
GROUP_x1="qcd_glr"

qcd_glr () {
  #  Algorithm 3 (QCD+): Bernoulli GLR on the reward stream, full restart on
  #  an alarm.  BLIND -- no driver, no latent, no coupling.  That is the point.
  run_one qcd_glr algorithm=qcd algorithm.detector=glr task.pact_enabled=false
}
qcd_rr () {
  #  Algorithm 2 (RR): restart at i.i.d. Geometric times.  The paper's own
  #  order-optimal baseline, and what its Theorem 4 says MASTER degenerates
  #  into.  The glr-vs-rr pair IS the paper's central comparison.
  run_one qcd_rr algorithm=qcd algorithm.detector=random task.pact_enabled=false
}
qcd_none () {
  #  The wrapper's control: same file, restarts off.  Any difference between
  #  this and mappo_blind is a bug in the wrapper, not a result.
  run_one qcd_none algorithm=qcd algorithm.detector=none task.pact_enabled=false
}

# --- X2  deep fictitious play for continuous mean field games ---------------
#  Magnino, Shao, Wu, Shen, Lauriere, arXiv 2510.22158 (NeurIPS 2025).
#  ONE ARM.  dedafp reports pibar_K, which is what Algorithm 3 returns.
#  dedafp_br (the last best response) is dropped.
#      ONLY=dedafp_br bash scripts/run_extra_baselines.sh
GROUP_x2="dedafp"

dedafp () {
  #  ns_observe_time is REQUIRED: the average policy and the conditional
  #  normalising flow are both functions of t, and a finite-horizon mean field
  #  equilibrium is not defined without it.
  #  The LAST fictitious-play iteration deploys pibar, which is what
  #  Algorithm 3 returns -- so the reported number is the equilibrium
  #  policy's.
  run_one dedafp algorithm=dedafp \
    task.ns_observe_time=true task.pact_enabled=false \
    experiment.share_policy_params=false
}
dedafp_br () {
  #  The same run reporting the last BEST RESPONSE instead of the average.
  #  Say which of the two the table shows; they are different policies.
  run_one dedafp_br algorithm=dedafp algorithm.deploy_average_last=false \
    task.ns_observe_time=true task.pact_enabled=false \
    experiment.share_policy_params=false
}

# --- X3  independent learning in performative Markov potential games --------
#  Sahitaj, Sasnauskas, Yalin, Mandal, Radanovic, arXiv 2504.20593.
#  ONE ARM.  ipga is the paper's proximal/projected update and steps through
#  BenchMARL's own optimiser, so it is the safer of the two to run first.
#  inpg (the natural-gradient variant, with the stronger last-iterate result)
#  is dropped -- run it once ipga has produced a number.
#      ONLY=inpg bash scripts/run_extra_baselines.sh
GROUP_x3="ipga"

ipga () {
  #  share_policy_params=false is REQUIRED in spirit: these are INDEPENDENT
  #  learners and the convergence results are about what N separate updates do
  #  to the joint policy.  With one shared policy "independent" is vacuous and
  #  the algorithm warns.
  run_one ipga algorithm=ipga algorithm.variant=ipga \
    experiment.share_policy_params=false task.pact_enabled=false
}
inpg () {
  #  The natural-gradient variant: ONE Fisher-preconditioned step per rollout,
  #  so the optimiser call count is set to 1.  Leaving BenchMARL's 45 x N in
  #  place would train the critic hundreds of times per rollout against INPG's
  #  once -- the same correction the LCPO row needs, for the same reason.
  run_one inpg algorithm=ipga algorithm.variant=inpg \
    experiment.share_policy_params=false task.pact_enabled=false \
    "experiment.on_policy_minibatch_size=${BATCH}" \
    "experiment.on_policy_n_minibatch_iters=1"
}

# --- X4  wavelet predictive representations (off-policy) --------------------
#  Wang, Li, He, Li, Bennis, Islam, Wang, arXiv 2510.04507.
#
#  NOTE: the context window is stored per transition, so the off-policy buffer
#  grows by memory_size * n_agents * time_steps * obs_dim.  At the stock
#  1M-transition buffer that does not fit; WISDOM_BUFFER shrinks it rather
#  than shortening the window, which is the published value.
#  ONE ARM.  wisdom carries CEMRL's decoder, which is the only version in
#  which the method can work; wisdom_release reproduces the released tree, in
#  which z collapses.  See baselines/docs/wisdom.md (A).
#      ONLY=wisdom_release bash scripts/run_extra_baselines.sh
#  NOT IN THE DEFAULT SWEEP -- TOO SLOW.  the context encoder reads a 30-frame window per transition and
#  runs one encoder pass per frame; off-policy, so it pays that on every
#  optimiser step rather than once per rollout.
#  Reachable by name:
#      GROUP=x4 bash scripts/run_extra_baselines.sh
#      ONLY=wisdom bash scripts/run_extra_baselines.sh
#      ONLY=wisdom EXTRA="algorithm.time_steps=10" bash scripts/run_extra_baselines.sh
GROUP_x4="wisdom"
WISDOM_BUFFER="${WISDOM_BUFFER:-100000}"

wisdom () {
  run_one wisdom algorithm=wisdom \
    task.ns_observe_prev_action=true task.ns_observe_prev_reward=true \
    task.pact_enabled=false \
    "experiment.off_policy_memory_size=${WISDOM_BUFFER}"
}
wisdom_release () {
  #  encoder_loss=kl_only reproduces the RELEASED tree exactly: its
  #  ReconstructionTrainer trains the encoder with the KL to the prior and
  #  nothing else, and there is no decoder anywhere in it.  z collapses.  Run
  #  it if a reviewer asks what the release does; do not read it as the method.
  #  See baselines/docs/wisdom.md.
  run_one wisdom_release algorithm=wisdom algorithm.encoder_loss=kl_only \
    task.ns_observe_prev_action=true task.ns_observe_prev_reward=true \
    task.pact_enabled=false \
    "experiment.off_policy_memory_size=${WISDOM_BUFFER}"
}

# --- X5  domain randomisation by entropy maximisation -----------------------
#  Tiboni, Klink, Peters, Tommasi, D'Eramo, Chalvatzaki, ICLR 2024.
#
#  SET DORAEMON_SUCCESS FROM YOUR OWN B0 ROW.  It is the return at which an
#  episode counts as solved and it is the only host-dependent number in the
#  method; the reference sets it per environment and has no default.  A
#  sensible choice is the MEDIAN return of the sigma=0 reference arm.
GROUP_x5="doraemon"
DORAEMON_SUCCESS="${DORAEMON_SUCCESS:-0.0}"

doraemon () {
  #  The B9 `dr_sigma` row with the range chosen by the method instead of by
  #  hand.  Same support, so the pair prices the CURRICULUM alone.  Evaluate
  #  the checkpoint at the committed sigma afterwards, exactly as
  #  baselines/docs/dr_sigma.md says for dr_sigma.
  run_one doraemon algorithm=doraemon task.pact_enabled=false \
    task.ns_dr_enabled=true task.ns_dr_dist=beta \
    task.ns_dr_low=0.0 task.ns_dr_high=3.0 \
    "algorithm.success_return=${DORAEMON_SUCCESS}"
}

# --- X6  MoE world model, with planning (off-policy) ------------------------
#  Zhao, Zhao, Xu, Fu, Chai, Zhu, Zhao, NeurIPS 2025.
#
#  THIS ROW IS SLOW.  Planning runs plan_iterations * horizon dynamics AND
#  reward forwards over (n_envs * num_samples) agent-token sets at EVERY
#  environment step.  M3W_ENVS drops the worker count rather than the
#  planner's published settings; raise it if you have the compute.
#  ONE ARM.  m3w plans, which is the method.  m3w_noplan is the ablation that
#  prices the search; dropped.
#      ONLY=m3w_noplan bash scripts/run_extra_baselines.sh
#  NOT IN THE DEFAULT SWEEP -- TOO SLOW.  MPPI planning runs 6 x 3 dynamics AND reward forwards through a
#  16-expert mixture at EVERY environment step, on top of a 9-frame window.
#  Reachable by name:
#      GROUP=x6 bash scripts/run_extra_baselines.sh
#      ONLY=m3w bash scripts/run_extra_baselines.sh
#      ONLY=m3w EXTRA="algorithm.use_plan=false" bash scripts/run_extra_baselines.sh
#  (use_plan=false drops the search and is much cheaper, but it is then
#  the ablation, not the method.)
GROUP_x6="m3w"
M3W_ENVS="${M3W_ENVS:-32}"
M3W_BATCH="${M3W_BATCH:-3200}"

m3w () {
  #  clip_grad_val=20 is the reference's own gradient clip on the world-model
  #  optimiser; the algorithm prints a warning if it is left at BenchMARL's 5.
  #  off_policy_init_random_frames is M3W's `warmup_steps`.
  run_one m3w algorithm=m3w \
    task.ns_observe_prev_action=true task.ns_observe_prev_reward=true \
    task.pact_enabled=false \
    "experiment.off_policy_n_envs_per_worker=${M3W_ENVS}" \
    "experiment.off_policy_collected_frames_per_batch=${M3W_BATCH}" \
    experiment.off_policy_init_random_frames=10000 \
    experiment.clip_grad_norm=true experiment.clip_grad_val=20
}
m3w_noplan () {
  #  use_plan=false acts with the actor directly.  The m3w-vs-m3w_noplan pair
  #  is what prices the SEARCH, with the same world model behind both.
  run_one m3w_noplan algorithm=m3w algorithm.use_plan=false \
    task.ns_observe_prev_action=true task.ns_observe_prev_reward=true \
    task.pact_enabled=false \
    "experiment.off_policy_n_envs_per_worker=${M3W_ENVS}" \
    "experiment.off_policy_collected_frames_per_batch=${M3W_BATCH}" \
    experiment.off_policy_init_random_frames=10000 \
    experiment.clip_grad_norm=true experiment.clip_grad_val=20
}

#  What a bare `run_baselines.sh` runs: the BASELINES.md rows only.  The stock
#  MAPPO reference rows are a class of their own and are deliberately absent --
#  see GROUP_reference above.  b3 (mappo_gnn) is absent for the same reason:
#  it is not wanted in the standard sweep.  Both are still reachable by name.
ALL_GROUPS="b1 b2 b4 b6 b9 b10 grants"

#  The EXTRA baselines (X1..X6).  A class of their own so that
#  `run_baselines.sh` is unchanged and `run_extra_baselines.sh` runs only the
#  new work.  See baselines/README_EXTRA.md.
EXTRA_GROUPS="x1 x2 x3 x5"

#  Every class the launcher will accept, including the ones outside the default.
KNOWN_GROUPS="reference b3 b5 b8 x4 x6 ${ALL_GROUPS} ${EXTRA_GROUPS}"
