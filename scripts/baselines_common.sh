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
    # BenchMARL nests a TIMESTAMPED folder under save_folder, so the finished
    # marker is <dir>/*/checkpoints, never <dir>/checkpoints.
    if compgen -G "${dir}/*/checkpoints" > /dev/null; then
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

# --- reference rows, so the baselines have something to be compared with ---
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
GROUP_b2="mappo_gru ippo_gru"

mappo_gru () {
  #  BenchMARL's own GRU model as the POLICY; the critic stays an MLP, so the
  #  row prices memory in the actor. This is R-MAPPO's construction.
  run_one mappo_gru algorithm=mappo model=layers/gru task.pact_enabled=false
}
ippo_gru () {
  run_one ippo_gru algorithm=ippo model=layers/gru task.pact_enabled=false
}

# --- B3  graph / communication ---------------------------------------------
GROUP_b3="mappo_gnn"

mappo_gnn () {
  #  ns_observe_prev_action puts each agent's own last action in its node
  #  features, so the graph carries the NEIGHBOURS' ACTIONS -- which is what
  #  BASELINES.md B3 asks for, and what an observation-only GNN would not do.
  run_one mappo_gnn algorithm=mappo model=layers/gnn \
    task.ns_observe_prev_action=true task.pact_enabled=false
}

# --- B4  mean field ---------------------------------------------------------
GROUP_b4="mfac mfac_team"

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
GROUP_b8="rma uposi"

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
GROUP_b10="eso rls_raw rls_raw_w"

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

ALL_GROUPS="reference b1 b2 b3 b4 b5 b6 b8 b9 b10 grants"
