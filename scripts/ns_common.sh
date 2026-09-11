#!/usr/bin/env bash
#  Shared launcher for every simple_ns configuration.
#
#  P-10.1 and I.4: every arm goes through ONE entry point with severity supplied
#  from outside the method, and the host configuration is IDENTICAL across arms.
#  If the arms differ in anything but the flags named in each ns_*.sh, the
#  comparison measures the wrong thing.
#
#  Overridable from the environment, applied to every arm at once:
#      DEVICE=cuda ITERS=60 SEEDS="0 1 2 3 4" bash scripts/ns_main.sh
#      HOST=sampling bash scripts/ns_main.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"

#  Separate knobs on purpose.  These hosts are small (3-4 agents, a handful of
#  small tensor ops per step), so sampling is kernel-LAUNCH bound on a GPU while
#  training is exactly what a GPU is for.  Measure both before the sweep:
#      SAMPLING_DEVICE=cpu TRAIN_DEVICE=cuda bash scripts/ns_main.sh
SAMPLING_DEVICE="${SAMPLING_DEVICE:-$DEVICE}"
TRAIN_DEVICE="${TRAIN_DEVICE:-$DEVICE}"

#  WHICH HOST.  The dial and the method are the same objects in all four, so a
#  difference between hosts is a difference in the task and nothing else.
#
#      transport   THE HEADLINE.  Blind keeps 50.3% of B0 at the committed
#                  severity and the free-answer ceiling is still full recovery.
#      sampling    second host, same layer
#      navigation  cheapest; debug here
#      balance     NOT MONOTONE under the dial.  See its yaml.  Not a headline.
HOST="${HOST:-transport}"

ITERS="${ITERS:-40}"
BATCH="${BATCH:-30000}"
FRAMES="${FRAMES:-$((ITERS * BATCH))}"
SEEDS="${SEEDS:-0 1 2 3 4}"

#  BenchMARL picks the action space itself, so absolute returns are NOT
#  comparable between the discrete and continuous groups.  Compare blind against
#  pact WITHIN an algorithm, which is the comparison that carries the claim.
ALGOS="${ALGOS:-ippo mappo iddpg maddpg isac masac iql qmix vdn}"

ENVS="${ENVS:-300}"
LOGGERS="${LOGGERS:-[csv]}"
EXTRA="${EXTRA:-}"
OUT_ROOT="${OUT_ROOT:-runs/simple_ns}"

COMMON=(
  "task=simple_ns/${HOST}"
  "experiment.render=false"
  "experiment.checkpoint_at_end=true"
  "experiment.max_n_frames=${FRAMES}"
  "experiment.sampling_device=${SAMPLING_DEVICE}"
  "experiment.train_device=${TRAIN_DEVICE}"
  "experiment.loggers=${LOGGERS}"
  "experiment.on_policy_n_envs_per_worker=${ENVS}"
  "experiment.off_policy_n_envs_per_worker=${ENVS}"
  # Both batches, so ITERS means the same number of iterations whether the
  # algorithm is on- or off-policy.  Left at BenchMARL's default the off-policy
  # arms would run many times the iterations of the on-policy ones for the same
  # frame budget, and the wall clocks would not be comparable.
  "experiment.on_policy_collected_frames_per_batch=${BATCH}"
  "experiment.off_policy_collected_frames_per_batch=${BATCH}"
  "experiment.on_policy_minibatch_size=4096"
)

# $1 config, $2 seed, $3 algo, $4 arm, rest: task overrides
run_one () {
  local cfg="$1" seed="$2" algo="$3" arm="$4"; shift 4
  local dir="${OUT_ROOT}/${HOST}/${cfg}/s${seed}/${algo}_${arm}"
  # BenchMARL nests a TIMESTAMPED folder under save_folder, so the finished
  # marker is <dir>/*/checkpoints, never <dir>/checkpoints.
  if compgen -G "${dir}/*/checkpoints" > /dev/null; then
    echo "== skip ${HOST} ${cfg} s${seed} ${algo} ${arm}"
    return 0
  fi
  echo "== ${HOST} | ${cfg} | seed ${seed} | ${algo} | ${arm}"
  mkdir -p "${dir}"
  # shellcheck disable=SC2086
  python simple_ns/run.py \
    "algorithm=${algo}" "${COMMON[@]}" "seed=${seed}" \
    "experiment.save_folder=${dir}" "$@" ${EXTRA}
}

#  Seed-major: seed 0 runs every algorithm and every arm, then seed 1.  An
#  interrupted sweep then has complete, comparable seeds rather than a partial
#  column for every one.
run_config () {
  local cfg="$1"; shift
  for seed in ${SEEDS}; do
    for algo in ${ALGOS}; do
      run_one "${cfg}" "${seed}" "${algo}" blind "$@" task.pact_enabled=false
      run_one "${cfg}" "${seed}" "${algo}" pact  "$@" task.pact_enabled=true
    done
  done
}

#  P-7.1 wiring check, ONCE rather than per seed: trust forced to 0 must be
#  bit-identical to the blind arm.  A proof that the wrapper is honest, not an
#  arm.
run_floor_check () {
  local cfg="$1"; shift
  run_one "${cfg}" 0 "${ALGOS%% *}" pactoff "$@" \
    task.pact_enabled=true task.pact_trust=0.0
}
