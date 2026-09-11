#!/usr/bin/env bash
#  Shared launcher for every road_ns configuration.
#
#  P-10.1 and I.4: every arm goes through ONE entry point with severity supplied
#  from outside the method, and the host configuration is IDENTICAL across arms.
#  That is why this block lives in exactly one file and is deliberately tiny --
#  if the arms differ in anything but the flags named in each cfg_*.sh, the
#  comparison measures the wrong thing.
#
#  Overridable from the environment, applied to every arm at once:
#      DEVICE=cuda ITERS=40 SEEDS="0 1 2 3 4" bash scripts/cfg_main.sh
#      TASK=road_ns/road_traffic ENVS=32 ITERS=4 bash scripts/cfg_provenance.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"

#  WHICH HOST.  Both carry the identical medium -- same map, same capacities,
#  same operator, same dial, same PACT -- and differ only in the vehicle model.
#
#      road_ns/lanelet_flow   the sweep host.   0.24 ms/frame at N=16/600 envs
#      road_ns/road_traffic   SigmaRL's own.  173    ms/frame at N=40/16  envs
#
#  Measured, CPU, via vmas.make_env.  1.2M frames is 5 minutes against 58 HOURS.
#  road_traffic is kept for the provenance row, run at low N (cfg_provenance.sh).
TASK="${TASK:-road_ns/lanelet_flow}"

#  Budget is set in ITERATIONS, because that is what you watch tick by.
#
#  frames = ITERS * BATCH, and BATCH/ENVS is the number of sequential simulator
#  steps per iteration.  Raising BATCH does NOT reduce collection work -- the
#  same frames are the same steps -- it reduces the number of optimizer rounds.
#  The lever that actually shortens a run is ITERS.
ITERS="${ITERS:-20}"
BATCH="${BATCH:-60000}"
FRAMES="${FRAMES:-$((ITERS * BATCH))}"

SEEDS="${SEEDS:-0 1 2 3 4}"
#  Every algorithm this BenchMARL checkout ships, minus `ensemble` (a wrapper,
#  not an algorithm) and `mappo_ctde` (left over from the superseded PCW work).
#  There is no HAPPO in this checkout.
#
#  BenchMARL picks the action space itself: prefer_continuous_actions=True gives
#  continuous to ippo/mappo/isac/masac, iddpg/maddpg are continuous-only, and
#  iql/qmix/vdn are discrete-only.  Absolute returns are therefore NOT
#  comparable between the discrete and continuous groups -- compare blind
#  against pact WITHIN an algorithm, which is the comparison that carries the
#  claim anyway.
ALGOS="${ALGOS:-ippo mappo iddpg maddpg isac masac iql qmix vdn}"

#  For lanelet_flow the per-step work is vectorised over the batch, so the cost
#  per FRAME falls as the batch widens and more parallel envs is close to free.
#
#  That was NOT true of road_traffic, and the comment here used to claim it was.
#  Its dominant cost is a python reset loop over (env, agent) pairs, which
#  scales linearly with envs and therefore does not amortise at all -- measured
#  30 ms/frame at 64 envs against 56 at 16, a 1.9x gain for 4x the envs.
#  If you set TASK=road_ns/road_traffic, drop ENVS as well.
ENVS="${ENVS:-600}"
LOGGERS="${LOGGERS:-[csv]}"
EXTRA="${EXTRA:-}"
OUT_ROOT="${OUT_ROOT:-runs/road_ns}"

COMMON=(
  "task=${TASK}"
  "experiment.render=false"
  "experiment.checkpoint_at_end=true"
  "experiment.max_n_frames=${FRAMES}"
  "experiment.sampling_device=${DEVICE}"
  "experiment.train_device=${DEVICE}"
  "experiment.loggers=${LOGGERS}"
  "experiment.on_policy_n_envs_per_worker=${ENVS}"
  "experiment.off_policy_n_envs_per_worker=${ENVS}"
  # Both batches, so ITERS means the same number of iterations whether the
  # algorithm is on- or off-policy.  Left at BenchMARL's 6000 default, the
  # off-policy arms would run 10x the iterations of the on-policy ones for the
  # same frame budget, and the wall clocks would not be comparable.
  "experiment.on_policy_collected_frames_per_batch=${BATCH}"
  "experiment.off_policy_collected_frames_per_batch=${BATCH}"
  "experiment.on_policy_minibatch_size=4096"
)

# $1 config name, $2 seed, $3 algo, $4 arm, rest: task overrides
run_one () {
  local cfg="$1" seed="$2" algo="$3" arm="$4"; shift 4
  local dir="${OUT_ROOT}/${cfg}/s${seed}/${algo}_${arm}"
  # BenchMARL nests a TIMESTAMPED run folder under save_folder, so the finished
  # marker is `<dir>/*/checkpoints`, never `<dir>/checkpoints`.  Checking the
  # latter meant the skip never fired and every re-run started from scratch and
  # left another folder behind.
  if compgen -G "${dir}/*/checkpoints" > /dev/null; then
    echo "== skip ${cfg} s${seed} ${algo} ${arm} (already has checkpoints)"
    return 0
  fi
  echo "== ${cfg} | seed ${seed} | ${algo} | ${arm}"
  mkdir -p "${dir}"
  # shellcheck disable=SC2086
  python road_ns/run.py \
    "algorithm=${algo}" \
    "${COMMON[@]}" \
    "seed=${seed}" \
    "experiment.save_folder=${dir}" \
    "$@" ${EXTRA}
}

#  Seed-major, as asked: seed 0 runs EVERY algorithm and both arms, then seed 1,
#  and so on.  That ordering means an interrupted sweep still has complete,
#  comparable seeds rather than a partial column for every one.
#
#  $1 config name, rest: task overrides shared by both arms of this config.
run_config () {
  local cfg="$1"; shift
  for seed in ${SEEDS}; do
    for algo in ${ALGOS}; do
      run_one "${cfg}" "${seed}" "${algo}" blind "$@" task.pact_enabled=false
      run_one "${cfg}" "${seed}" "${algo}" pact  "$@" task.pact_enabled=true
    done
  done
}

#  P-7.1 wiring check, run ONCE rather than per seed: trust forced to 0 must be
#  bit-identical to the blind arm.  It is a proof that the wrapper is honest,
#  not a third arm.
run_floor_check () {
  local cfg="$1"; shift
  run_one "${cfg}" 0 "${ALGOS%% *}" pactoff "$@" \
    task.pact_enabled=true task.pact_trust=0.0
}
