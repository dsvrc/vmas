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

#  SPLIT THE DEVICES.  These default to DEVICE, but they are separate knobs and
#  on this host the best setting is usually NOT the same for both.
#
#  lanelet_flow's per-step work is many SMALL tensor ops -- VMAS calls
#  process_action / observation / reward once per agent, and the vehicle dynamics
#  is an rk4 over (B,) state -- so on a GPU it is kernel-LAUNCH bound rather than
#  throughput bound, and 600 envs is not wide enough to amortise the launches.
#  Training, by contrast, is 675 optimizer steps per iteration and is exactly
#  what a GPU is for.  So try:
#
#      SAMPLING_DEVICE=cpu TRAIN_DEVICE=cuda bash scripts/cfg_main.sh
#
#  and compare against DEVICE=cuda on a two-iteration run before committing the
#  sweep.  Which wins depends on the machine; measure, do not assume.
SAMPLING_DEVICE="${SAMPLING_DEVICE:-$DEVICE}"
TRAIN_DEVICE="${TRAIN_DEVICE:-$DEVICE}"

#  WHICH HOST.  Both carry the identical medium -- same map, same capacities,
#  same operator, same dial, same PACT -- and differ only in the vehicle model.
#
#      road_ns/lanelet_flow   the sweep host.   0.155 ms/frame at N=16/600 envs
#      road_ns/road_traffic   SigmaRL's own.  173     ms/frame at N=40/16  envs
#
#  Measured, CPU, via vmas.make_env.  1.2M frames is 3 minutes against 58 HOURS.
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
#  per FRAME falls as the batch widens.  That was NOT true of road_traffic, whose
#  dominant cost is a python loop over (env, agent) pairs and therefore does not
#  amortise: if you set TASK=road_ns/road_traffic, drop ENVS as well.
#
#  600 envs against BATCH=60000 is 100 sequential simulator steps per iteration.
#
#  Raising ENVS keeps cutting wall clock -- measured 0.155 ms/frame at 600,
#  0.095 at 2400, 0.075 at 4800 -- but it is NOT free: at a fixed frame budget
#  it buys fewer DISTINCT timesteps.  1.2M frames is 2000 sequential steps at
#  600 envs (10 episodes per env at max_steps=200) and only 250 at 4800 (1.25
#  episodes).  On-policy learning needs sequential coverage, not 4800
#  near-identical copies of the same 250 steps.  Raise ENVS and ITERS together,
#  or not at all.
ENVS="${ENVS:-600}"
LOGGERS="${LOGGERS:-[csv]}"
EXTRA="${EXTRA:-}"
OUT_ROOT="${OUT_ROOT:-runs/road_ns}"

COMMON=(
  "task=${TASK}"
  "experiment.render=false"
  "experiment.checkpoint_at_end=true"
  "experiment.max_n_frames=${FRAMES}"
  "experiment.sampling_device=${SAMPLING_DEVICE}"
  "experiment.train_device=${TRAIN_DEVICE}"
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
