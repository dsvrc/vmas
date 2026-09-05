#!/usr/bin/env bash
#  The SLC / PACT pipeline, in the order PACT_PIPELINE_SPEC section 11 mandates.
#
#      bash pact2/run_pipeline.sh check     # arithmetic + Phase 0, no GPU, ~3 min
#      bash pact2/run_pipeline.sh smoke     # wiring, needs vmas, ~1 min
#      bash pact2/run_pipeline.sh certify   # the Part D gate output -- COMMIT IT
#      bash pact2/run_pipeline.sh arms      # the training ladder
#      bash pact2/run_pipeline.sh arm pact  # a single arm
#
#  Every arm inherits ONE shared block of overrides.  If those differ between
#  arms the comparison is meaningless, so they live in exactly one place and
#  that block is deliberately tiny.  Opt in with DEVICE=cuda, FRAMES=..,
#  LOGGERS=.., SEED=.. or EXTRA=".." -- which apply to every arm at once.
#
#  The host configuration is not part of this contribution and arms MUST NOT
#  differ in it.

set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${TASK:-vmas_slc/sampling}"
ALGO="${ALGO:-ippo}"
FRAMES="${FRAMES:-3000000}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-0}"
LOGGERS="${LOGGERS:-[csv]}"
EXTRA="${EXTRA:-}"
OUT="${OUT:-runs/slc}"

COMMON=(
  "task=${TASK}"
  "algorithm=${ALGO}"
  "experiment.render=false"
  "experiment.checkpoint_at_end=true"
  "experiment.max_n_frames=${FRAMES}"
  "experiment.sampling_device=${DEVICE}"
  "experiment.train_device=${DEVICE}"
  "experiment.loggers=${LOGGERS}"
  "seed=${SEED}"
)

run_arm () {
  local name="$1"; shift
  local dir="${OUT}/${name}"
  if [ -d "${dir}/checkpoints" ]; then
    echo "== skip ${name} (already has checkpoints)"
    return 0
  fi
  echo "== arm ${name}"
  mkdir -p "${dir}"
  # shellcheck disable=SC2086
  python pact2/run.py "${COMMON[@]}" "experiment.save_folder=${dir}" "$@" ${EXTRA}
}

# ---------------------------------------------------------------------------
#  the arms.  Each isolates exactly one thing; see pact2/README.md section 6.
# ---------------------------------------------------------------------------
arm_b0        () { run_arm b0        task.slc_harm_enabled=false; }
arm_blind     () { run_arm blind; }
arm_ff        () { run_arm ff        task.pact_enabled=true task.pact_mode=ff; }
arm_pact      () { run_arm pact      task.pact_enabled=true; }
arm_peeronly  () { run_arm peer_only task.pact_enabled=true task.pact_ff_gain=0 \
                                     task.pact_own_gain=0; }
arm_noloop    () { run_arm noloop    task.pact_enabled=true \
                                     task.slc_phi_reads_executed=false; }
arm_tracegate () { run_arm trace     task.pact_enabled=true task.pact_gate=trace; }
arm_nowindup  () { run_arm nowindup  task.pact_enabled=true \
                                     task.pact_p_max_mult=1e30; }
arm_level     () { run_arm level     task.pact_enabled=true task.pact_mode=level; }
arm_placebo   () { run_arm placebo   task.slc_p_quiet=1.0; }
arm_n1        () { run_arm n1        task.n_agents=1; }

case "${1:-}" in
  check)
    python pact2/check_plumbing.py
    python pact2/selfcheck.py
    python pact2/ceiling.py   --csv "${OUT}/ceiling.csv"
    python pact2/calibrate.py --csv "${OUT}/calibration.csv"
    ;;
  smoke)
    python pact2/smoke_test.py
    ;;
  certify)
    # Commit this output BEFORE running any method.  The history is the
    # evidence that the environment was not retuned after seeing a method fail.
    mkdir -p "${OUT}"
    { python pact2/check_plumbing.py; python pact2/selfcheck.py; python pact2/ceiling.py; python pact2/smoke_test.py; } \
      | tee "${OUT}/certificates.txt"
    echo "wrote ${OUT}/certificates.txt -- git add it now"
    ;;
  b0)       arm_b0 ;;
  arms)
    arm_b0; arm_blind; arm_ff; arm_pact; arm_peeronly
    ;;
  ablations)
    arm_noloop; arm_tracegate; arm_nowindup; arm_level; arm_placebo; arm_n1
    ;;
  sweep-trust)
    for t in 0.0 0.2 0.4 0.8 1.2 1.6; do
      run_arm "trust_${t}" task.pact_enabled=true "task.pact_max_trust=${t}"
    done
    ;;
  sweep-n)
    for n in 3 6 9 12; do
      run_arm "n_${n}" task.pact_enabled=true "task.n_agents=${n}"
      run_arm "n_${n}_ff" task.pact_enabled=true task.pact_mode=ff "task.n_agents=${n}"
    done
    ;;
  arm)
    "arm_${2:?usage: run_pipeline.sh arm <name>}"
    ;;
  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
