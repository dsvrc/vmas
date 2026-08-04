#!/usr/bin/env bash
#
#  The complete Navigation-PCW + PACT pipeline, in order.
#
#      bash pact/run_pipeline.sh check     # sanity, no GPU time            (~2 min)
#      bash pact/run_pipeline.sh b0        # the baseline, needed by everything else
#      bash pact/run_pipeline.sh phase1    # certify sigma*                 (~10 min)
#      bash pact/run_pipeline.sh arms      # the 6 Phase-2 training arms
#      bash pact/run_pipeline.sh report    # the final table
#      bash pact/run_pipeline.sh all       # all of the above, in sequence
#
#  Individual arms can be run on their own:
#      bash pact/run_pipeline.sh arm blind_ippo
#
#  Every arm MUST share the same experiment overrides (see COMMON below) or the
#  comparison is meaningless.  Change them in one place, never per arm.
#
#  Everything not listed below is left at BenchMARL's own defaults on purpose:
#  the host configuration is not part of this contribution, and an arm that
#  differs from the others in anything but its algorithm/model/pact_* flags is
#  not comparable to them.
#
#  Optional environment variables (unset = BenchMARL default, nothing passed):
#      RUNS=<dir>      where checkpoints go            (default: ./runs)
#      DEVICE=cuda     train/sample/buffer device      (default: unset -> cpu)
#      FRAMES=<int>    experiment.max_n_frames         (default: unset -> 3e6)
#      LOGGERS=...     experiment.loggers              (default: csv)
#      EXTRA="a=1 b=2" any further hydra overrides, applied to EVERY arm
#      EVAL_DEVICE=... device for the eval scripts     (default: cpu)
#      EPISODES=<int>  envs per evaluation cell        (default: 40)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUNS="${RUNS:-$REPO_ROOT/runs}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EPISODES="${EPISODES:-40}"
# BenchMARL defaults to [csv,wandb] and WandbLogger raises on import if wandb is
# absent, so csv-only is the working default rather than a preference.
# LOGGERS=csv,wandb re-enables it once wandb is installed and configured.
LOGGERS="${LOGGERS:-csv}"

# ---------------------------------------------------------------------------
# Shared settings.  IDENTICAL for every arm -- that is the point.
# These three are the minimum the pipeline needs:
#   render=false           no display on a compute node
#   checkpoint_at_end      there is nothing to evaluate without it
#   save_folder (per arm)  so the report can find the checkpoints
# ---------------------------------------------------------------------------
COMMON=(
  "task=vmas_ns/navigation_pcw"
  "experiment.render=false"
  "experiment.checkpoint_at_end=true"
  "experiment.loggers=[$LOGGERS]"
)

if [[ -n "${DEVICE:-}" ]]; then
  COMMON+=(
    "experiment.sampling_device=$DEVICE"
    "experiment.train_device=$DEVICE"
    "experiment.buffer_device=$DEVICE"
  )
fi
if [[ -n "${FRAMES:-}" ]]; then
  COMMON+=("experiment.max_n_frames=$FRAMES")
fi
if [[ -n "${EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON+=($EXTRA)
fi

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# BenchMARL creates <save_folder>/<generated_name>/checkpoints/, and refuses to
# create save_folder itself (mkdir parents=False), so it must exist first.
launch() {
  local name="$1"; shift
  local folder="$RUNS/$name"
  if [[ -d "$folder" ]] && find "$folder" -name 'checkpoint_*.pt' | grep -q .; then
    log "$name -- already has a checkpoint, skipping (delete $folder to redo)"
    return 0
  fi
  mkdir -p "$folder"
  log "$name"
  ( set -x
    python pact/run.py "${COMMON[@]}" "experiment.save_folder=$folder" "$@" \
      2>&1 | tee "$folder/train.log"
  )
  latest_ckpt "$folder" >/dev/null || fail "$name produced no checkpoint"
}

latest_ckpt() {
  local found
  found="$(find "$1" -name 'checkpoint_*.pt' -printf '%T@ %p\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)"
  [[ -n "$found" ]] || return 1
  echo "$found"
}

require_ckpt() {
  latest_ckpt "$RUNS/$1" 2>/dev/null \
    || fail "no checkpoint under $RUNS/$1 -- run 'bash pact/run_pipeline.sh $2' first"
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

stage_check() {
  log "arithmetic unit tests (torch only, no simulator)"
  python test/test_pact_pcw.py
  log "Phase 0 calibration (reproduces pact/README.md section 2)"
  python pact/calibrate.py
  log "integration smoke test (needs vmas + torchrl)"
  python pact/smoke_test.py
}

# B0 is the undisturbed baseline.  It is SEVERITY-INDEPENDENT, so this single
# checkpoint is reused at every point of the Phase-1 sweep and as the
# normaliser for the final table.  Train it first.
stage_b0() {
  launch b0 "algorithm=ippo" "task.ns_severity=0"
}

stage_phase1() {
  local ckpt; ckpt="$(require_ckpt b0 b0)"
  log "Phase 1 -- certify sigma* from $ckpt"
  python pact/phase1_certify.py "$ckpt" \
    --episodes "$EPISODES" --device "$EVAL_DEVICE" \
    2>&1 | tee "$RUNS/phase1.log"
  echo "Saved to $RUNS/phase1.log"
}

# ---------------------------------------------------------------------------
# The Phase-2 arms.  Only the algorithm / model / pact_* flags differ.
#
# MEMORYLESS arms (cheap).  PACT itself needs no recurrence: the mechanism is
# env-side and the host is untouched.  What a memoryless policy cannot do is
# modulate beta with the driver phase, so it settles on one constant gain -- the
# "constant-beta" tier, worth roughly half the gap in the PACT reference.  These
# five already make a complete result: blind vs PACT vs the ceiling.
# ---------------------------------------------------------------------------
arm_blind_ippo()  { launch blind_ippo  "algorithm=ippo"; }
arm_blind_mappo() { launch blind_mappo "algorithm=mappo"; }
arm_pact_ippo()   { launch pact_ippo   "algorithm=ippo"  "task.pact_enabled=true"; }
arm_pact_mappo()  { launch pact_mappo  "algorithm=mappo" "task.pact_enabled=true"; }
arm_pact_ctde()   { launch pact_ctde   "algorithm=mappo_ctde" "task.pact_enabled=true"; }
# The ceiling compensates with the TRUE deflection, so the environment it sees is
# stationary and memory would buy it nothing -- MLP on purpose, not to save time.
arm_ceiling()     { launch ceiling     "algorithm=ippo" \
                                       "task.pact_enabled=true" "task.pact_oracle=true"; }

# ---------------------------------------------------------------------------
# RECURRENT arms (expensive).  Run these only after the memoryless ones show
# which host wins, and only if beta comes out phase-blind (flat across the
# cycle), which is the one thing recurrence is there to fix.
#
# BenchMARL sets sequence_length = collected_frames_per_batch / n_envs_per_worker
# and unrolls it in a Python loop, so the default 10 workers means a 600-step
# unroll per optimizer step.  Use
#     EXTRA="experiment.on_policy_n_envs_per_worker=100"
# to make each sequence exactly one 60-step episode: ~10x faster and no
# sequence straddles an episode boundary.
# ---------------------------------------------------------------------------
arm_blind_gru()     { launch blind_gru     "algorithm=ippo" "model=layers/gru"; }
arm_pact_gru()      { launch pact_gru      "algorithm=ippo" "model=layers/gru" \
                                           "task.pact_enabled=true"; }
arm_pact_ctde_gru() { launch pact_ctde_gru "algorithm=mappo_ctde" "model=layers/gru" \
                                           "task.pact_enabled=true"; }

ARMS_MEMORYLESS=(blind_ippo blind_mappo pact_ippo pact_mappo pact_ctde ceiling)
ARMS_RECURRENT=(blind_gru pact_gru pact_ctde_gru)

stage_arms_fast() { for a in "${ARMS_MEMORYLESS[@]}"; do "arm_$a"; done; }
stage_arms_rnn()  { for a in "${ARMS_RECURRENT[@]}";  do "arm_$a"; done; }
stage_arms()      { stage_arms_fast; stage_arms_rnn; }

stage_report() {
  local args=(--b0 "$(require_ckpt b0 b0)")
  local any=0
  for name in "${ARMS_MEMORYLESS[@]}" "${ARMS_RECURRENT[@]}"; do
    if ckpt="$(latest_ckpt "$RUNS/$name" 2>/dev/null)"; then
      args+=(--arm "$name=$ckpt")
      any=1
    else
      echo "  (skipping $name -- no checkpoint)"
    fi
  done
  [[ $any -eq 1 ]] || fail "no arm checkpoints found under $RUNS"
  log "Phase 2 -- arm report"
  python pact/evaluate_arms.py "${args[@]}" \
    --episodes "$EPISODES" --device "$EVAL_DEVICE" \
    2>&1 | tee "$RUNS/report.log"
  echo "Saved to $RUNS/report.log"
}

# Optional: the irreducibility certificate as a *training* run.  The smoke test
# already proves the environments are byte-identical, so these two only need to
# be long enough to compare learning curves.
stage_n1() {
  launch n1_severity_on  "algorithm=ippo" "task.n_agents=1"
  launch n1_severity_off "algorithm=ippo" "task.n_agents=1" "task.ns_severity=0"
  echo "The two curves must be indistinguishable: at N=1 there is no channel."
}

mkdir -p "$RUNS"

case "${1:-all}" in
  check)     stage_check ;;
  b0)        stage_b0 ;;
  phase1)    stage_phase1 ;;
  arms-fast) stage_arms_fast ;;    # memoryless only -- a complete result on its own
  arms-rnn)  stage_arms_rnn ;;     # the recurrent tier
  arms)      stage_arms ;;
  arm)       "arm_${2:?usage: run_pipeline.sh arm <name>}" ;;
  report)    stage_report ;;
  n1)        stage_n1 ;;
  all)       stage_check; stage_b0; stage_phase1; stage_arms_fast; stage_report ;;
  *)         fail "unknown stage '${1}' (check|b0|phase1|arms-fast|arms-rnn|arms|arm|report|n1|all)" ;;
esac

log "done: ${1:-all}"
