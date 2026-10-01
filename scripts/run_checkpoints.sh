#!/usr/bin/env bash
#  Train the two arms a video is wanted of, and leave CHECKPOINTS behind.
#
#      bash scripts/run_checkpoints.sh
#
#  Nothing here renders, and nothing here needs a display, xvfb, ffmpeg or any
#  other video dependency: rendering on a compute node needs an X/EGL context
#  the node does not have, which is what the earlier attempt died on.  The
#  split is instead
#
#      server  ->  checkpoints  ->  scripts/pack_checkpoints.sh  ->  scp
#      laptop  ->  scripts/make_videos.py  ->  mp4
#
#  so the only thing produced here is weights, and every arm is otherwise
#  IDENTICAL to the ones in the table -- same host, same severity, same batch,
#  same frame budget -- so the clip is of the arm that produced the numbers.
#
#  Checkpointing itself comes from scripts/baselines_common.sh (CKPT_EVERY,
#  KEEP_CKPT, exclude_buffer_from_checkpoint); read the block above COMMON
#  there for what is written and why.  Defaults give six checkpoints per run
#  over a 3M budget, which is the before/after of training, at a few MB each.
#
#  Overridable:
#      SIGMA=1.0 SEEDS="0 1" bash scripts/run_checkpoints.sh
#      CKPT_EVERY=1500000 bash scripts/run_checkpoints.sh   # just start / end
#      ONLY=pact_sev3 bash scripts/run_checkpoints.sh
#      LIST=1 bash scripts/run_checkpoints.sh               # print, run nothing
#
#  Output, per row and seed:
#      runs/clips/<host>/sigma<S>/s<seed>/<row>/seed_<N>/<run>/
#          config.pkl                          <- half of what a replay needs
#          checkpoints/checkpoint_<frames>.pt  <- the other half

OUT_ROOT="${OUT_ROOT:-runs/clips}"
SIGMA="${SIGMA:-3.0}"
SEEDS="${SEEDS:-0}"

source "$(dirname "$0")/baselines_common.sh"

blind_sev3 () {
  run_one blind_sev3 algorithm=mappo task.pact_enabled=false
}
pact_sev3 () {
  run_one pact_sev3 algorithm=mappo task.pact_enabled=true
}

rows="${ONLY:-blind_sev3 pact_sev3}"

echo "== checkpoint run (no rendering) =="
echo "host         ${HOST}"
echo "sigma        ${SIGMA}"
echo "seeds        ${SEEDS}"
echo "frames       ${FRAMES}"
echo "rows         ${rows}"
echo "checkpoint   every $(ckpt_interval "${BATCH}") frames, keeping ${KEEP_CKPT}"
echo "output       ${OUT_ROOT}/${HOST}/sigma${SIGMA}"
echo

for row in ${rows}; do
  if ! declare -F "${row}" > /dev/null; then
    echo "unknown row '${row}'" >&2
    exit 2
  fi
  "${row}"
done

echo
echo "== done =="
echo "Checkpoints:"
find "${OUT_ROOT}" -name "checkpoint_*.pt" -type f 2>/dev/null | sort || true
echo
echo "Next, to get the videos onto your laptop:"
echo "    bash scripts/pack_checkpoints.sh ${OUT_ROOT}"
echo "    # then, on the laptop, after unpacking:"
echo "    python scripts/make_videos.py <run folder> --out videos"
