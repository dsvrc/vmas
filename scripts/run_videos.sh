#!/usr/bin/env bash
#  Re-run blind and PACT at one severity WITH RENDERING, to get videos.
#
#      xvfb-run -a bash scripts/run_videos.sh
#
#  Why a re-run rather than a replay: every other launcher here passes
#  `experiment.render=false`, so a finished run's `videos/` folder is empty and
#  there are no frames in its logs to recover.  `scripts/make_videos.py`
#  re-renders from a checkpoint without retraining and is the cheap option; this
#  script is the "just run it again" option, and it produces a run that is
#  otherwise IDENTICAL to the plotted ones -- same host, same severity, same
#  batch, same frame budget -- so the video is of the arm in the table.
#
#  XVFB IS NOT OPTIONAL.  VMAS renders through pyglet, which needs a display.
#  Without `xvfb-run` this dies with a GL error rather than a Python one.
#
#  Overridable:
#      SIGMA=1.0 SEEDS="0 1" bash scripts/run_videos.sh
#      VIDEO_EVERY=3000000 bash scripts/run_videos.sh   # one video, at the end
#      ONLY=pact_sev3 bash scripts/run_videos.sh
#
#  Output, per row and seed:
#      runs/videos/<host>/sigma<S>/s<seed>/<row>/seed_<N>/<run>/<run>/videos/

#  ---------------------------------------------------------------------
#  A GL CONTEXT, NOT A MONITOR.
#
#  VMAS renders with pyglet, and pyglet needs an X connection to create an
#  OpenGL context even for an INVISIBLE window -- it calls
#  `pyglet.window.Window(visible=False)`, which still has to talk to a display
#  server.  So "I do not need to see it" does not remove the requirement; it is
#  the GL driver that needs somewhere to draw, not you.  Nothing is ever shown:
#  the frames go straight into an mp4 you download.
#
#  VMAS's own render() docstring prescribes the second branch below.  This
#  picks whichever of the three works on the node, so you do not have to:
#
#    1. xvfb-run          an in-memory X server, torn down automatically
#    2. Xvfb :99          the same thing started by hand
#    3. PYGLET_HEADLESS   pyglet's EGL path, no X server at all
#
#  VIDEO_NO_XVFB=1 skips all of this if you have your own arrangement.
#  ---------------------------------------------------------------------
if [ -z "${DISPLAY:-}" ] && [ "${VIDEO_NO_XVFB:-0}" != "1" ] && [ "${LIST:-0}" != "1" ]; then
  if command -v xvfb-run > /dev/null 2>&1; then
    echo "[videos] no DISPLAY -> re-running under xvfb-run (an in-memory"
    echo "         display; nothing is shown, the mp4 is still written)"
    exec xvfb-run -a --server-args="-screen 0 1400x900x24" bash "$0" "$@"
  elif command -v Xvfb > /dev/null 2>&1; then
    echo "[videos] no DISPLAY, no xvfb-run -> starting Xvfb :99 by hand"
    Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
    _XVFB_PID=$!
    trap 'kill ${_XVFB_PID} 2> /dev/null || true' EXIT
    export DISPLAY=:99.0
    sleep 2
  else
    echo "[videos] no DISPLAY and no Xvfb -> trying pyglet's EGL headless path."
    echo "         If this fails with a GL/EGL error, get Xvfb on the node:"
    echo "             module load xorg-server      # or"
    echo "             conda install -c conda-forge xorg-x11-server-xvfb-cos7-x86_64"
    export PYGLET_HEADLESS=1
  fi
fi

OUT_ROOT="${OUT_ROOT:-runs/videos}"
SIGMA="${SIGMA:-3.0}"
SEEDS="${SEEDS:-0}"

source "$(dirname "$0")/baselines_common.sh"

#  How often an evaluation -- and therefore a video -- is taken, in frames.
#  BenchMARL evaluates when `total_frames % evaluation_interval == 0`, plus
#  once at iteration 0, so 600000 over a 3M budget gives five videos and an
#  untrained one.  The untrained clip is worth keeping: it is the "before" of
#  the before/after.
VIDEO_EVERY="${VIDEO_EVERY:-600000}"
VIDEO_EPISODES="${VIDEO_EPISODES:-1}"

#  VMAS renders WORLD 0 only, so more evaluation episodes buys a less noisy
#  mean and a slower render, not more footage.
RENDER=(
  "experiment.render=true"
  "experiment.evaluation=true"
  "experiment.evaluation_episodes=${VIDEO_EPISODES}"
  "experiment.evaluation_interval=${VIDEO_EVERY}"
  "experiment.evaluation_deterministic_actions=true"
  "experiment.checkpoint_at_end=true"
  "experiment.loggers=[csv]"
)

blind_sev3 () {
  run_one blind_sev3 algorithm=mappo task.pact_enabled=false "${RENDER[@]}"
}
pact_sev3 () {
  run_one pact_sev3 algorithm=mappo task.pact_enabled=true "${RENDER[@]}"
}

rows="${ONLY:-blind_sev3 pact_sev3}"

echo "== video run =="
echo "host        ${HOST}"
echo "sigma       ${SIGMA}"
echo "seeds       ${SEEDS}"
echo "frames      ${FRAMES}"
echo "rows        ${rows}"
echo "video every ${VIDEO_EVERY} frames, ${VIDEO_EPISODES} episode(s)"
echo "output      ${OUT_ROOT}/${HOST}/sigma${SIGMA}"
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
echo "Videos:"
find "${OUT_ROOT}" -path "*/videos/*" -type f 2>/dev/null | sort || true
