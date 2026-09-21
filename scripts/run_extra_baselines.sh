#!/usr/bin/env bash
#  THE entry point for the SIX EXTRA baselines (X1 .. X6).
#
#      bash scripts/run_extra_baselines.sh                 # all six classes
#      LIST=1 bash scripts/run_extra_baselines.sh          # print the plan
#      GROUP="x1 x5" bash scripts/run_extra_baselines.sh   # one or more classes
#      ONLY="qcd_glr ipga" bash scripts/run_extra_baselines.sh
#      KEEP_GOING=1 bash scripts/run_extra_baselines.sh    # overnight sweep
#
#  SMOKE TEST FIRST.  Nothing here has been run end to end; the point of a
#  smoke run is to reach every algorithm's construction banner and one training
#  iteration, which is where a wiring error shows up:
#
#      FRAMES=12000 BATCH=6000 ENVS=60 SEEDS=0 OUT_ROOT=runs/smoke_extra \
#        EXTRA="experiment.off_policy_n_optimizer_steps=20 experiment.evaluation=false" \
#        KEEP_GOING=1 bash scripts/run_extra_baselines.sh
#
#  A different OUT_ROOT means the smoke runs do not mark the real ones as
#  already finished.
#
#  The classes:
#      x1  QCD+ / RR      prior-free NS-RL: detect and restart   (arXiv 2410.13772)
#      x2  DEDA-FP        deep fictitious play, continuous MFGs  (arXiv 2510.22158)
#      x3  IPGA / INPG    independent learning, performative MPGs(arXiv 2504.20593)
#      x4  WISDOM         wavelet predictive representations     (arXiv 2510.04507)
#      x5  DORAEMON       DR by entropy maximisation             (arXiv 2311.01885)
#      x6  M3W            MoE world model, with planning         (NeurIPS 2025)
#
#  TWO NUMBERS YOU MUST SET FOR A NEW HOST, because they are the only
#  host-dependent quantities in the six:
#
#      DORAEMON_SUCCESS=<return>   the return at which an episode counts as
#                                  solved.  Take the MEDIAN return of your
#                                  sigma=0 (B0) reference arm.  Left at 0 the
#                                  curriculum may never start or start
#                                  instantly; watch doraemon_success.
#      algorithm.reward_low/high   the per-step reward range the QCD detector
#                                  maps onto [0,1] (conf/algorithm/qcd.yaml).
#                                  Watch qcd_clip_frac.
#
#  Everything else -- host, severity, seeds, devices, frame budget -- is set in
#  scripts/baselines_common.sh and overridable from the environment:
#
#      HOST=transport SIGMA=2.0 SEEDS="0 1 2 3 4" DEVICE=cuda \
#        DORAEMON_SUCCESS=-42.0 bash scripts/run_extra_baselines.sh
#
#  Run `python baselines/verify.py` first: it checks every config and every
#  published formula offline, in about a second.

source "$(dirname "$0")/baselines_common.sh"

GROUP="${GROUP:-${EXTRA_GROUPS}}"
ONLY="${ONLY:-}"

if [ "${LIST}" != "1" ]; then
  echo "== offline verification =="
  python baselines/verify.py
  python simple_ns/check_plumbing.py
  echo
fi

rows=""
if [ -n "${ONLY}" ]; then
  rows="${ONLY}"
else
  for group in ${GROUP}; do
    var="GROUP_${group}"
    if [ -z "${!var:-}" ]; then
      echo "unknown class '${group}'; expected one or more of: ${KNOWN_GROUPS}" >&2
      exit 2
    fi
    rows="${rows} ${!var}"
  done
fi

echo "== plan =="
echo "host        ${HOST}"
echo "sigma       ${SIGMA}"
echo "seeds       ${SEEDS}"
echo "frames      ${FRAMES}"
echo "output      ${OUT_ROOT}/${HOST}/sigma${SIGMA}"
echo "rows        ${rows}"
case " ${rows} " in
  *" doraemon "*)
    echo "doraemon    success_return=${DORAEMON_SUCCESS}"
    if [ "${DORAEMON_SUCCESS}" = "0.0" ]; then
      echo "            !! that is the DEFAULT, not a measured value. It is the"
      echo "               return at which an episode counts as solved and it"
      echo "               decides the whole curriculum. Set DORAEMON_SUCCESS"
      echo "               from your B0 row's median return."
    fi ;;
esac
case " ${rows} " in
  *" m3w "*|*" m3w_noplan "*)
    echo "m3w         ${M3W_ENVS} workers, ${M3W_BATCH} frames/batch (planning is"
    echo "            expensive; this is deliberately smaller than the others)" ;;
esac
echo

for row in ${rows}; do
  if ! declare -F "${row}" > /dev/null; then
    echo "unknown row '${row}'; see scripts/baselines_common.sh" >&2
    exit 2
  fi
  "${row}"
done

echo
echo "== done =="
echo "Each run wrote pact_debug.csv beside itself, and each algorithm logs its"
echo "own diagnostics -- qcd_restarts, dedafp_sl_nll, ipga_dist_move,"
echo "wisdom_td, doraemon_entropy, m3w_dynamics. Read those BEFORE the return:"
echo "a row whose method never engaged is a row about nothing. Each"
echo "baselines/docs/<name>.md has a 'what to watch' table."

if [ -n "${BASELINE_FAILURES}" ]; then
  echo
  echo "!! these rows FAILED and were skipped (KEEP_GOING=1):"
  for failed in ${BASELINE_FAILURES}; do echo "     ${failed}"; done
  exit 1
fi
