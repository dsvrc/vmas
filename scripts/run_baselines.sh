#!/usr/bin/env bash
#  THE entry point for the BASELINES.md baselines.
#
#      bash scripts/run_baselines.sh                 # every row, every class
#      LIST=1 bash scripts/run_baselines.sh          # print the plan, run nothing
#      GROUP="b1 b10" bash scripts/run_baselines.sh  # one or more classes
#      ONLY="happo lcpo" bash scripts/run_baselines.sh
#      KEEP_GOING=1 bash scripts/run_baselines.sh    # don't let one bad row
#                                                    # kill an overnight sweep
#
#  SMOKE TEST FIRST.  Nothing in this tree has ever been run end to end; the
#  point of a smoke run is to reach every algorithm's construction banner and
#  one training iteration, which is where a wiring error shows up:
#
#      FRAMES=12000 BATCH=6000 ENVS=60 SEEDS=0 OUT_ROOT=runs/smoke \
#        EXTRA="experiment.off_policy_n_optimizer_steps=20 experiment.evaluation=false" \
#        KEEP_GOING=1 bash scripts/run_baselines.sh
#
#  A different OUT_ROOT means the smoke runs do not mark the real ones as
#  already finished.
#
#  Class names: b1 b2 b3 b4 b5 b6 b8 b9 b10 grants
#
#  A bare run does the BASELINES.md rows and nothing else.  The stock MAPPO
#  reference rows -- mappo_blind, mappo_pact, mappo_b0 -- are the class
#  `reference` and are NOT in the default: the PACT ladder (scripts/ns_*.sh)
#  already produces them, and the baseline rows are read against those numbers.
#  `GROUP=reference bash scripts/run_baselines.sh` if you want them here too.
#  Everything else -- host, severity, seeds, devices, frame budget -- is set in
#  scripts/baselines_common.sh and overridable from the environment:
#
#      HOST=transport SIGMA=2.0 SEEDS="0 1 2 3 4" DEVICE=cuda \
#        bash scripts/run_baselines.sh
#
#  Run `python baselines/verify.py` first: it checks every config and every
#  published formula offline, in about a second, and it has already caught one
#  wrong observer gain.

source "$(dirname "$0")/baselines_common.sh"

GROUP="${GROUP:-${ALL_GROUPS}}"
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
  *" mappo_blind "*) ;;
  *) echo "reference   NOT re-run (mappo_blind / mappo_pact / mappo_b0)."
     echo "            Read the baseline rows against the numbers the PACT"
     echo "            ladder already produced; GROUP=reference to redo them." ;;
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
echo "Each run wrote pact_debug.csv beside itself. Read it in the order the"
echo "columns are documented in benchmarl/environments/simple_ns/common.py:"
echo "did the dial fire, how big was it, and only then, did the row win."

if [ -n "${BASELINE_FAILURES}" ]; then
  echo
  echo "!! these rows FAILED and were skipped (KEEP_GOING=1):"
  for failed in ${BASELINE_FAILURES}; do echo "     ${failed}"; done
  exit 1
fi
