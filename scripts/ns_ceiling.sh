#!/usr/bin/env bash
#  ns_ceiling -- the free-answer controller, as a trained arm.
#
#  ANT_complete_story.md 2.2: a learner cannot beat a controller that already
#  knows the answer, so where this fails, nothing can.  Run it at the operating
#  point to put the ceiling on the same axes as the result, and at the stress
#  severity to show the row is still not past sigma*.
#
#  Not a baseline and not a competitor -- it is handed the true hidden load.  Label
#  it that way in every figure it appears in.
source "$(dirname "$0")/ns_common.sh"
ALGOS="${ALGOS:-ippo}"
for s in 1 2 3; do
  run_one "ceiling_sigma${s}" 0 "${ALGOS%% *}" oracle     "task.ns_severity=${s}" task.pact_enabled=true task.pact_oracle=true
done
