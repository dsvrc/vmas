#!/usr/bin/env bash
#  ns_ladder -- the severity ladder, both arms.  THE HEADLINE FIGURE.
#
#  Blind falls monotonically; PACT recovers a large and shrinking fraction of it.
#  Measured offline on the shipped heuristic before any training (3 seeds):
#
#      sigma   blind    PACT    oracle (free answer)
#        0    100.0%   100.0%    100.0%
#        1     87.8%   103.8%    108.1%
#      1.5     64.5%    90.0%    109.7%
#        2     50.3%    78.9%    111.7%
#        3     37.1%    62.7%    110.3%
#
#  The oracle staying near 110% at every row is what says none of these is past
#  sigma*: the ceiling is full recovery throughout, so every gap below it is the
#  method's to close rather than the environment's fault.
source "$(dirname "$0")/ns_common.sh"
for s in 0 1 1.5 2 3; do
  run_config "sigma${s}" "task.ns_severity=${s}"
done
