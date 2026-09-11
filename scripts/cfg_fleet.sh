#!/usr/bin/env bash
#  cfg_fleet -- the N-scaling prediction (NS-4.2).
#
#  The coordination gap rises with CONTROLLABLE SHARE because irreducible load
#  disappears while peer load does not.  Measured offline, no training
#  (road_ns/report.py, committed to runs/road_ns_ceiling.csv):
#
#       8 of 40   irred 79.8   own 17.3   PEER  2.9%
#      16 of 40   irred 55.7   own 17.6   PEER 26.7%
#      24 of 40   irred 32.6   own 17.7   PEER 49.6%
#      32 of 40   irred  9.4   own 17.8   PEER 72.8%
#      40 of 40   irred  0.0   own 17.1   PEER 82.9%
#
#  So PACT's margin over blind MUST widen along this sweep.  No competing
#  credit-assignment method predicts that, which makes this the falsifiable
#  experiment rather than another horse race -- and a flat margin here is
#  evidence against the mechanism even if the headline row looks good.
#
#  THE TOTAL FLEET IS HELD AT 40.  What varies is how many of those 40 are
#  controllable; the rest are background demand.  This used to sweep
#  `task.n_agents` alone, which shrinks the whole fleet instead: every row was
#  then 100% controllable, the irreducible share was 0 in all of them, and the
#  mechanism the comment quoted -- irreducible load disappearing -- was not the
#  one being exercised.  The numbers it printed belonged to a different
#  experiment.

source "$(dirname "$0")/_common.sh"
TOTAL="${TOTAL:-40}"
for k in 8 16 24 32 40; do
  run_config "fleet_k${k}" task.ns_severity=1.0 \
    "task.n_agents=${k}" "task.flow_n_background=$((TOTAL - k))"
done
