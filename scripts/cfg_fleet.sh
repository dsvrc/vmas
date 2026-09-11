#!/usr/bin/env bash
#  cfg_fleet -- the N-scaling prediction (NS-4.2).
#
#  The coordination gap rises with controllable share because irreducible load
#  disappears while peer load does not.  Measured offline, no training:
#
#       8/40   2.9%       24/40  49.6%
#      16/40  26.7%       32/40  72.8%
#                         40/40  82.9%
#
#  So PACT's margin over blind MUST widen along this sweep.  No competing
#  credit-assignment method predicts that, which makes this the falsifiable
#  experiment rather than another horse race -- and a flat margin here is
#  evidence against the mechanism even if the headline row looks good.

source "$(dirname "$0")/_common.sh"
for n in 4 8 12 16; do
  run_config "fleet_n${n}" task.ns_severity=1.0 "task.n_agents=${n}"
done
