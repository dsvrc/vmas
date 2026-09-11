#!/usr/bin/env bash
#  ns_cells -- THE CLASSIFICATION EXPERIMENT.  Four arms, one flag each.
#
#  Same host, same driver, same severity, same reward, same method.  What varies
#  is (i) where the drift enters and (ii) whether the method may use the peer
#  channels:
#
#      coupled_full        (C) + all channels     the method
#      coupled_intercept   (C) + intercept only   a per-agent adaptive bias
#      direct_full         (B) + all channels
#      direct_intercept    (B) + intercept only
#
#  The prediction, and what would falsify it:
#
#    * on (B) the disturbance is the same for every agent, so the regression puts
#      all of it in the intercept and `intercept` should lose NOTHING;
#    * on (C) it differs per agent, so a fleet-average cannot represent it and
#      `intercept` should lose most of the recovery.
#
#  Note what is NOT predicted.  "PACT fails on (B)" would be the wrong claim and
#  this experiment would refute it -- the intercept tracks a level shift perfectly
#  well.  The claim is that on (B) nothing about the problem is multi-agent, so a
#  single-agent adaptive bias suffices; on (C) the peer channels are load-bearing.
#  That is the pair the classification rests on.
source "$(dirname "$0")/ns_common.sh"
run_config coupled_full      task.ns_direct=false task.pact_channels=full
run_config coupled_intercept task.ns_direct=false task.pact_channels=intercept
run_config direct_full       task.ns_direct=true  task.pact_channels=full
run_config direct_intercept  task.ns_direct=true  task.pact_channels=intercept
