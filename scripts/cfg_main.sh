#!/usr/bin/env bash
#  cfg_main -- sigma = 1, full fleet.  THE HEADLINE.
#
#  sigma = 1 is anchored at the Highway Capacity Manual's heavy-rain capacity
#  adjustment factor: capacity falls to 0.86 of dry, a published constant, so
#  this row needs no defence.
#
#  n_agents = 40 is path_to_loop's own maximum and the operating point chosen on
#  the Part C decomposition BEFORE any method code existed (NS-4.1):
#
#      irreducible 0.0%   own 17.1%   PEER 82.9%      loading u = 0.167
#
#  82.9% of the loading excess is peer-caused, which is the ceiling on what any
#  coordination method can recover here.  Anything PACT wins must come out of
#  that budget, and anything it wins beyond it is a bug.

source "$(dirname "$0")/_common.sh"
run_config main task.ns_severity=1.0 task.n_agents=40
run_floor_check main task.ns_severity=1.0 task.n_agents=40
