#!/usr/bin/env bash
#  cfg_main -- sigma = 1, the committed operating point.  THE HEADLINE.
#
#  sigma = 1 is anchored at the Highway Capacity Manual's heavy-rain capacity
#  adjustment factor: capacity falls to 0.86 of dry, a published constant, so
#  this row needs no defence.
#
#  The fleet is the task config's own: 16 controllable of 40 total, URB's 40%
#  operating point, chosen on the Part C decomposition BEFORE any method code
#  existed (NS-4.1):
#
#      irreducible 55.7%   own 17.6%   PEER 26.7%
#
#  26.7% of the loading excess is peer-caused, which is the CEILING on what any
#  coordination method can recover here.  Anything PACT wins must come out of
#  that budget, and anything it wins beyond it is a bug.
#
#  Do NOT override n_agents here.  The fleet size and the controllable share are
#  a committed choice; cfg_fleet.sh is where they vary, and it varies the share
#  while holding the total at 40.

source "$(dirname "$0")/_common.sh"
run_config main task.ns_severity=1.0
run_floor_check main task.ns_severity=1.0
