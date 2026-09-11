#!/usr/bin/env bash
#  ns_placebo -- NS-2.5.  The cheapest credibility you will ever buy.
#
#  ns_wet_fraction = 0 makes the driver return an exact zero at every step, so
#  the dial is PROVABLY inert for every sigma rather than merely small.  Run at
#  three times the operating point: every arm must still come out identical to
#  ns_stock.  A reviewer alleging a rigged knob then has to explain why the rig
#  switches itself off.
source "$(dirname "$0")/ns_common.sh"
run_config placebo task.ns_severity=6.0 task.ns_wet_fraction=0.0
