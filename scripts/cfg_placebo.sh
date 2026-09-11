#!/usr/bin/env bash
#  cfg_placebo -- NS-2.5.  The cheapest credibility you will ever buy.
#
#  ns_wet_fraction = 0 makes the driver return an exact zero at every step, so
#  the dial is PROVABLY inert for every sigma -- not merely small.  Run at
#  sigma = 3, three times the physical anchor: every arm must still come out
#  identical to cfg_stock.
#
#  A reviewer alleging a rigged knob then has to explain why the rig switches
#  itself off.

source "$(dirname "$0")/_common.sh"
run_config placebo task.ns_severity=3.0 task.ns_wet_fraction=0.0
