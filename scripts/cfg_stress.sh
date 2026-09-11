#!/usr/bin/env bash
#  cfg_stress -- sigma = 3.  BEYOND-PHYSICAL.
#
#  Three times the HCM anchor: capacity falls to 0.58 of dry at the storm peak
#  and 10.5% is removed over the cycle.  NS-2.4 requires this to be labelled a
#  beyond-physical stress test EVERYWHERE it appears, including in figures.
#  It is not a headline number and must never be quoted as one.
#
#  Its use is that the harm is far larger here than at sigma = 1, so if the
#  mechanism works at all it should be most visible in this row.

source "$(dirname "$0")/_common.sh"
run_config stress task.ns_severity=3.0
