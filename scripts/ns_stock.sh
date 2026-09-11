#!/usr/bin/env bash
#  ns_stock -- sigma = 0.  B0, and the proof the layer is a no-op when it says so.
#
#  At sigma = 0 the disturbance is EXACTLY zero at every driver value, so the
#  PACT target is exactly zero, beta stays exactly zero and the correction is
#  exactly zero: the two arms must come out bit-identical.  If they do not, the
#  compensator is acting on a disturbance that does not exist, and that is worth
#  catching before it flatters a result.
source "$(dirname "$0")/ns_common.sh"
run_config stock task.ns_severity=0
