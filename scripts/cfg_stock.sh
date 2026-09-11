#!/usr/bin/env bash
#  cfg_stock -- sigma = 0.  Performance on the undisturbed environment.
#
#  NS-2.1: at sigma = 0 the capacity multiplier is exactly 1.0 at every driver
#  value, so harm == 1.0 bit for bit and this IS the stock host.  It is the B0
#  every later number is read against, and it is also the check that the layer
#  is a no-op when it says it is.
#
#  Both arms are run even though PACT has nothing to compensate.  It is not a
#  formality: at sigma=0 the target y is exactly 0, so beta stays exactly 0,
#  every prediction is identical, the shift is exactly 1.0 and the two arms are
#  bit-identical.  If they are NOT, the compensator is acting on a disturbance
#  that does not exist -- and that is a bug worth catching before it flatters a
#  result.  Verified offline; this is the in-simulator confirmation.

source "$(dirname "$0")/_common.sh"
run_config stock task.ns_severity=0
