#!/usr/bin/env bash
#  cfg_provenance -- the same medium on SigmaRL's own vmas/road_traffic.
#
#  One row, not a sweep.  Its job is to show that the dial, the harm channel and
#  PACT behave the same way on the published scenario as on lanelet_flow, so the
#  cheap host is a vehicle-model substitution and not a different experiment.
#
#  It is SLOW -- measured 173 ms/frame at N=40/16 envs on CPU, against 0.24 for
#  lanelet_flow -- because road_traffic's per-agent geometry (an O(N^2) interX
#  loop with a device sync per pair, five boundary distances per agent, the
#  short-term path resampling) is most of its 4035 lines.  So: few agents, few
#  envs, few iterations, one seed, one algorithm.  Budget hours, not minutes.
#
#  road_traffic has no background traffic, so it is 100% controllable by
#  construction and its decomposition row is the 8/8 one, not 8/40.

export TASK="${TASK:-road_ns/road_traffic}"
export ENVS="${ENVS:-32}"      # its cost does NOT amortise over envs; see _common.sh
export ITERS="${ITERS:-4}"
export BATCH="${BATCH:-6400}"
export SEEDS="${SEEDS:-0}"
export ALGOS="${ALGOS:-ippo}"

source "$(dirname "$0")/_common.sh"
run_config provenance task.ns_severity=1.0 task.n_agents=8
run_floor_check provenance task.ns_severity=1.0 task.n_agents=8
