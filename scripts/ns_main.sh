#!/usr/bin/env bash
#  ns_main -- THE HEADLINE.  The committed operating point, both arms.
#
#  transport at sigma = 2: the blind arm keeps 50.3% of B0 while the free-answer
#  ceiling is 111.7%, so the row measures the METHOD and not the environment.
#  Calibrated before any training -- see simple_ns/calibrate.py and the table in
#  the task yaml.
source "$(dirname "$0")/ns_common.sh"
run_config main
run_floor_check main
