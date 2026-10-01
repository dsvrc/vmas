#!/usr/bin/env bash
#  Collect everything a later render needs, and nothing else, into one tarball.
#
#      bash scripts/pack_checkpoints.sh runs/clips
#      bash scripts/pack_checkpoints.sh runs/clips runs/baselines/balance
#
#  A replay (scripts/make_videos.py, on a machine with a display) needs exactly
#  two files per run:
#
#      <run>/config.pkl                           the task, its kwargs
#                                                 (severity, the PACT switch,
#                                                 the observation flags), the
#                                                 algorithm and model configs,
#                                                 and the seed
#      <run>/checkpoints/checkpoint_<frames>.pt   the weights
#
#  Those are what goes in.  The csv logs, the hydra output and pact_debug.csv
#  do not, because a video does not use them and they are what makes a naive
#  `scp -r` of a run tree slow.
#
#  Overridable:
#      LAST=1 bash scripts/pack_checkpoints.sh runs/clips    # newest ckpt only
#      PATTERN=pact bash scripts/pack_checkpoints.sh runs/clips
#      OUT=/tmp/clips.tar.gz bash scripts/pack_checkpoints.sh runs/clips
#      DRY=1 bash scripts/pack_checkpoints.sh runs/clips     # list, pack nothing

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/pack_checkpoints.sh <run root> [<run root> ...]" >&2
  exit 2
fi

OUT="${OUT:-checkpoints_$(date +%Y%m%d_%H%M%S).tar.gz}"
LAST="${LAST:-0}"
PATTERN="${PATTERN:-}"
DRY="${DRY:-0}"

list="$(mktemp)"
manifest="$(mktemp)"
trap 'rm -f "${list}" "${manifest}"' EXIT

#  The commit matters as much as the files.  config.pkl is a PICKLE: unpickling
#  it imports benchmarl, the simple_ns task class and the algorithm config
#  dataclass by name, so the laptop has to be on a checkout where those exist
#  with the same fields.  Mismatched code shows up as an AttributeError on
#  load, not as a bad video.
{
  echo "packed:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:    $(hostname)"
  echo "repo:    $(pwd)"
  echo "commit:  $(git rev-parse HEAD 2> /dev/null || echo 'not a git checkout')"
  echo "dirty:   $(git status --porcelain 2> /dev/null | wc -l) modified file(s)"
  echo "roots:   $*"
  echo "last:    ${LAST}  (1 = newest checkpoint per run only)"
  echo "pattern: ${PATTERN:-<none>}"
  echo
  echo "Unpack, then render on a machine with a display:"
  echo "    tar xzf $(basename "${OUT}")"
  echo "    python scripts/make_videos.py <run folder> --out videos"
  echo
  echo "runs:"
} > "${manifest}"

runs=0
for root in "$@"; do
  if [ ! -d "${root}" ]; then
    echo "!! no such directory: ${root}" >&2
    continue
  fi
  #  A run folder is one that holds config.pkl -- that is BenchMARL's own
  #  marker for "this is an experiment", and it is a level above checkpoints/.
  while IFS= read -r config; do
    run="$(dirname "${config}")"
    if [ -n "${PATTERN}" ] && ! printf '%s' "${run}" | grep -q "${PATTERN}"; then
      continue
    fi
    #  Sort by the frame count in the name, not lexically: checkpoint_600000
    #  sorts after checkpoint_3000000 as a string.
    mapfile -t ckpts < <(
      find "${run}/checkpoints" -maxdepth 1 -name "checkpoint_*.pt" -type f \
        2> /dev/null | sed 's/.*checkpoint_\([0-9]*\)\.pt/\1 &/' \
        | sort -n | cut -d' ' -f2-
    )
    if [ "${#ckpts[@]}" -eq 0 ]; then
      echo "   (no checkpoints, skipped) ${run}" >> "${manifest}"
      continue
    fi
    if [ "${LAST}" = "1" ]; then
      ckpts=("${ckpts[${#ckpts[@]}-1]}")
    fi
    echo "${config}" >> "${list}"
    for ckpt in "${ckpts[@]}"; do
      echo "${ckpt}" >> "${list}"
    done
    echo "   ${run}  (${#ckpts[@]} checkpoint(s))" >> "${manifest}"
    runs=$((runs + 1))
  done < <(find "${root}" -name "config.pkl" -type f | sort)
done

if [ "${runs}" -eq 0 ]; then
  echo "nothing to pack: no folder under $* holds a config.pkl with checkpoints." >&2
  echo "Those are written by the launchers; check the runs actually finished a" >&2
  echo "checkpoint interval (CKPT_EVERY) or reached the end." >&2
  exit 1
fi

cat "${manifest}"
echo
echo "files:"
wc -l < "${list}" | tr -d ' ' | sed 's/^/   /;s/$/ file(s)/'
du -ch $(tr '\n' ' ' < "${list}") 2> /dev/null | tail -1 | sed 's/^/   /'

if [ "${DRY}" = "1" ]; then
  echo
  echo "DRY=1, nothing written. The files that would go in:"
  sed 's/^/   /' "${list}"
  exit 0
fi

cp "${manifest}" MANIFEST_checkpoints.txt
tar czf "${OUT}" -T "${list}" MANIFEST_checkpoints.txt
rm -f MANIFEST_checkpoints.txt

echo
echo "== wrote ${OUT} ($(du -h "${OUT}" | cut -f1)) =="
echo "Download it with, from your laptop:"
echo "    scp $(whoami)@<this host>:$(pwd)/${OUT} ."
