#!/usr/bin/env bash
#
# gpu_guard.sh - Detect and clean up stranded GPU processes before a run.
#
# Background: launching many run_test_exp.py jobs and suspending them (Ctrl+Z,
# state "T") instead of killing them leaves each holding a CUDA context. On the
# H100-40C vGPU this eventually exhausts the host-side context table, so even a
# clean GPU returns CUDA_ERROR_OUT_OF_MEMORY (cuCtxCreate -> 2) on the very
# first allocation. This guard kills those strays so a fresh run can grab a
# context.
#
# Usage:
#   ./gpu_guard.sh                 # report + clean stranded processes, then exit
#   ./gpu_guard.sh --dry-run       # report only, kill nothing
#   ./gpu_guard.sh -- <command>    # clean, then exec <command> (pre-run guard)
#
# Examples:
#   ./gpu_guard.sh -- python robust-kge/test_scripts/run_test_exp.py --loss_fn GCELoss
#   ./gpu_guard.sh --dry-run
#
# Safety:
#   - Only ever targets processes owned by the current user ($USER).
#   - By default kills only matching processes that are STOPPED (state T/t),
#     i.e. the suspended strays. Live (R/S) jobs are left alone unless --all.
#   - Matches the GPU workload by command pattern (PATTERN below) AND by
#     actually holding an NVIDIA device open, to avoid false positives.

set -uo pipefail

# Command-name pattern that identifies our GPU workloads. Extend as needed.
PATTERN="run_test_exp"

NVIDIA_DEV="/dev/nvidia0"
DRY_RUN=0
KILL_ALL_STATES=0   # --all: also kill live (running/sleeping) matches
CMD=()

# ---- parse args -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --all)     KILL_ALL_STATES=1; shift ;;
    --)        shift; CMD=("$@"); break ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *)
      echo "gpu_guard: unknown arg '$1' (use -- before a command)" >&2; exit 2 ;;
  esac
done

# ---- find stranded PIDs ----------------------------------------------------
# PIDs of the current user that currently hold the NVIDIA device open.
mapfile -t GPU_PIDS < <(
  fuser "$NVIDIA_DEV" 2>/dev/null \
    | tr ' ' '\n' \
    | sed 's/[a-z]*$//' \
    | grep -E '^[0-9]+$' \
    | sort -u
)

declare -a STRANDED=()
for pid in "${GPU_PIDS[@]:-}"; do
  [[ -z "$pid" ]] && continue
  # owner check: skip anything not owned by us
  owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  [[ "$owner" != "$USER" ]] && continue
  # command + state
  read -r stat cmd < <(ps -o stat=,cmd= -p "$pid" 2>/dev/null)
  [[ -z "${stat:-}" ]] && continue
  # must match our workload pattern
  [[ "$cmd" != *"$PATTERN"* ]] && continue
  # state filter: by default only stopped (T/t); --all takes any state
  if [[ "$KILL_ALL_STATES" -eq 0 && "${stat:0:1}" != "T" && "${stat:0:1}" != "t" ]]; then
    continue
  fi
  STRANDED+=("$pid")
done

n=${#STRANDED[@]}

if [[ "$n" -eq 0 ]]; then
  echo "gpu_guard: no stranded '$PATTERN' processes holding $NVIDIA_DEV. GPU is clear."
else
  echo "gpu_guard: found $n stranded '$PATTERN' process(es) holding the GPU:"
  for pid in "${STRANDED[@]}"; do
    ps -o pid=,stat=,etime=,cmd= -p "$pid" 2>/dev/null | sed 's/^/    /'
  done
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "gpu_guard: --dry-run set, killing nothing."
  else
    echo "gpu_guard: sending SIGKILL to $n process(es)..."
    kill -9 "${STRANDED[@]}" 2>/dev/null
    sleep 2
    # verify
    remaining=$(fuser "$NVIDIA_DEV" 2>/dev/null | tr ' ' '\n' | sed 's/[a-z]*$//' \
                 | grep -E '^[0-9]+$' | sort -u \
                 | while read -r p; do
                     [[ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" == "$USER" ]] \
                       && ps -o cmd= -p "$p" 2>/dev/null | grep -q "$PATTERN" && echo "$p"
                   done | wc -l)
    echo "gpu_guard: done. Remaining '$PATTERN' GPU holders for $USER: $remaining"
  fi
fi

# ---- pre-run guard mode ----------------------------------------------------
if [[ "${#CMD[@]}" -gt 0 ]]; then
  echo "gpu_guard: launching -> ${CMD[*]}"
  exec "${CMD[@]}"
fi
