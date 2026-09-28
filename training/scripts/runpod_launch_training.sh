#!/usr/bin/env bash
set -euo pipefail

# podの上で train_glyph_gan.py をバックグラウンドで始める(runpod_launch_job.sh で、ジョブの名前を train_<学習の名前> にして起動する)。
# ログは学習の名前ごとに train_<名前>.log へ書くので、1つのpodで複数の学習を同時に回せる
DEFAULT_REMOTE_DIR="/workspace/ghosts/training"

usage() {
  echo "Usage: $0 <ip> <port> --name <run-name> [--resume] [--remote-dir <path>] [-- <train_glyph_gan.py args>...]" >&2
  echo "  --name: the run's name; checkpoints go to data/checkpoints/glyph_gan/<run-name>/ and the log to train_<run-name>.log" >&2
  echo "  --resume: continue the run <run-name> from its latest checkpoint (the log is appended to)" >&2
  echo "  --remote-dir: launch a copy deployed under a path other than ${DEFAULT_REMOTE_DIR} (see runpod_deploy_code.sh)" >&2
  echo "  -- <args>...: extra arguments passed to train_glyph_gan.py as they are (e.g. -- --kimg 960 --seed 1)" >&2
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
fi

POD_IP="$1"
POD_PORT="$2"
shift 2

REMOTE_DIR="${DEFAULT_REMOTE_DIR}"
RUN_NAME=""
RESUME=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --name) [ "$#" -ge 2 ] || usage; RUN_NAME="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --remote-dir) [ "$#" -ge 2 ] || usage; REMOTE_DIR="$2"; shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
[ -n "${RUN_NAME}" ] || usage

JOB_ARGS=(--job "train_${RUN_NAME}" --remote-dir "${REMOTE_DIR}")
TRAIN_COMMAND=(scripts/train_glyph_gan.py --name "${RUN_NAME}")
if [ "${RESUME}" = 1 ]; then
  JOB_ARGS+=(--append)
  TRAIN_COMMAND+=(--resume)
fi
exec "$(dirname "$0")/runpod_launch_job.sh" "${POD_IP}" "${POD_PORT}" "${JOB_ARGS[@]}" -- \
  "${TRAIN_COMMAND[@]}" "$@"
