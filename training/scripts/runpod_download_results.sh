#!/usr/bin/env bash
set -euo pipefail

# Network Volumeは環境再利用のため使い回す(pod削除時にも消えない)ため、古い実行のチェックポイントが残っていることがある。
# 明示的に指定しない限り、更新時刻が最新の*.ptを対象にすることで取り違えを防ぐ
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
REMOTE_DIR="/workspace/ghosts/training"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <ip> <port> [remote-checkpoint-name|latest] [local-name]" >&2
  exit 1
fi

POD_IP="$1"
POD_PORT="$2"
REMOTE_NAME="${3:-latest}"
LOCAL_NAME="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CHECKPOINT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/data/checkpoints"

if [ "${REMOTE_NAME}" = "latest" ]; then
  REMOTE_NAME="$(ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" \
    "ls -t ${REMOTE_DIR}/data/checkpoints/vae_*.pt | grep -v _resume.pt | head -1 | xargs basename")"
  echo "Latest checkpoint: ${REMOTE_NAME}"
fi

scp -i "${SSH_KEY}" -P "${POD_PORT}" \
  root@"${POD_IP}":"${REMOTE_DIR}/data/checkpoints/${REMOTE_NAME}" "${LOCAL_CHECKPOINT_DIR}/"
# ログはチェックポイントと同じディレクトリ(gitignore済み)に、対応する名前で残す
scp -i "${SSH_KEY}" -P "${POD_PORT}" \
  root@"${POD_IP}":"${REMOTE_DIR}/train.log" "${LOCAL_CHECKPOINT_DIR}/${REMOTE_NAME%.pt}_train.log"

if [ -n "${LOCAL_NAME}" ]; then
  mv "${LOCAL_CHECKPOINT_DIR}/${REMOTE_NAME}" "${LOCAL_CHECKPOINT_DIR}/${LOCAL_NAME}"
  mv "${LOCAL_CHECKPOINT_DIR}/${REMOTE_NAME%.pt}_train.log" "${LOCAL_CHECKPOINT_DIR}/${LOCAL_NAME%.pt}_train.log"
  echo "Renamed to ${LOCAL_NAME}"
fi
