#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
REMOTE_DIR="/workspace/ghosts/training"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <ip> <port> [n-lines]" >&2
  exit 1
fi

POD_IP="$1"
POD_PORT="$2"
N_LINES="${3:-3}"

ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" \
  "ps aux | grep train_vae_v2 | grep -v grep || echo '(train_vae_v2.py is not running)'; echo ---; tail -n ${N_LINES} ${REMOTE_DIR}/train.log"
