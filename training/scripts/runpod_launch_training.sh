#!/usr/bin/env bash
set -euo pipefail

# 標準入出力を全てSSHチャネルから切り離してnohup+disownでバックグラウンド化することで、
# SSHセッションを切っても学習が止まらないようにする。標準入力(< /dev/null)を切り離し忘れると、
# ローカルのsshコマンド自体が学習終了まで返ってこなくなる(学習プロセスへの影響はない)
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
REMOTE_DIR="/workspace/ghosts/training"
# runpod_deploy_code.shでNetwork Volume上に固定したPythonインタプリタ・キャッシュの場所と一致させる
UV_ENV_VARS="UV_PYTHON_INSTALL_DIR=/workspace/.uv-python UV_CACHE_DIR=/workspace/.uv-cache"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <ip> <port> [remote-resume-path]" >&2
  exit 1
fi

POD_IP="$1"
POD_PORT="$2"
RESUME_PATH="${3:-}"

RESUME_ARG=""
if [ -n "${RESUME_PATH}" ]; then
  RESUME_ARG="--resume ${RESUME_PATH}"
fi

ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" \
  "export ${UV_ENV_VARS}; cd ${REMOTE_DIR} && nohup uv run python3 -u scripts/train_vae_v2.py ${RESUME_ARG} > train.log 2>&1 < /dev/null & disown; sleep 2; ps aux | grep train_vae_v2 | grep -v grep"
