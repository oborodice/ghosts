#!/usr/bin/env bash
set -euo pipefail

# 標準入出力を全てSSHチャネルから切り離してnohup+disownでバックグラウンド化することで、
# SSHセッションを切っても学習が止まらないようにする。標準入力(< /dev/null)を切り離し忘れると、
# ローカルのsshコマンド自体が学習終了まで返ってこなくなる(学習プロセスへの影響はない)
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
DEFAULT_REMOTE_DIR="/workspace/ghosts/training"

usage() {
  echo "Usage: $0 <ip> <port> [remote-resume-path] [--remote-dir <path>] [--train-args \"<args>\"]" >&2
  echo "  --train-args: extra arguments passed to train_vae_v2.py as one quoted string (e.g. \"--max-epochs 40\")" >&2
  echo "  --remote-dir: launch a copy deployed under a path other than ${DEFAULT_REMOTE_DIR} (see" >&2
  echo "  runpod_deploy_code.sh --remote-dir, for running several sweep configs on one pod)" >&2
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
fi

POD_IP="$1"
POD_PORT="$2"
shift 2

REMOTE_DIR="${DEFAULT_REMOTE_DIR}"
RESUME_PATH=""
TRAIN_ARGS=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --remote-dir)
      [ "$#" -ge 2 ] || usage
      REMOTE_DIR="$2"
      shift 2
      ;;
    --train-args)
      [ "$#" -ge 2 ] || usage
      TRAIN_ARGS="$2"
      shift 2
      ;;
    *)
      RESUME_PATH="$1"
      shift
      ;;
  esac
done

RESUME_ARG=""
if [ -n "${RESUME_PATH}" ]; then
  RESUME_ARG="--resume ${RESUME_PATH}"
fi

# uv runではなく.venvのpythonを直接呼ぶ(runpod_deploy_code.shのCUDA確認と同じ理由:
# --link-venvで他ディレクトリからシンボリックリンクしたvenvに対してuvがロック検証・
# 再同期を試みる可能性を避ける)
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" \
  "cd ${REMOTE_DIR} && nohup .venv/bin/python3 -u scripts/train_vae_v2.py ${RESUME_ARG} ${TRAIN_ARGS} > train.log 2>&1 < /dev/null & disown; sleep 2; ps aux | grep train_vae_v2 | grep -v grep"
