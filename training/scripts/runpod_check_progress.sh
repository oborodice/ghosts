#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
DEFAULT_REMOTE_DIR="/workspace/ghosts/training"

usage() {
  echo "Usage: $0 <ip> <port> [n-lines] [--remote-dir <path>]" >&2
  echo "  --remote-dir: check a path other than ${DEFAULT_REMOTE_DIR} (see runpod_deploy_code.sh" >&2
  echo "  --remote-dir, for a config deployed alongside others on one pod)" >&2
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
fi

POD_IP="$1"
POD_PORT="$2"
shift 2

REMOTE_DIR="${DEFAULT_REMOTE_DIR}"
N_LINES="3"
POSITIONAL_INDEX=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --remote-dir)
      [ "$#" -ge 2 ] || usage
      REMOTE_DIR="$2"
      shift 2
      ;;
    *)
      if [ "${POSITIONAL_INDEX}" -eq 0 ]; then
        N_LINES="$1"
      else
        usage
      fi
      POSITIONAL_INDEX=$((POSITIONAL_INDEX + 1))
      shift
      ;;
  esac
done

# pidだけでなくcwdも表示する: 1つのpodに複数構成を同時実行しているとtrain_vae_v2.pyが
# 複数ヒットするため、どのremote-dirのものかをcwdで見分けられるようにする。
# pgrep -fの候補を/proc/<pid>/commが"python3"のものだけに絞る: 起動に使ったnohup+disownの
# bashラッパー自身のコマンドライン文字列にも(埋め込まれた元のコマンドとして)"train_vae_v2"が
# 含まれてしまい、文字列パターンだけでは絞り込めない(実測: ラッパーのbashプロセスが
# disown後も生き残り、pgrep -fに誤って拾われるケースがあった)。commで実プロセスの
# 実行ファイル名を見れば、bashやuv(uv run経由の旧起動方式)のラッパーを確実に除外できる
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" \
  "pids=\$(for p in \$(pgrep -f '[t]rain_vae_v2'); do [ \"\$(cat /proc/\${p}/comm 2>/dev/null)\" = python3 ] && echo \${p}; done); if [ -z \"\${pids}\" ]; then echo '(train_vae_v2.py is not running)'; else for pid in \${pids}; do echo \"pid=\${pid} cwd=\$(readlink -f /proc/\${pid}/cwd)\"; done; fi; echo ---; tail -n ${N_LINES} ${REMOTE_DIR}/train.log"
