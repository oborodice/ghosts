#!/usr/bin/env bash
set -euo pipefail

# podの上の学習(train_glyph_gan.py)が動いているかと、ログ(train_<名前>.log)の直近の行・最新の指標・最新の評価を表示する
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
DEFAULT_REMOTE_DIR="/workspace/ghosts/training"
DEFAULT_LINES=3

usage() {
  echo "Usage: $0 <ip> <port> --name <run-name> [--lines <n>] [--remote-dir <path>]" >&2
  echo "  --name: the run to check (the name given to runpod_launch_training.sh)" >&2
  echo "  --lines: number of recent log lines to show (default ${DEFAULT_LINES})" >&2
  echo "  --remote-dir: check a path other than ${DEFAULT_REMOTE_DIR} (see runpod_deploy_code.sh --remote-dir)" >&2
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
N_LINES="${DEFAULT_LINES}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --name) [ "$#" -ge 2 ] || usage; RUN_NAME="$2"; shift 2 ;;
    --remote-dir) [ "$#" -ge 2 ] || usage; REMOTE_DIR="$2"; shift 2 ;;
    --lines) [ "$#" -ge 2 ] || usage; N_LINES="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "${RUN_NAME}" ] || usage

# podの上で動かすスクリプト。変数は引数で渡す(ヒアドキュメントを引用符で囲み、ローカルでは展開しない)
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${POD_PORT}" "root@${POD_IP}" \
  bash -s -- "${RUN_NAME}" "${REMOTE_DIR}" "${N_LINES}" <<'EOF'
run_name="$1"; log="$2/train_$1.log"; n_lines="$3"
# コマンドラインがpythonで始まるものだけを探す(起動に使ったシェルのコマンドラインにも、学習のスクリプトの名前が含まれるため)
pids=$(pgrep -f "^[^ ]*python[^ ]* .*train_glyph_gan\.py --name ${run_name}( |$)" | tr '\n' ' ')
if [ -n "${pids}" ]; then echo "running: pid ${pids}"; else echo "(${run_name} is not running)"; fi
[ -f "${log}" ] || { echo "no log at ${log}"; exit 1; }
echo ---
tail -n "${n_lines}" "${log}"
echo ---
grep 'metrics at step' "${log}" | tail -n 1
grep 'evaluation at step' "${log}" | tail -n 1
EOF
