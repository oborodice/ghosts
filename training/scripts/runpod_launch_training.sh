#!/usr/bin/env bash
set -euo pipefail

# podの上で train_glyph_gan.py をバックグラウンドで始める。ログは学習の名前ごとに train_<名前>.log へ書く
# (1つのpodで複数の学習を同時に回せるように)。setsid で新しいセッションにし、標準入出力もすべてSSHから切り離すので、
# SSHを切っても学習は止まらず、ローカルのsshコマンドもすぐに戻る(ssh -n で標準入力も渡さない)
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
DEFAULT_REMOTE_DIR="/workspace/ghosts/training"

usage() {
  echo "Usage: $0 <ip> <port> --name <run-name> [--resume] [--remote-dir <path>] [--train-args \"<args>\"]" >&2
  echo "  --name: the run's name; checkpoints go to data/checkpoints/glyph_gan/<run-name>/ and the log to train_<run-name>.log" >&2
  echo "  --resume: continue the run <run-name> from its latest checkpoint" >&2
  echo "  --train-args: extra arguments passed to train_glyph_gan.py as one quoted string (e.g. \"--kimg 960 --seed 1\")" >&2
  echo "  --remote-dir: launch a copy deployed under a path other than ${DEFAULT_REMOTE_DIR} (see runpod_deploy_code.sh)" >&2
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
RESUME_ARG=""
TRAIN_ARGS=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --name) [ "$#" -ge 2 ] || usage; RUN_NAME="$2"; shift 2 ;;
    --resume) RESUME_ARG="--resume"; shift ;;
    --remote-dir) [ "$#" -ge 2 ] || usage; REMOTE_DIR="$2"; shift 2 ;;
    --train-args) [ "$#" -ge 2 ] || usage; TRAIN_ARGS="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "${RUN_NAME}" ] || usage

SSH=(ssh -n -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${POD_PORT}" "root@${POD_IP}")
# 再開のときはログを上書きせずに追記する
REDIRECT=">"
[ -n "${RESUME_ARG}" ] && REDIRECT=">>"
# 起動直後の失敗(引数の誤り・データの読み込みの失敗など)を見つけるため、少し待ってから動いているかを確かめる
STARTUP_WAIT_SECONDS=5
# uv runではなく.venvのpythonを直接呼ぶ(--link-venvで他ディレクトリからシンボリックリンクしたvenvに対して、
# uvがロック検証・再同期を試みる可能性を避ける)
"${SSH[@]}" "cd ${REMOTE_DIR} && setsid nohup .venv/bin/python3 -u scripts/train_glyph_gan.py --name ${RUN_NAME} ${RESUME_ARG} ${TRAIN_ARGS} ${REDIRECT} train_${RUN_NAME}.log 2>&1 < /dev/null &"
sleep "${STARTUP_WAIT_SECONDS}"
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${POD_PORT}" "root@${POD_IP}" \
  bash -s -- "${RUN_NAME}" "${REMOTE_DIR}" <<'EOF'
run_name="$1"; log="$2/train_$1.log"
# コマンドラインがpythonで始まるものだけを探す(起動に使ったシェルのコマンドラインにも、学習のスクリプトの名前が含まれるため)
if pgrep -af "^[^ ]*python[^ ]* .*train_glyph_gan\.py --name ${run_name}( |$)"; then
  exit 0
fi
# 短い学習は、待っているうちに終わることがある。最後まで終わっていれば成功とする
if tail -n 1 "${log}" | grep -q '^done$'; then
  echo "The run already finished:"
  tail -n 3 "${log}"
  exit 0
fi
echo "The run is not running. Last lines of the log:"
tail -n 20 "${log}"
exit 1
EOF
