#!/usr/bin/env bash
set -euo pipefail

# podの上のジョブ(launch_job.sh・launch_training.sh で始めたもの)が動いているか・どう終わったかと、ログの直近の行を表示する。
# 学習(--name)なら、最新の指標・最新の評価も出す。--follow では、終わるまで新しい行を流し、ジョブの終了コードで終わる
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
DEFAULT_REMOTE_DIR="/workspace/ghosts/training"
DEFAULT_LINES=3
DEFAULT_INTERVAL_SECONDS=60
# --follow で学習のログから流す行(評価・終わり・失敗)。学習以外のジョブは、すべての行を流す
TRAINING_FOLLOW_PATTERN='^evaluation at|^done$|Traceback|Error|Killed|out of memory'

usage() {
  echo "Usage: $0 <ip> <port> (--name <run-name> | --job <job-name>) [--lines <n>] [--follow [--interval <seconds>]] [--remote-dir <path>]" >&2
  echo "  --name: the training run to check (the name given to launch_training.sh)" >&2
  echo "  --job: any job to check (the job name given to launch_job.sh)" >&2
  echo "  --lines: number of recent log lines to show (default ${DEFAULT_LINES})" >&2
  echo "  --follow: keep printing new log lines (for a training run: evaluations, the end and errors) until the job ends," >&2
  echo "    polling every --interval seconds (default ${DEFAULT_INTERVAL_SECONDS}); exits with the job's exit code" >&2
  echo "  --remote-dir: check a path other than ${DEFAULT_REMOTE_DIR} (see deploy_code.sh --remote-dir)" >&2
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
fi

POD_IP="$1"
POD_PORT="$2"
shift 2

REMOTE_DIR="${DEFAULT_REMOTE_DIR}"
JOB=""
IS_TRAINING=0
LINES_TO_SHOW="${DEFAULT_LINES}"
FOLLOW=0
INTERVAL_SECONDS="${DEFAULT_INTERVAL_SECONDS}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --name) [ "$#" -ge 2 ] || usage; JOB="train_$2"; IS_TRAINING=1; shift 2 ;;
    --job) [ "$#" -ge 2 ] || usage; JOB="$2"; shift 2 ;;
    --remote-dir) [ "$#" -ge 2 ] || usage; REMOTE_DIR="$2"; shift 2 ;;
    --lines) [ "$#" -ge 2 ] || usage; LINES_TO_SHOW="$2"; shift 2 ;;
    --follow) FOLLOW=1; shift ;;
    --interval) [ "$#" -ge 2 ] || usage; INTERVAL_SECONDS="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "${JOB}" ] || usage
[[ "${LINES_TO_SHOW}" =~ ^[0-9]+$ && "${INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] || usage

SSH=(ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "${POD_PORT}" "root@${POD_IP}")
SSH_CONNECTION_FAILED=255

remote_report() {
  # podの上で、1行目に「<改行で終わった行の数> <ジョブの状態(running: pid <n> / exit code <n> / not running)>」を出し、
  # 2行目以降に、$1 が follow なら $2 行目以降の完成した行のうち $3 に合う行を、そうでなければ直近 $2 行(学習なら最新の指標・評価も)を出す。
  # 長いログを丸ごと送らないよう、選ぶのはpodの上で行う。sshは引数を空白でつなぎ直すので、1つずつ引用して渡す
  "${SSH[@]}" "bash -s -- $(printf '%q ' "${REMOTE_DIR}" "${JOB}" "$1" "$2" "$3" "${IS_TRAINING}")" <<'EOF'
cd "$1" || exit 1; job="$2"; mode="$3"; count="$4"; pattern="$5"; is_training="$6"
[ -f "${job}.log" ] || { echo "no log at $1/${job}.log" >&2; exit 1; }
# 状態を行の数より先に見る(終わったと読んだ時点で、最後の行まで書き終わっている)
if [ -f "${job}.pid" ] && kill -0 "$(cat "${job}.pid")" 2>/dev/null; then
  status="running: pid $(cat "${job}.pid")"
elif tail -n 1 "${job}.log" | grep -qx "exit code [0-9]*"; then
  status="$(tail -n 1 "${job}.log")"
else
  status="not running"
fi
# 書きかけの行(最後の改行がまだない行)は数えず、次の回に完成してから読む
total=$(wc -l < "${job}.log" | tr -d ' ')
echo "${total} ${status}"
if [ "${mode}" = follow ]; then
  [ "${count}" -le "${total}" ] && sed -n "${count},${total}p" "${job}.log" | grep -E "${pattern}"
  exit 0
fi
echo ---
tail -n "${count}" "${job}.log"
if [ "${is_training}" = 1 ]; then
  echo ---
  grep 'metrics at step' "${job}.log" | tail -n 1
  grep 'evaluation at step' "${job}.log" | tail -n 1
fi
exit 0
EOF
}

if [ "${FOLLOW}" = 0 ]; then
  OUTPUT="$(remote_report list "${LINES_TO_SHOW}" "")"
  # 1行目の行の数は、follow で使うもの。ここでは状態だけを出す
  HEADER="${OUTPUT%%$'\n'*}"
  echo "${HEADER#* }"
  echo "${OUTPUT#*$'\n'}"
  exit 0
fi

PATTERN=""
[ "${IS_TRAINING}" = 1 ] && PATTERN="${TRAINING_FOLLOW_PATTERN}"
# 次に読む行の番号を覚えておき、新しい行だけを取り寄せる(SSHがつながらないときは、次の回にもう一度試す)
NEXT_LINE=1
while true; do
  if OUTPUT="$(remote_report follow "${NEXT_LINE}" "${PATTERN}")"; then
    HEADER="${OUTPUT%%$'\n'*}"
    [ "${OUTPUT}" != "${HEADER}" ] && echo "${OUTPUT#*$'\n'}"
    NEXT_LINE=$(( ${HEADER%% *} + 1 ))
    STATUS="${HEADER#* }"
    case "${STATUS}" in
      running:*) ;;
      "exit code "*) echo "${JOB} finished (${STATUS})"; exit "${STATUS#exit code }" ;;
      *) echo "${JOB} is not running and left no exit code" >&2; exit 1 ;;
    esac
  else
    RETURN_CODE=$?
    # ジョブの名前の誤り(ログがない)などは、待っても直らないので止まる
    [ "${RETURN_CODE}" = "${SSH_CONNECTION_FAILED}" ] || exit "${RETURN_CODE}"
  fi
  sleep "${INTERVAL_SECONDS}"
done
