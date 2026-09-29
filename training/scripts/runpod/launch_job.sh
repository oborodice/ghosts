#!/usr/bin/env bash
set -euo pipefail

# podの上で、scripts/ の任意のPythonスクリプトを、SSHから切り離してバックグラウンドで始める(学習のほか、文字認識のモデルの学習・
# 評価・分析など)。ログは <ジョブの名前>.log、プロセスの番号は <ジョブの名前>.pid に書き、終わるとログの最後に「exit code <n>」を足す
# (check_progress.sh --job で、動いているか・どう終わったかを見る)。setsidで新しいセッションにし、標準入出力もすべて
# ログへ向けるので、SSHを切っても止まらず、ローカルのsshコマンドもすぐに戻る
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
DEFAULT_REMOTE_DIR="/workspace/ghosts/training"
# 起動直後の失敗(引数の誤り・データの読み込みの失敗など)を見つけるため、少し待ってから動いているかを確かめる
STARTUP_WAIT_SECONDS=5

usage() {
  echo "Usage: $0 <ip> <port> --job <job-name> [--append] [--remote-dir <path>] -- <script> [args...]" >&2
  echo "  --job: the job's name; the log goes to <job-name>.log and the process id to <job-name>.pid under the remote dir" >&2
  echo "  --append: append to an existing log instead of overwriting it (e.g. when resuming)" >&2
  echo "  --remote-dir: launch in a copy deployed under a path other than ${DEFAULT_REMOTE_DIR} (see deploy_code.sh)" >&2
  echo "  <script> [args...]: run as '.venv/bin/python3 -u <script> [args...]' in the remote dir (e.g. scripts/train_classifier.py)" >&2
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
APPEND=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --job) [ "$#" -ge 2 ] || usage; JOB="$2"; shift 2 ;;
    --append) APPEND=1; shift ;;
    --remote-dir) [ "$#" -ge 2 ] || usage; REMOTE_DIR="$2"; shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
[ -n "${JOB}" ] && [ "$#" -ge 1 ] || usage

SSH=(ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${POD_PORT}" "root@${POD_IP}")
# podの上で動かすスクリプト(ヒアドキュメントを引用符で囲み、ローカルでは展開しない)。引数は位置引数で渡す。
# sshは引数を空白でつないでpodのシェルに渡し直すので、空白や改行を含む引数が崩れないよう、1つずつ引用してから渡す
"${SSH[@]}" "bash -s -- $(printf '%q ' "${REMOTE_DIR}" "${JOB}" "${APPEND}" "$@")" <<'EOF'
dir="$1"; job="$2"; append="$3"; shift 3
cd "${dir}" || exit 1
# 背後に回すのは、出力をすべてログへ向けた1つのコマンドだけにする(複数のコマンドをまとめて背後に回すと、その子シェルの出力が
# SSHにつながったままになり、SSHが戻らない)。uv runではなく .venv のpythonを直接呼ぶ(--link-venv でつないだvenvに対して、
# uvがロックの確認・同期をし直そうとするのを避ける)。部品のパッケージ(glyph)は、この配置先のものを使うようPYTHONPATHの先頭に置く
# (--link-venv でつないだvenvは、venvを作った配置先のglyphを指しているため)
wrapper='"$@"; echo "exit code $?"'
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
if [ "${append}" = 1 ]; then
  setsid nohup bash -c "${wrapper}" job .venv/bin/python3 -u "$@" >> "${job}.log" 2>&1 < /dev/null &
else
  setsid nohup bash -c "${wrapper}" job .venv/bin/python3 -u "$@" > "${job}.log" 2>&1 < /dev/null &
fi
echo $! > "${job}.pid"
EOF

sleep "${STARTUP_WAIT_SECONDS}"
"${SSH[@]}" "bash -s -- $(printf '%q ' "${REMOTE_DIR}" "${JOB}")" <<'EOF'
cd "$1" || exit 1; job="$2"
if ps -o pid=,args= -p "$(cat "${job}.pid")"; then
  exit 0
fi
# 短いジョブは、待っているうちに終わることがある。正常に終わっていれば成功とする
if tail -n 1 "${job}.log" | grep -qx "exit code 0"; then
  echo "The job already finished:"
  tail -n 3 "${job}.log"
  exit 0
fi
echo "The job is not running. Last lines of the log:"
tail -n 20 "${job}.log"
exit 1
EOF
