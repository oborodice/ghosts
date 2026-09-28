#!/usr/bin/env bash
set -euo pipefail

# podの上の学習の結果(data/checkpoints/glyph_gan/<名前>/ のチェックポイント・スナップショット・評価の値 evaluation.csv と、
# ログ train_<名前>.log)を、ローカルの同じ場所へ落とす。
# 落としたあと、チェックポイント・スナップショット・評価の値の数と中身(SHA-256)をpodの上と比べ、すべて一致したときだけ
# podの上のチェックポイントとスナップショットを消す(評価の値は小さいので残す)
# (Network Volumeの容量を空けるため。一致しなければ消さずに止まる)。podを消す前に必ず実行する
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
DEFAULT_REMOTE_DIR="/workspace/ghosts/training"

usage() {
  echo "Usage: $0 <ip> <port> --name <run-name> [--keep-remote] [--remote-dir <path>]" >&2
  echo "  --name: the run to download (the name given to launch_training.sh)" >&2
  echo "  --keep-remote: do not delete the checkpoints and snapshots on the pod after a verified download" >&2
  echo "  --remote-dir: download from a path other than ${DEFAULT_REMOTE_DIR} (see deploy_code.sh --remote-dir)" >&2
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
KEEP_REMOTE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --name) [ "$#" -ge 2 ] || usage; RUN_NAME="$2"; shift 2 ;;
    --keep-remote) KEEP_REMOTE="1"; shift ;;
    --remote-dir) [ "$#" -ge 2 ] || usage; REMOTE_DIR="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "${RUN_NAME}" ] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_RUN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)/data/checkpoints/glyph_gan/${RUN_NAME}"
REMOTE_RUN_DIR="${REMOTE_DIR}/data/checkpoints/glyph_gan/${RUN_NAME}"
SSH=(ssh -n -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${POD_PORT}" "root@${POD_IP}")
SCP=(scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no -P "${POD_PORT}")

mkdir -p "$(dirname "${LOCAL_RUN_DIR}")"
# 学習の名前のディレクトリをまるごと、ローカルの同じ名前のディレクトリへ落とす
"${SCP[@]}" -r "root@${POD_IP}:${REMOTE_RUN_DIR}" "$(dirname "${LOCAL_RUN_DIR}")/"
"${SCP[@]}" "root@${POD_IP}:${REMOTE_DIR}/train_${RUN_NAME}.log" "${LOCAL_RUN_DIR}/train.log"

# podの上にあるファイルだけを、ローカルの同じ名前のファイルと比べる(前に落として、podの上からは消した分がローカルに残っていてもよい)
# (チェックポイントがないときは何も出さない。sha256sum に存在しない *.pt を渡してエラーで止まらないように。
# 評価の値は、学習に --classifier を指定したときだけある)
REMOTE_SUMS="$("${SSH[@]}" "cd ${REMOTE_RUN_DIR} && ls *.pt >/dev/null 2>&1 && sha256sum *.pt \$(ls evaluation.csv 2>/dev/null) || true" | awk '{print $1, $2}' | sort)"
if [ -z "${REMOTE_SUMS}" ]; then
  echo "Error: no checkpoints found in ${REMOTE_RUN_DIR} on the pod" >&2
  exit 1
fi
REMOTE_NAMES=($(echo "${REMOTE_SUMS}" | awk '{print $2}'))
LOCAL_SUMS="$(cd "${LOCAL_RUN_DIR}" && shasum -a 256 ${REMOTE_NAMES[@]+"${REMOTE_NAMES[@]}"} | awk '{print $1, $2}' | sort)"
if [ "${REMOTE_SUMS}" != "${LOCAL_SUMS}" ]; then
  echo "Error: downloaded files do not match the pod (remote files are kept)" >&2
  diff <(echo "${REMOTE_SUMS}") <(echo "${LOCAL_SUMS}") >&2 || true
  exit 1
fi
echo "Verified $(echo "${REMOTE_SUMS}" | wc -l | tr -d ' ') files in ${LOCAL_RUN_DIR}"

if [ -z "${KEEP_REMOTE}" ]; then
  echo "Cleaning up remote checkpoints and snapshots in ${REMOTE_RUN_DIR} ..."
  "${SSH[@]}" "rm -f ${REMOTE_RUN_DIR}/*.pt"
fi
