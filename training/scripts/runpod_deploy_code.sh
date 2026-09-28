#!/usr/bin/env bash
set -euo pipefail

# コード(scripts・pyproject.toml・uv.lock)は変わるたびに毎回送る想定のため常に送る。
# 学習データ(build_glyph_dataset.py で作った data/glyphs_<解像度>.npz)は滅多に変わらないため、指定したときだけ送る。
# 学習中に種類の数を測る文字認識のモデルの重みも、指定したときだけ送る。
# データはどのソースを使う場合でも実データ(本番のtraining/data/)から送る
# (アブレーション用のスクラッチディレクトリはコードのみでデータを複製していないため)
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
DEFAULT_REMOTE_DIR="/workspace/ghosts/training"
# uvはデフォルトでPythonインタプリタを/root配下(コンテナ固有のエフェメラルディスク)に置くため、
# Network Volumeを使い回して新しいpodを作ると、venv自体(training/.venv、Volume上にあり中身は
# 残っている)へのシンボリックリンクが指す先だけ存在しなくなり、再ダウンロードが発生する。
# インタプリタだけ/workspace配下(Network Volume上)に固定すれば十分で、venv本体は元から
# Volume上にあるため再インストールは走らない(リモート側の環境変数として渡す)。
# パッケージのダウンロードキャッシュ(UV_CACHE_DIR)は永続化しない: このファイルシステムでは
# ハードリンクが効かず(uv実行時に"Failed to hardlink files"の警告が出る)、キャッシュとvenvが
# 別々にフルコピーされるため、永続化すると数GB〜10GB規模でNetwork Volumeを圧迫する
UV_ENV_VARS="UV_PYTHON_INSTALL_DIR=/workspace/.uv-python"

usage() {
  echo "Usage: $0 <ip> <port> [--data <file>] [--classifier <file>] [--source <local-dir>] [--remote-dir <path>] [--link-venv <remote-dir>]" >&2
  echo "  --data: send a dataset file under training/data/ (e.g. glyphs_64.npz) to the same place on the pod" >&2
  echo "  --classifier: send a classifier weight file under training/data/ (for train_glyph_gan.py --classifier)" >&2
  echo "  --source: send scripts/pyproject.toml/uv.lock from a directory other than training/ (e.g. a scratch copy for a parallel ablation)" >&2
  echo "  --remote-dir: deploy under a path other than ${DEFAULT_REMOTE_DIR} (a self-contained copy with its own data/;" >&2
  echo "  checkpoint/data paths are relative to the script file, not the CWD, so it needs --data too)." >&2
  echo "  Several runs can also share one remote-dir: train_glyph_gan.py writes each run under its own --name" >&2
  echo "  --link-venv: skip 'uv sync' and symlink .venv from another already-synced remote-dir on the same pod instead" >&2
  echo "  (saves re-downloading ~3GB of CUDA/torch packages; only valid when pyproject.toml/uv.lock are identical" >&2
  echo "  to that remote-dir's, which holds as long as only the swept constants differ, not dependencies)" >&2
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
fi

POD_IP="$1"
POD_PORT="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${TRAINING_DIR}"
REMOTE_DIR="${DEFAULT_REMOTE_DIR}"
DATA_FILES=()
LINK_VENV_FROM=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data|--classifier) [ "$#" -ge 2 ] || usage; DATA_FILES+=("$2"); shift 2 ;;
    --source) [ "$#" -ge 2 ] || usage; SOURCE_DIR="$(cd "$2" && pwd)"; shift 2 ;;
    --remote-dir) [ "$#" -ge 2 ] || usage; REMOTE_DIR="$2"; shift 2 ;;
    --link-venv) [ "$#" -ge 2 ] || usage; LINK_VENV_FROM="$2"; shift 2 ;;
    *) usage ;;
  esac
done

SSH=(ssh -n -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${POD_PORT}" "root@${POD_IP}")
SCP=(scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no -P "${POD_PORT}")

"${SSH[@]}" "mkdir -p ${REMOTE_DIR}/data"

echo "Sending scripts/pyproject.toml/uv.lock from ${SOURCE_DIR} ..."
"${SCP[@]}" -r "${SOURCE_DIR}/scripts" "${SOURCE_DIR}/pyproject.toml" "${SOURCE_DIR}/uv.lock" "root@${POD_IP}:${REMOTE_DIR}/"

for data_file in ${DATA_FILES[@]+"${DATA_FILES[@]}"}; do
  echo "Sending data/${data_file} ..."
  "${SCP[@]}" "${TRAINING_DIR}/data/${data_file}" "root@${POD_IP}:${REMOTE_DIR}/data/"
done

if [ -n "${LINK_VENV_FROM}" ]; then
  echo "Linking .venv from ${LINK_VENV_FROM} (skipping uv sync) ..."
  "${SSH[@]}" "ln -sfn ${LINK_VENV_FROM}/.venv ${REMOTE_DIR}/.venv"
else
  echo "Running uv sync ..."
  "${SSH[@]}" "export ${UV_ENV_VARS}; cd ${REMOTE_DIR} && uv sync"
fi

echo "Checking CUDA is recognized ..."
# uv runではなく.venvのpythonを直接呼ぶ: --link-venv時はシンボリックリンク先のuv.lockと
# このディレクトリのuv.lockが同一である保証がuv側にはなく、uv runがロック検証や再同期を
# 試みる可能性があるため(通常時もvenvは既に解決済みなので直接呼んで実質的な差はない)
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${POD_PORT}" "root@${POD_IP}" \
  "cd ${REMOTE_DIR} && .venv/bin/python3 -" <<'EOF'
import torch
available = torch.cuda.is_available()
print("cuda available:", available, torch.cuda.get_device_name(0) if available else "")
EOF
