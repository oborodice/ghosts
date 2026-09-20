#!/usr/bin/env bash
set -euo pipefail

# データ(stroke_features_v2.npz)は滅多に変わらないため、毎回送ると無駄な転送になる。
# コード(scripts・pyproject.toml・uv.lock)は変わるたびに毎回送る想定のため常に送る。
# データはどのソースを使う場合でも実データ(本番のtraining/data/)から送る
# (アブレーション用のスクラッチディレクトリはコードのみでデータを複製していないため)
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
REMOTE_DIR="/workspace/ghosts/training"
# uvはデフォルトでPythonインタプリタ・パッケージキャッシュを/root配下(コンテナ固有のエフェメラルディスク)に
# 置くため、Network Volumeを使い回して新しいpodを作っても再ダウンロードが発生する(.venvへのシンボリック
# リンクが指す先が新しいコンテナに存在しないため)。/workspace配下(Network Volume上)に固定することで、
# 同じVolumeを使う限りインタプリタ・キャッシュを再利用できるようにする(リモート側の環境変数として渡す)
UV_ENV_VARS="UV_PYTHON_INSTALL_DIR=/workspace/.uv-python UV_CACHE_DIR=/workspace/.uv-cache"

usage() {
  echo "Usage: $0 <ip> <port> [--with-data] [--source <local-dir>]" >&2
  echo "  --source: send scripts/pyproject.toml/uv.lock from a directory other than training/ (e.g. a scratch copy for a parallel ablation)" >&2
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
WITH_DATA=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-data)
      WITH_DATA="1"
      shift
      ;;
    --source)
      [ "$#" -ge 2 ] || usage
      SOURCE_DIR="$(cd "$2" && pwd)"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" "mkdir -p ${REMOTE_DIR}/data"

echo "Sending scripts/pyproject.toml/uv.lock from ${SOURCE_DIR} ..."
scp -i "${SSH_KEY}" -P "${POD_PORT}" -r \
  "${SOURCE_DIR}/scripts" "${SOURCE_DIR}/pyproject.toml" "${SOURCE_DIR}/uv.lock" \
  root@"${POD_IP}":"${REMOTE_DIR}/"

if [ -n "${WITH_DATA}" ]; then
  echo "Sending stroke_features_v2.npz ..."
  scp -i "${SSH_KEY}" -P "${POD_PORT}" \
    "${TRAINING_DIR}/data/stroke_features_v2.npz" root@"${POD_IP}":"${REMOTE_DIR}/data/"
fi

echo "Running uv sync ..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" \
  "export ${UV_ENV_VARS}; cd ${REMOTE_DIR} && uv sync"

echo "Checking CUDA is recognized ..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" \
  "export ${UV_ENV_VARS}; cd ${REMOTE_DIR} && uv run python3 -c 'import sys; sys.path.insert(0, \"scripts\"); from vae_model_v2 import select_device; print(select_device())'"
