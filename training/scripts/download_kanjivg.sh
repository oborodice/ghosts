#!/usr/bin/env bash
set -euo pipefail

# 「-all」を使う("-main"や"-stripped"ではない): 異体字も含めて全字種を対象にしたい上、
# 筆画種別・部首・構成要素の属性も欠けていないものが必要なため。

KANJIVG_VERSION="20250816"
RELEASE_TAG="r${KANJIVG_VERSION}"
ASSET="kanjivg-${KANJIVG_VERSION}-all.zip"
SHA256="9485959d64ac64da315e88c8dcd9209e7ec0eeaeec20121c760517d4d30f7cb9"
URL="https://github.com/KanjiVG/kanjivg/releases/download/${RELEASE_TAG}/${ASSET}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/kanjivg"

if [ -n "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]; then
  echo "KanjiVG data already exists at ${DATA_DIR}, skipping download."
  exit 0
fi

mkdir -p "${DATA_DIR}"
TMP_ZIP="$(mktemp -t kanjivg.XXXXXX.zip)"
trap 'rm -f "${TMP_ZIP}"' EXIT

echo "Downloading ${ASSET} ..."
curl -fsSL -o "${TMP_ZIP}" "${URL}"

echo "${SHA256}  ${TMP_ZIP}" | shasum -a 256 -c -

echo "Extracting to ${DATA_DIR} ..."
unzip -q -j "${TMP_ZIP}" -d "${DATA_DIR}"

echo "Done: $(ls "${DATA_DIR}" | wc -l | tr -d ' ') SVG files in ${DATA_DIR}"
