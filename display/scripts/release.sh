#!/usr/bin/env bash
set -euo pipefail

# 表示の64ビットARMのLinux向けの実行ファイルを組み立て、GitHubのリリースに上げる。
# - 組み立てはDockerのarm64のDebian 13(trixie)の中で行う(Apple SiliconのMacではエミュレーションなしで動く)。
#   ortが取り込む組み立て済みのONNX Runtimeがglibc 2.38とGCC 14のC++の標準ライブラリを前提にしているので、
#   それより古いDebian 12では組み立てられず、動かす環境もDebian 13以降と同じ世代のOSが必要
# - 実行ファイルに埋め込むのと同じ生成器のONNX(training/data/onnx/glyph_generator.onnx)も、ライセンスの表示と一緒にアーカイブにして上げる
# - 版はCargo.tomlのversionから決め、mainの今のコミットに v<版> のタグが付いたリリースを作る
RUST_IMAGE="rust:1.94.0-trixie"
# ortが組み立てのときに実行ファイルへ取り込むONNX Runtimeの版(ortの版を変えたら合わせる)。そのライセンスの文を添えるのに使う
ONNXRUNTIME_VERSION="1.28.0"
WEIGHTS_LICENSE_URL="https://creativecommons.org/licenses/by-nc-sa/4.0/"
# Mac向けの target/ の中身と混ざらないよう、組み立ての結果を分けて置く(display/ からの相対)
LINUX_TARGET_DIR="target/linux-aarch64"

usage() {
  echo "Usage: $0 [--build-only]" >&2
  echo "  Builds the aarch64 Linux binary in Docker, packages it with its licenses," >&2
  echo "  packages the generator ONNX with its license notice, and creates a GitHub release tagged v<version in Cargo.toml>" >&2
  echo "  on the current commit of main with both archives" >&2
  echo "  --build-only: stop after packaging (no tag, no release; the working tree may have uncommitted changes)" >&2
  exit 1
}

BUILD_ONLY=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --build-only) BUILD_ONLY=true; shift ;;
    *) usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPLAY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${DISPLAY_DIR}/.." && pwd)"
MODEL="${REPO_DIR}/training/data/onnx/glyph_generator.onnx"
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "${DISPLAY_DIR}/Cargo.toml" | head -1)"
TAG="v${VERSION}"
OUTPUT_DIR="${DISPLAY_DIR}/target/release-package"
DISPLAY_PACKAGE="ghosts-display-${TAG}-aarch64-unknown-linux-gnu"
MODEL_PACKAGE="glyph-generator-${TAG}"

for command in docker cargo-about gh curl; do
  command -v "${command}" >/dev/null || { echo "${command} is not installed" >&2; exit 1; }
done
[ -f "${MODEL}" ] || { echo "${MODEL} not found (export it with training/scripts/export_onnx.py first)" >&2; exit 1; }

if [ "${BUILD_ONLY}" = false ]; then
  if [ "$(git -C "${REPO_DIR}" branch --show-current)" != "main" ]; then
    echo "Releases are made from main (check out main first)" >&2; exit 1
  fi

  # 次の2つで、組み立てるソースとタグが指すコミットを一致させ、そのコミットをGitHubに置いておく
  # (GPLで配るので、受け取った人がタグから同じソースを得られるように)
  if [ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]; then
    echo "The working tree has uncommitted changes" >&2; exit 1
  fi
  git -C "${REPO_DIR}" fetch -q origin
  if [ -z "$(git -C "${REPO_DIR}" branch -r --contains HEAD)" ]; then
    echo "The current commit is not pushed to origin" >&2; exit 1
  fi

  if git -C "${REPO_DIR}" rev-parse -q --verify "refs/tags/${TAG}" >/dev/null || [ -n "$(git -C "${REPO_DIR}" ls-remote --tags origin "${TAG}")" ]; then
    echo "Tag ${TAG} already exists (bump version in display/Cargo.toml)" >&2; exit 1
  fi
fi

echo "Building ${DISPLAY_PACKAGE} in ${RUST_IMAGE} (arm64)..."
# 依存のダウンロード(crates.io・ONNX Runtime)は名前付きのvolumeにキャッシュし、2回目から速くする
docker run --rm --platform linux/arm64 \
  -v "${REPO_DIR}:/work" -w /work/display \
  -v ghosts-display-cargo-registry:/usr/local/cargo/registry \
  -v ghosts-display-cache:/root/.cache \
  -e CARGO_TARGET_DIR="/work/display/${LINUX_TARGET_DIR}" \
  "${RUST_IMAGE}" bash -euo pipefail -c '
    apt-get update -qq && apt-get install -y -qq libsdl2-dev >/dev/null
    cargo build --release --locked
    binary="${CARGO_TARGET_DIR}/release/ghosts-display"
    "${binary}" --version
    echo "Newest glibc symbol required: $(objdump -T "${binary}" | grep -o "GLIBC_[0-9.]*" | sort -uV | tail -1)"
  '

# 各アーカイブの中身は、同じ名前のフォルダに集めてからまとめる(展開すると、そのフォルダが1つできる)
rm -rf "${OUTPUT_DIR}"

stage="${OUTPUT_DIR}/${DISPLAY_PACKAGE}"
mkdir -p "${stage}/onnxruntime"
cp "${DISPLAY_DIR}/${LINUX_TARGET_DIR}/release/ghosts-display" "${REPO_DIR}/LICENSE" "${stage}/"
(cd "${DISPLAY_DIR}" && cargo about generate --fail about.hbs -o "${stage}/THIRD_PARTY_LICENSES.txt")
for file in LICENSE ThirdPartyNotices.txt; do
  curl -sSfL "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_VERSION}/${file}" -o "${stage}/onnxruntime/${file}"
done
cat > "${stage}/NOTICE" <<EOF
The generator weights embedded in ghosts-display are licensed under CC BY-NC-SA 4.0 (${WEIGHTS_LICENSE_URL}).
EOF

stage="${OUTPUT_DIR}/${MODEL_PACKAGE}"
mkdir -p "${stage}"
cp "${MODEL}" "${stage}/"
cat > "${stage}/NOTICE" <<EOF
glyph_generator.onnx is licensed under CC BY-NC-SA 4.0 (${WEIGHTS_LICENSE_URL}).
EOF

ARCHIVES=()
for package in "${DISPLAY_PACKAGE}" "${MODEL_PACKAGE}"; do
  archive="${OUTPUT_DIR}/${package}.tar.gz"
  tar -czf "${archive}" -C "${OUTPUT_DIR}" "${package}"
  ARCHIVES+=("${archive}")
  echo "Saved ${archive} ($(du -h "${archive}" | cut -f1))"
done

if [ "${BUILD_ONLY}" = true ]; then
  exit 0
fi

echo "Creating the release ${TAG}..."
# タグは、リリースを作るときにGitHubの側で付ける(先に手元でタグを付けてpushすると、リリースの作成に失敗したときにタグだけが残る)
(cd "${REPO_DIR}" && gh release create "${TAG}" "${ARCHIVES[@]}" --target "$(git rev-parse HEAD)" --title "${TAG} ($(date +%Y-%m-%d))" --notes "")
git -C "${REPO_DIR}" fetch -q --tags origin
