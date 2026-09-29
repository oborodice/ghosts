# 表示側

- 生成器のONNXをONNX Runtime(CPU)で1フレームずつ動かし、形を変え続ける字を、GPUのシェーダーでフィルタをかけて描き続ける
- 学習・評価・ONNXへの書き出しは [学習側](../training/README.md) で行う
- 組み立てのときに、学習側で書き出した生成器( `training/data/onnx/glyph_generator.onnx` )を実行ファイルに埋め込む

## 構成

- `src/` : Rustのソース
- `shaders/` : フィルタのシェーダー(実行ファイルに埋め込む)
- `scripts/` : リリースのスクリプト

## セットアップ

```sh
# ウィンドウとOpenGLの文脈を作るのに使う
$ brew install sdl2

# 録画(--features record)で、mp4をx264で書き出すのに使う
$ brew install ffmpeg

# リリースで、64ビットARMのLinux向けの実行ファイルを組み立てるのに使う(Docker Desktop)
$ brew install --cask docker

# リリースで、実行ファイルに取り込むライブラリのライセンスの文をまとめるのに使う
$ brew install cargo-about

# リリースで、GitHubのリリースを作るのに使う(初回はログインする)
$ brew install gh
$ gh auth login
```

## 実行手順

```sh
# ウィンドウで表示する。1秒ごとに、FPSと、生成器・描画の1フレームの時間を表示する。Escかqで終わる
$ cargo run --release

# 全画面で表示する
$ cargo run --release -- --fullscreen

# 埋め込んだ生成器の代わりに別の生成器のONNXを使い、字の変わり方(simplex noiseの軌跡)を乱数の種で変える
$ cargo run --release -- --model <ONNXのパス> --seed 1

# ウィンドウを実寸(1280x720)で開く(既定は、開発機のMacで実物に近い見かけになるよう0.6倍に縮める)
$ cargo run --release -- --scale 1

# 録画(Macでの確認・共有のため、--features record で組み込む。配る実行ファイルには入れない)。
# ウィンドウを出さずに、30秒ぶんを動画に書き出す(画面全体を実寸の1280x720で)
$ cargo run --release --features record -- --record <mp4のパス> --seconds 30

# ウィンドウを出さずに、10秒ぶんをGIFに書き出す(字の正方形だけを350pxで)
$ cargo run --release --features record -- --record <GIFのパス> --seconds 10
```

## リリース

```sh
# 64ビットARMのLinux向けの実行ファイルをDocker(Debian 13)の中で組み立て、ライセンスの文と一緒にアーカイブにまとめる。
# 生成器のONNXもライセンスの表示と一緒にアーカイブにし、mainの今のコミットに v<Cargo.toml の version> のタグが付いたGitHubのリリースに2つのアーカイブを上げる
# (動かす環境は、組み立て済みのONNX Runtimeの都合でDebian 13以降と同じ世代のOSが必要)
$ ./scripts/release.sh

# タグとリリースは作らず、2つのアーカイブを target/release-package/ に作るところまで行う
$ ./scripts/release.sh --build-only
```

## ライセンス

- このディレクトリのコードが使う、第三者のコード

|対象|ライセンス|利用箇所|
|---|---|---|
|[hash-prospector](https://github.com/skeeto/hash-prospector) の整数のハッシュ `lowbias32` (関数をそのままコピーしたもの)|The Unlicense|`shaders/glyph.frag`|
