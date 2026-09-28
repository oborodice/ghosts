# 表示側

- 幽霊文字の表示(Raspberry Pi 5 + Touch Display 2 で動かす本番の表示)
- 生成器のONNXをONNX Runtime(CPU)で1フレームずつ動かし、形を変え続ける字を、GPUのシェーダーでフィルタをかけて描き続ける
- 学習・評価・ONNXへの書き出しは [学習側](../training/README.md) で行う
- 組み立てのときに、学習側で書き出した生成器( `training/data/onnx/glyph_generator.onnx` )を実行ファイルに埋め込む

## 構成

- `src/` : Rustのソース
- `shaders/` : フィルタのシェーダー(実行ファイルに埋め込む)

## セットアップ

```sh
# ウィンドウとOpenGLの文脈を作るのに使う
$ brew install sdl2

# 録画(--features record)で、mp4 を x264 で書き出すのに使う
$ brew install ffmpeg
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

# 録画(Macでの確認・共有のため、--features record で組み込む。Piで動かす本番の実行ファイルには入れない)。
# ウィンドウを出さずに、30秒ぶんを動画に書き出す(画面全体を実寸の1280x720で)
$ cargo run --release --features record -- --record <mp4のパス> --seconds 30

# ウィンドウを出さずに、10秒ぶんをGIFに書き出す(字の正方形だけを350pxで)
$ cargo run --release --features record -- --record <GIFのパス> --seconds 10
```

## ライセンス

- このディレクトリのコードが使う、第三者のコード

|対象|ライセンス|利用箇所|
|---|---|---|
|[hash-prospector](https://github.com/skeeto/hash-prospector) の整数のハッシュ `lowbias32` (関数をそのままコピーしたもの)|The Unlicense|`shaders/glyph.frag`|
