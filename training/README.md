# 学習側

- 幽霊文字の生成モデルの学習コード

## セットアップ

```sh
# 依存パッケージのインストール
$ uv sync

# KanjiVG (https://kanjivg.tagaini.net/, CC BY-SA 3.0) データセットの取得
# data/kanjivg/配下にSVGファイルが展開される(取得済みの場合は再ダウンロードをスキップする)
$ ./scripts/download_kanjivg.sh

# ダウンロードしたSVGのストロークデータを、いくつかの漢字を選んで可視化する
$ uv run scripts/view_kanji.py

# 各漢字の画数を集計し、分布(最小/最大/平均/パーセンタイル)を表示する
$ uv run scripts/stroke_count_stats.py

# 各漢字のストロークを固定長テンソルに変換し、data/stroke_features.npzへ保存する
# (画数がスロット数(22)を超える漢字は除外する)
$ uv run scripts/extract_stroke_features.py

# フェーズ1のVAEを学習し、data/checkpoints/vae_phase1.ptへ保存する
$ uv run scripts/train_vae.py

# 学習済みモデル(vae_phase1.pt)の品質を数値で確認する
# (損失の内訳、潜在次元ごとのKL、重みの健全性、validation全体の誤差分布)
$ uv run scripts/evaluate_vae.py

# validationサンプルの元データと再構成結果(model.decode(mu))を並べて目視確認する
$ uv run scripts/visualize_reconstruction.py
```
