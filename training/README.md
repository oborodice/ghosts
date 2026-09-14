# 学習側

- 幽霊文字の生成モデルの学習コード

## セットアップ

```sh
# 依存パッケージのインストール
$ uv sync

# KanjiVG (https://kanjivg.tagaini.net/, CC BY-SA 3.0) データセットの取得
# data/kanjivg/配下にSVGファイルが展開される(取得済みの場合は再ダウンロードをスキップする)
$ ./scripts/download_kanjivg.sh
```

## 実行手順

```sh
# ダウンロードしたSVGのストロークデータを、いくつかの漢字を選んで可視化する
$ uv run scripts/view_kanji.py

# 各漢字の画数を集計し、分布(最小/最大/平均/パーセンタイル)を表示する
$ uv run scripts/stroke_count_stats.py

# 各漢字のストロークを、折れ・ハネで分割したセグメント単位の固定長テンソルに変換し、セグメント同士の
# 接続関係を表す接続行列とあわせてdata/stroke_features.npzへ保存する(セグメント数がスロット数(26)を
# 超える漢字は除外する)
$ uv run scripts/extract_stroke_features.py

# 各漢字のストロークを、接続されている端点同士をクリークでクラスタ化した「頂点」テーブルと、
# 頂点ペアを参照する「ストローク」テーブルに変換し、data/stroke_features_v2.npzへ保存する
# (同じ頂点を参照するストローク同士は必ず接続している、という保証をデータ構造として持たせるための
# 表現。漢字以外のグリフ(ひらがな・カタカナ等)は除外する)
$ uv run scripts/extract_stroke_features_v2.py

# VAEを学習し、data/checkpoints/vae.ptへ保存する
$ uv run scripts/train_vae.py

# 収束済みのvae.ptに対し、ストローク数の多様性・長さ・曲がり具合・交差数・3本以上合流を実データに
# 近づける追加の微調整を行い、data/checkpoints/vae.ptを上書き保存する
# (2段階目。5軸の複合乖離が最小の時点でearly stoppingする)
$ uv run scripts/finetune_synthetic_stats.py

# 学習済みモデル(vae.pt)の品質を数値で確認する
# (損失の内訳、潜在次元ごとのKL、重みの健全性、validation全体の誤差分布、丸暗記化していないかの確認、
# 重複スロットの検出)
$ uv run scripts/evaluate_vae.py

# 頂点のみの最小構成(頂点座標MSE+existence BCE+KL)でVAEを学習し、data/checkpoints/vae_v2.ptへ保存する
$ uv run scripts/train_vae_v2.py

# 学習済みモデル(vae_v2.pt)の品質を数値で確認する
# (損失の内訳、潜在次元ごとのKL、重みの健全性、validation全体の誤差分布(標準化スケール・実座標スケール)、
# 丸暗記化していないかの確認、重複スロットの検出)
$ uv run scripts/evaluate_vae_v2.py

# 頂点のみの段階でハイパーパラメータ候補(KLの重み・warm-up速度・潜在次元数)を複数比較する。
# 候補ごとにdata/checkpoints/vae_v2_sweep_<候補名>.ptへ保存し、beta非依存の指標(実座標スケールの
# 頂点距離・dead次元数)で比較する
$ uv run scripts/sweep_vae_v2.py

# validationサンプルの元データと再構成結果(model.decode(mu))を頂点の点群として並べて目視確認する。
# 典型的な4字に加え、再構成誤差が最悪だった字も表示する
$ uv run scripts/visualize_reconstruction_v2.py

# validationサンプルの元データと再構成結果(model.decode(mu))を並べて目視確認する。
# 接続点周辺は文字全体の表示では崩れが見えづらいため、個別にズームインした図も表示する
$ uv run scripts/visualize_reconstruction.py

# 本物(実データ)/偽物(生成結果)を判別する小さい分類器を学習させ、その正答率で生成結果が統計的に
# どれだけ実在字らしいかを診断する(人間が名指しした軸に限らず、未知の差も拾い上げられる)
$ uv run scripts/evaluate_generation_realism.py

# 本物/偽物を、数値特徴量ではなく実際にレンダリングした画像で判別する診断分類器。
# 数値特徴量では見えている差異が、人間の視覚に近い形(低解像度・軽いぼかし)でも見分けられるかを確認する
$ uv run scripts/evaluate_generation_realism_visual.py

# 生成結果の各軸(ストローク数・長さ・曲がり具合・面積・軸方向率・孤立率・交差・3本以上合流・
# 同方向ストロークの束)を実データと揃えた方法で数値化する
$ uv run scripts/report_generation_stats.py

# 事前分布N(0,I)からサンプリングし、実データへカーネル重み付けで引き寄せた後のzのdecode結果を目視確認する
$ uv run scripts/visualize_prior_samples.py

# validationの2サンプル間を潜在空間上で線形補間し、字形が滑らかに変化するか目視確認する
$ uv run scripts/visualize_latent_interpolation.py

# 学習済みモデル(vae.pt)をONNX形式でエクスポートし、web/public/vae.onnxへ保存する
# (web側で読み込めるよう、生成用zの実データへのカーネル重み付け・decode・標準化の逆変換・existenceの
# Sigmoidまでを1つのグラフに含める)
$ uv run scripts/export_onnx.py
```
