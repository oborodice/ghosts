# 学習側

- 幽霊文字の生成モデルの学習コード

## セットアップ

```sh
# 依存パッケージのインストール
$ uv sync

# KanjiVG (https://kanjivg.tagaini.net/, CC BY-SA 3.0) データセットの取得
# data/kanjivg/配下にSVGファイルが展開される(取得済みの場合は再ダウンロードをスキップする)
$ ./scripts/download_kanjivg.sh

# RunPodへのデプロイスクリプト(scripts/runpod_*.sh)が、pod作成時のJSON出力から
# id・IP・ポートを取り出すために使う
$ brew install jq
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

# 各漢字のストロークを、接続されている端点同士をクリークでクラスタ化した「頂点」テーブルと、
# 頂点ペアを参照する「ストローク」テーブルに変換し、data/stroke_features_v2.npzへ保存する
# (同じ頂点を参照するストローク同士は必ず接続している、という保証をデータ構造として持たせるための
# 表現。漢字以外のグリフ(ひらがな・カタカナ等)は除外する)
$ uv run scripts/extract_stroke_features_v2.py

# 頂点+ストローク全体(頂点座標のビン分類cross entropy+existence BCE+ストロークのポインタ分類cross entropy+
# オフセットMSE+existence BCE+KL)でVAEを学習する。保存先は既存ファイルとの衝突を避けるため起動時刻ベースの
# ファイル名(data/checkpoints/vae_<timestamp>.pt)になる(他のスクリプトが読むvae_v2.ptを
# 更新する場合は、確認の上で手動でコピー・リネームする)
$ uv run scripts/train_vae_v2.py
# 学習するエポック数の上限を変える(動作確認などで、早期終了を待たずに短く止めたい場合)
$ uv run scripts/train_vae_v2.py --max-epochs 40
# 途中で落ちた学習を、同時に保存される<出力先>_resume.ptから再開する
$ uv run scripts/train_vae_v2.py --resume data/checkpoints/vae_<timestamp>_resume.pt

# 学習済みモデル(vae_v2.pt)の品質を数値で確認する
# (再構成側の損失の内訳(学習と同じ関数で計算した、重みを掛ける前の値。混ぜ合わせたzの損失は含まない)、
# 潜在次元ごとのKLの要約、重みの健全性、validation全体の誤差分布(頂点の標準化スケール・実座標スケール、
# ストロークのポインタ分類精度・オフセットMSE)、丸暗記していないかの確認、頂点の重複スロットの検出)
$ uv run scripts/evaluate_vae_v2.py
# 評価対象のVAEチェックポイントを指定する(省略時は既定のチェックポイント。`--checkpoint`を持つ他のスクリプトも同じ)
$ uv run scripts/evaluate_vae_v2.py --checkpoint data/checkpoints/<name>.pt

# validationサンプルの元データと再構成結果(model.decode(mu))をストロークの曲線として並べて目視確認する。
# 典型的な4字に加え、頂点の再構成誤差が最悪だった字も表示する
$ uv run scripts/visualize_reconstruction_v2.py

# 生成(事前分布サンプル+LatentSampler)時の孤立率・3本以上合流・交差・角度の
# 自然さ・offset分散・ストローク長・ストローク数・キャンバス占有率を、実データ・reconstructionと
# 比較できる一貫した方法で数値化する。あわせて、reconstructionの頂点再構成誤差(実スケール)、
# 頂点の重複スロットが3本以上合流のカウントを狂わせていないか、ポインタの構造的な破綻
# (自己ループ・幽霊参照)の頻度、生成したzのnear_dup_rate(実在字とほぼ重複している
# 割合)・effective_k(実質何字を混ぜて作られているか)も確認する。後者2つは、合成z領域に新しい
# 損失を試す際、目的の指標の改善がencoder表現の崩壊の副産物でないかを切り分けるための診断。
# 生成分は、ポインタ・頂点座標を確率加重平均で選ぶソフトデコード(`GENERATION_SOFT_TEMPERATURE`)で作る
$ uv run scripts/report_generation_stats_v2.py
$ uv run scripts/report_generation_stats_v2.py --checkpoint data/checkpoints/<name>.pt

# 生成した字の、ストローク数の分布・字の大きさ・実在字との近さ・構造の破綻(自己ループなど)・なめらかさを、
# 実在字と比べて数値化する(ストローク数の分布・字の大きさなどの実在字とのWasserstein距離、実在字までの最寄り距離、
# 自己ループなど、1字ごとの合格率(参考)、軌跡の上でのジャンプ率・ストローク数の変化)
$ uv run scripts/report_generation_diversity_v2.py
$ uv run scripts/report_generation_diversity_v2.py --checkpoint data/checkpoints/<name>.pt

# 潜在変数をsimplex noiseで動かしたときの、生成結果のなめらかさを数値化する。フレーム間の端点のジャンプ率
# (信頼区間つき)、zの移動距離あたりのジャンプ回数、ストロークの出現・消失の頻度を出す。
# 生成時のソフトデコード(`GENERATION_SOFT_TEMPERATURE`)で、ノイズの周期30秒・60秒の2通りを測る
$ uv run scripts/report_morph_smoothness_v2.py
$ uv run scripts/report_morph_smoothness_v2.py --checkpoint data/checkpoints/<name>.pt

# simplex noiseで動かした潜在変数から生成した字の変化を、アニメーションGIFとして書き出す目視確認用のツール。
# なめらかさの診断(report_morph_smoothness_v2.py)と同じ軌跡・デコード(周期30秒)で作る。書き出し先の指定は必須
$ uv run scripts/visualize_morph_gif_v2.py --output morph.gif
$ uv run scripts/visualize_morph_gif_v2.py --output morph.gif --checkpoint data/checkpoints/<name>.pt

# 本物/偽物を、数値特徴量ではなく実際にレンダリングした画像で判別する診断分類器。頂点+ストローク構造
# (vae_v2.pt)の生成結果に対して使う。数値特徴量では見えている差異が、人間の視覚に近い形
# (低解像度・軽いぼかし)でも見分けられるかを確認する
$ uv run scripts/evaluate_generation_realism_visual_v2.py
$ uv run scripts/evaluate_generation_realism_visual_v2.py --checkpoint data/checkpoints/<name>.pt
# 学習した分類器を保存する(省略時は保存しない)
$ uv run scripts/evaluate_generation_realism_visual_v2.py --save-model data/checkpoints/<classifier>.pt

# evaluate_generation_realism_visual_v2.pyで`--save-model`保存した分類器が、実データ/生成結果を
# 何を根拠に見分けているかを分析する。テストサンプルごとの分類確率と、triple_junctions・ストローク長・
# offsetのサンプル内ばらつきといった既知指標、ストローク数・総ストローク長といった単純な交絡との
# 相関(ピアソン相関係数)を計算する。分類器の指定は必須
$ uv run scripts/analyze_classifier_scores_v2.py --classifier data/checkpoints/<classifier>.pt
$ uv run scripts/analyze_classifier_scores_v2.py --classifier data/checkpoints/<classifier>.pt --checkpoint data/checkpoints/<name>.pt

# 頂点のみだった段階でハイパーパラメータ候補(KLの重み・warm-up速度・潜在次元数)を複数比較したスクリプト。
# モデルの形状(頂点+ストローク対応)が変わったため現在は実行できない
$ uv run scripts/sweep_vae_v2.py
```

## RunPodへのデプロイ

- 外部GPU( [RunPod](https://www.runpod.io/) 、RTX 4090)でフルスケール学習を実行するためのスクリプト
- 初回のみ、RunPodアカウントの作成、 `runpodctl` のセットアップ(APIキー・SSH鍵)、Network Volumeの作成が別途必要

```sh
# pod作成(Network Volume ID指定)。SSH接続確認・マウント確認まで行い、pod_id/ip/portを表示する。
# GPU機種が同じでもCPU側の当たり外れがあるため、シングルスレッドの簡易ベンチマークも実行する
$ ./scripts/runpod_create_pod.sh <network-volume-id> [pod-name] [gpu-id]
$ ./scripts/runpod_create_pod.sh <network-volume-id>
# GPU機種を指定する(省略時はRTX 4090。在庫切れの場合に使う。このワークロードはCPUがボトルネックの
# ため、機種を変えてもエポック時間はほぼ変わらない)
$ ./scripts/runpod_create_pod.sh <network-volume-id> ghosts-a "NVIDIA RTX PRO 4000 Blackwell"

# コード(scripts・pyproject.toml・uv.lock)を転送しuv syncする。
# このワークロードはGPUを数%しか使わないため、1つのpodに複数構成を同時に置いて並列実行できる。
# ただしCPUは1プロセスあたり約3〜4コアを使う(大容量の構成での実測)ため、並列数はpodのCPUの上限
# (nprocではなく/sys/fs/cgroup/cpu.maxで確認する)で決まる
$ ./scripts/runpod_deploy_code.sh <ip> <port> [--with-data] [--source <local-dir>] [--remote-dir <path>] [--link-venv <remote-dir>]
$ ./scripts/runpod_deploy_code.sh <ip> <port>
# データも送る(初回のみ)
$ ./scripts/runpod_deploy_code.sh <ip> <port> --with-data
# 本番のtraining/scripts以外(アブレーション用のスクラッチコピーなど)を送る
$ ./scripts/runpod_deploy_code.sh <ip> <port> --source <local-dir>
# 同じpodに、別の配置先として並べて置く(配置先ごとにdata/を持つため--with-dataも必要)。
# 既に同期済みの配置先の.venvをシンボリックリンクすれば、依存関係が同じなら再ダウンロードなしで済む
$ ./scripts/runpod_deploy_code.sh <ip> <port> --with-data --remote-dir /workspace/ghosts/training_b --link-venv /workspace/ghosts/training

# 学習をnohup+disownでバックグラウンド起動する
$ ./scripts/runpod_launch_training.sh <ip> <port> [remote-resume-path] [--remote-dir <path>] [--train-args "<args>"]
$ ./scripts/runpod_launch_training.sh <ip> <port>
# 中断した学習を再開する
$ ./scripts/runpod_launch_training.sh <ip> <port> /workspace/ghosts/training/data/checkpoints/vae_<timestamp>_resume.pt
# 別の配置先(runpod_deploy_code.shの--remote-dir)で起動する
$ ./scripts/runpod_launch_training.sh <ip> <port> --remote-dir /workspace/ghosts/training_b
# train_vae_v2.pyに渡す追加の引数を、1つの文字列にまとめて指定する
$ ./scripts/runpod_launch_training.sh <ip> <port> --train-args "--max-epochs 40"

# 学習プロセスが動いているか・train.logの直近n行を確認する(train.logはNetwork Volume単位で
# 永続化されるため、プロセスが動いていないのに前回の内容が表示されることがある点に注意)
$ ./scripts/runpod_check_progress.sh <ip> <port> [n-lines] [--remote-dir <path>]
$ ./scripts/runpod_check_progress.sh <ip> <port>
# 表示する行数を指定する
$ ./scripts/runpod_check_progress.sh <ip> <port> 50
# 別の配置先を確認する
$ ./scripts/runpod_check_progress.sh <ip> <port> --remote-dir /workspace/ghosts/training_b

# チェックポイント・train.logをダウンロードする。
# ダウンロード成功後、Network Volume上の当該ディレクトリの*.ptを全て削除してクォータを空ける
# (podを使い回すたびに複数世代のチェックポイントが積み上がりクォータ超過する事故を防ぐため)
$ ./scripts/runpod_download_results.sh <ip> <port> [remote-checkpoint-name|latest] [local-name] [--remote-dir <path>]
# 最新のチェックポイントを取得する
$ ./scripts/runpod_download_results.sh <ip> <port>
# 最新のチェックポイントを、ローカルでの名前を指定して取得する
$ ./scripts/runpod_download_results.sh <ip> <port> latest vae_v2_<name>.pt
# 別の配置先から取得する
$ ./scripts/runpod_download_results.sh <ip> <port> latest vae_v2_<name>.pt --remote-dir /workspace/ghosts/training_b

# podを削除して課金を止める(Network Volumeは残る)
$ ./scripts/runpod_terminate.sh <pod-id>
```
