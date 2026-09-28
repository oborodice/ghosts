# 学習側

- 幽霊文字の生成モデルの学習コード

## セットアップ

```sh
# 依存パッケージのインストール
$ uv sync

# KanjiVG (https://kanjivg.tagaini.net/, CC BY-SA 3.0) データセットの取得
# data/kanjivg/配下にSVGファイルが展開される(取得済みの場合は再ダウンロードをスキップする)
$ ./scripts/download_kanjivg.sh

# 学習データ(フォントで描いた漢字の画像)に使う27書風のフォント(Google Fonts、OFL / Apache License 2.0)の取得
# data/fonts/配下に、フォントと各ファミリーのライセンスのファイルが置かれる。取得元のコミットを固定し、
# 各フォントをgitのblobの識別子で照合する(照合済みのファイルがある場合は再ダウンロードをスキップする)
$ uv run scripts/download_fonts.py

# RunPodへのデプロイスクリプト(scripts/runpod_*.sh)が、pod作成時のJSON出力から
# id・IP・ポートを取り出すために使う
$ brew install jq
```

## 実行手順(フォントで描いた漢字の画像で学習するGAN)

```sh
# 27書風のフォントで、各フォントが持っている漢字をすべて1字ずつ描き、data/glyphs_<解像度>.npzへ保存する
# (download_fonts.pyの取得が前提)。字の数はフォントによって違う(標準的なフォントで約6,700字、多いもので
# 約14,200字)。入れるのはCJK統合漢字(基本の範囲と拡張A〜J)と、CJK互換漢字のうち統合漢字と重複しない字・
# 統合漢字と違う形(旧字形)に描かれる字。部首や筆画の記号、かな、記号、IVSの字形違いは入れない(詳しくは
# スクリプトの冒頭のコメント)。字はフォント本来の大きさで描き、インクの外接の枠の中心を画像の中心に合わせる。
# 画像はインクの濃さ(0=紙〜255=インク)で、黒地に白の字として見える向き。中身が空の字と、そのフォントの
# 豆腐(.notdef)と同じ形になる字は飛ばす
$ uv run scripts/build_glyph_dataset.py

# 解像度を変える(省略時は64)
$ uv run scripts/build_glyph_dataset.py --resolution 128

# 学習データ(data/glyphs_64.npz)でGAN(写像ネットワーク・生成器・判別器、scripts/glyph_gan.py)を学習し、
# data/checkpoints/glyph_gan/<名前>/checkpoint_<歩数>.ptへ保存する。チェックポイントには、推論に使う移動平均の版の重みと、
# 再開に使う最適化の状態も入る。既定の設定は、同じ字ばかり出る崩壊と長い学習での崩れを避けるために確かめた構成
# (ロジスティック損失、判別器の学習率=生成器、写像ネットワークの学習率の倍率0.01、強い勾配の罰則、左右反転なしの増強)。
# 学習中は、スタイルの散らばりと崩れ(外周の枠・白黒の反転・塗りつぶし)の割合を定期的に表示する。
# GPU(CUDA)、Apple Silicon(MPS)、CPUの順に使えるものを使う
$ uv run scripts/train_glyph_gan.py --name <名前>

# 学習の長さ(千枚単位、省略時は1920 = バッチ64で3万歩)と乱数の種を指定する
$ uv run scripts/train_glyph_gan.py --name <名前> --kimg 960 --seed 1

# 途中で止まった学習を、同じ名前の最新のチェックポイントから続ける(上書きした設定も、保存した値に戻る)
$ uv run scripts/train_glyph_gan.py --name <名前> --resume

# 学習の設定(スクリプトの冒頭の定数)を、この学習だけ変える(スイープ用。何度でも指定できる。使った値はチェックポイントに残る)
$ uv run scripts/train_glyph_gan.py --name <名前> --override LEARNING_RATE=1e-4 --override CAPACITY=16

# 学習中に、文字認識のモデルで2字の種類の数(同じ字ばかり出る崩壊の目安)も表示する
$ uv run scripts/train_glyph_gan.py --name <名前> --classifier <文字認識のモデルの重み>

# 別の解像度の学習データで学習する(省略時はdata/glyphs_64.npz)
$ uv run scripts/train_glyph_gan.py --name <名前> --data data/glyphs_128.npz

# 生成した字の評価の物差しにする文字認識のモデルを学習し、data/glyph_classifier.ptへ保存する。
# 3書風を学習から外し、見たことのない書風でも読めるかを確かめる。学習のあと、実在字で特徴の距離の分布を測り、
# 評価に使うしきい値(同じ種類の字とみなす距離、別の字への急な切り替わりとみなす距離)も一緒に保存する
$ uv run scripts/train_glyph_classifier.py

# 学習データと保存先を指定する
$ uv run scripts/train_glyph_classifier.py --data data/glyphs_128.npz --output data/glyph_classifier_128.pt

# 学習したチェックポイント(推論に使う移動平均の版の重み)を評価する。新しさ(知らない字になっているか)、
# 2字の種類の数(同じ字ばかり出ていないか)、崩れの割合、精度・再現率・密度・網羅率(実在字の分布との重なり。
# 実在字どうしの値を並べる)、なめらかさ(表示側と同じsimplex noiseの軌跡での、急な切り替わりの割合)を表示し、
# 生成した字の一覧の画像をチェックポイントの隣に保存する
$ uv run scripts/evaluate_glyph_gan.py --checkpoint data/checkpoints/glyph_gan/<名前>/checkpoint_<歩数>.pt

# 字の数(省略時は生成物・実在字それぞれ1万字)と軌跡の数(省略時は64本)を減らして、手早く確かめる
$ uv run scripts/evaluate_glyph_gan.py --checkpoint <チェックポイント> --samples 1000 --walks 4

# 文字認識のモデル・実在字のデータ・乱数の種を指定する
$ uv run scripts/evaluate_glyph_gan.py --checkpoint <チェックポイント> --classifier <文字認識のモデルの重み> --data <学習データ> --seed 1

# 学習したチェックポイントの生成器を、data/onnx/glyph_generator.onnxへ書き出す(推論には移動平均の版の重みを使う)。
# 入力はsimplex noiseの値で、正規分布への変換 → 写像ネットワーク → 生成器 → インクの画像。
# 書き出したあと、同じ入力でPyTorchとONNX Runtimeの出力を比べる
$ uv run scripts/export_glyph_onnx.py --checkpoint data/checkpoints/glyph_gan/<名前>/checkpoint_<歩数>.pt

# 書き出し先を指定する
$ uv run scripts/export_glyph_onnx.py --checkpoint <チェックポイント> --output <ONNXのパス>

# 書き出した生成器をONNX Runtime(CPU)で1フレームずつ動かし、GPUのシェーダーでフィルタ(輪郭のやわらげとくっきりさせる処理、
# 色づけ、発光、境界を背景に溶け込ませるノイズ、背景のノイズ)をかけて、形を変え続ける字をウィンドウに表示し続ける
# (ウィンドウはRaspberry Pi Touch Display 2を横向きにした1280x720を、既定では0.6倍に縮めた大きさで、字は中央に描く)。
# 1秒ごとに、FPSと、生成器・描画の1フレームの時間を表示する。Escかqで終わる
$ uv run scripts/run_glyph_onnx.py

# 全画面で表示する
$ uv run scripts/run_glyph_onnx.py --fullscreen

# 別の生成器のONNXを使い、字の変わり方(simplex noiseの軌跡)を乱数の種で変える
$ uv run scripts/run_glyph_onnx.py --model <ONNXのパス> --seed 1

# ウィンドウを実寸(1280x720)で開く(既定は、開発機のMacで実物に近い見かけになるよう0.6倍に縮める)
$ uv run scripts/run_glyph_onnx.py --scale 1

# ウィンドウを出さずに、30秒ぶんを動画に書き出す(画面全体を実寸の1280x720で)
$ uv run scripts/run_glyph_onnx.py --record <mp4のパス> --seconds 30

# ウィンドウを出さずに、10秒ぶんをGIFに書き出す(字の正方形だけを350pxで)
$ uv run scripts/run_glyph_onnx.py --record <GIFのパス> --seconds 10
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
# 表現。漢字以外のグリフ(ひらがな・カタカナ等)と、同じ字の字形バリアントのファイル(楷書の字形・書き順違い等)は
# 除外する)
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

- 外部GPU( [RunPod](https://www.runpod.io/) )でGANの学習(scripts/train_glyph_gan.py)を実行するためのスクリプト
- 初回のみ、RunPodアカウントの作成、 `runpodctl` のセットアップ(APIキー・SSH鍵)が別途必要(Network Volumeを使う場合はその作成も)
- podの作成は課金が発生するので1つずつ行い、作れなかったときは `runpodctl pod list` で作られていないことを確かめてから、条件を変えて作り直す

```sh
# pod作成。SSH接続の確認と、ホストのCPUの速さの簡易ベンチマーク(学習の速さはCPU側で決まりやすく、同じGPUの
# 機種でもホストによって違う)まで行い、pod_id/ip/portを表示する。Network Volumeを使わない場合は、pod自身の
# ディスク(podを消すと中身も消える)を使い、空きのあるデータセンターをRunPodに選ばせる
$ ./scripts/runpod_create_pod.sh [--volume <network-volume-id>] [--gpu <gpu-id>] [--name <pod-name>]
$ ./scripts/runpod_create_pod.sh
# GPU機種を指定する(省略時はRTX 4090。在庫切れの場合に使う)
$ ./scripts/runpod_create_pod.sh --gpu "NVIDIA L40S"
# Network Volumeを/workspaceに付ける(そのVolumeのデータセンターで作るので、GPUの空きがないと作れないことがある)
$ ./scripts/runpod_create_pod.sh --volume <network-volume-id>

# コード(scripts・pyproject.toml・uv.lock)を転送しuv syncして、CUDAが使えるかを確かめる
$ ./scripts/runpod_deploy_code.sh <ip> <port> [--data <file>] [--classifier <file>] [--source <local-dir>] [--remote-dir <path>] [--link-venv <remote-dir>]
$ ./scripts/runpod_deploy_code.sh <ip> <port>
# 学習データ(training/data/配下のファイル)も送る(初回のみ)
$ ./scripts/runpod_deploy_code.sh <ip> <port> --data glyphs_64.npz
# 学習中に種類の数を測る文字認識のモデルの重み(training/data/配下のファイル)も送る
$ ./scripts/runpod_deploy_code.sh <ip> <port> --data glyphs_64.npz --classifier <文字認識のモデルの重み>
# 本番のtraining/scripts以外(アブレーション用のスクラッチコピーなど)を送る
$ ./scripts/runpod_deploy_code.sh <ip> <port> --source <local-dir>
# 同じpodに、別の配置先として並べて置く(配置先ごとにdata/を持つため--dataも必要)。
# 既に同期済みの配置先の.venvをシンボリックリンクすれば、依存関係が同じなら再ダウンロードなしで済む
$ ./scripts/runpod_deploy_code.sh <ip> <port> --data glyphs_64.npz --remote-dir /workspace/ghosts/training_b --link-venv /workspace/ghosts/training

# 学習をバックグラウンドで始める(SSHを切っても止まらない)。ログは学習の名前ごとにtrain_<名前>.logへ書くので、
# 1つのpodで複数の学習を同時に回せる
$ ./scripts/runpod_launch_training.sh <ip> <port> --name <名前> [--resume] [--remote-dir <path>] [--train-args "<args>"]
$ ./scripts/runpod_launch_training.sh <ip> <port> --name <名前>
# train_glyph_gan.pyに渡す追加の引数を、1つの文字列にまとめて指定する
$ ./scripts/runpod_launch_training.sh <ip> <port> --name <名前> --train-args "--kimg 960 --seed 1"
# 止まった学習を、最新のチェックポイントから続ける
$ ./scripts/runpod_launch_training.sh <ip> <port> --name <名前> --resume

# 学習が動いているか・ログの直近n行・最新の指標(スタイルの散らばり、崩れの割合など)を確認する
$ ./scripts/runpod_check_progress.sh <ip> <port> --name <名前> [--lines <n>] [--remote-dir <path>]
$ ./scripts/runpod_check_progress.sh <ip> <port> --name <名前>
# 表示する行数を指定する
$ ./scripts/runpod_check_progress.sh <ip> <port> --name <名前> --lines 50

# 学習の結果(チェックポイントのディレクトリとログ)を、ローカルのdata/checkpoints/glyph_gan/<名前>/へ落とす。
# ファイルの数と中身(SHA-256)をpodの上と比べ、すべて一致したときだけpodの上のチェックポイントを消す。
# podを消す前に必ず実行する(Network Volumeを使わないpodでは、podを消すと中身も消えるため)
$ ./scripts/runpod_download_results.sh <ip> <port> --name <名前> [--keep-remote] [--remote-dir <path>]
$ ./scripts/runpod_download_results.sh <ip> <port> --name <名前>

# podを削除して課金を止める(Network Volumeは残る)
$ ./scripts/runpod_terminate.sh <pod-id>
```

## ライセンス

- このディレクトリのコードが使う、第三者のコード・データ・フォントと、そのライセンス

| 対象 | ライセンス | 利用箇所 |
|---|---|---|
| [lucidrains/stylegan2-pytorch](https://github.com/lucidrains/stylegan2-pytorch) の、写像ネットワーク・生成器・判別器のコード(部分的にコピーし、改変したもの) | MIT License(全文は [licenses/stylegan2-pytorch.txt](licenses/stylegan2-pytorch.txt)) | `scripts/glyph_gan.py` |
| [DiffAugment](https://github.com/mit-han-lab/data-efficient-gans) の増強(位置ずれ・切り抜き)(コードは含まず、同じ動きになるよう書き直したもの) | BSD 2-Clause License | `scripts/train_glyph_gan.py` |
| [hash-prospector](https://github.com/skeeto/hash-prospector) の整数のハッシュ `lowbias32` (関数をそのままコピーしたもの) | The Unlicense | `scripts/shaders/glyph.frag` |
| [Google Fonts](https://github.com/google/fonts) の25書風(Kosugi・Kosugi Maru以外) | SIL Open Font License 1.1 | 学習データ(`scripts/download_fonts.py` で取得) |
| Google Fonts の2書風(Kosugi・Kosugi Maru) | Apache License 2.0 | 学習データ(同上) |

- フォントの各ファミリーのライセンスの全文は、 `scripts/download_fonts.py` がフォントと一緒に `data/fonts/` へ取得する(`<ファミリーのディレクトリ名>_OFL.txt` / `<ファミリーのディレクトリ名>_LICENSE.txt`)
