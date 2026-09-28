# 学習側

- 幽霊文字の生成モデルの学習コード

## 構成

- `glyph/` : スクリプトが共通に使う部品
- `scripts/` : 実行するスクリプト
- `scripts/runpod/` : RunPodで動かすためのスクリプト
- `data/` : フォント・学習データ・チェックポイント・ONNX(gitの管理外)

## セットアップ

```sh
# 依存パッケージのインストール
$ uv sync

# 学習データ(フォントで描いた漢字の画像)に使う27書風のフォント(Google Fonts、OFL / Apache License 2.0)の取得
# data/fonts/配下に、フォントと各ファミリーのライセンスのファイルが置かれる。取得元のコミットを固定し、
# 各フォントをgitのblobの識別子で照合する(照合済みのファイルがある場合は再ダウンロードをスキップする)
$ uv run scripts/download_fonts.py

# RunPodへのデプロイスクリプト(scripts/runpod/*.sh)が、pod作成時のJSON出力から
# id・IP・ポートを取り出すために使う
$ brew install jq
```

## 実行手順

```sh
# 27書風のフォントで、各フォントが持っている漢字をすべて1字ずつ描き、data/glyphs_<解像度>.npzへ保存する
# (download_fonts.pyの取得が前提)。字の数はフォントによって違う(標準的なフォントで約6,700字、多いもので
# 約14,200字)。入れるのはCJK統合漢字(基本の範囲と拡張A〜J)と、CJK互換漢字のうち統合漢字と重複しない字・
# 統合漢字と違う形(旧字形)に描かれる字。部首や筆画の記号、かな、記号、IVSの字形違いは入れない(詳しくは
# スクリプトの冒頭のコメント)。字はフォント本来の大きさで描き、インクの外接の枠の中心を画像の中心に合わせる。
# 画像はインクの濃さ(0=紙〜255=インク)で、黒地に白の字として見える向き。中身が空の字と、そのフォントの
# 豆腐(.notdef)と同じ形になる字は飛ばす
$ uv run scripts/build_dataset.py

# 解像度を変える(省略時は64)
$ uv run scripts/build_dataset.py --resolution 128

# 学習データ(data/glyphs_64.npz)でGAN(写像ネットワーク・生成器・判別器、glyph/gan.py)を学習し、
# data/checkpoints/glyph_gan/<名前>/checkpoint_<歩数>.ptへ保存する。チェックポイントには、推論に使う移動平均の版の重みと、
# 再開に使う最適化の状態も入る(最新の3つと1万歩ごとのものを残す)。保存ごとに、推論に使う移動平均の版の重みだけを
# snapshot_<歩数>.pt としてすべて残す(学習の途中のどの時点も、あとで評価し直せる)。既定の設定は、同じ字ばかり出る崩壊と長い学習での崩れを避けるために確かめた構成
# (ロジスティック損失、判別器の学習率=生成器、写像ネットワークの学習率の倍率0.01、強い勾配の罰則、左右反転なしの増強)。
# 学習中は、スタイルの散らばりと崩れ(外周の枠・白黒の反転・塗りつぶし)の割合を定期的に表示する。
# GPU(CUDA)、Apple Silicon(MPS)、CPUの順に使えるものを使う
$ uv run scripts/train_gan.py --name <名前>

# 学習の長さ(千枚単位、省略時は1920 = バッチ64で3万歩)と乱数の種を指定する
$ uv run scripts/train_gan.py --name <名前> --kimg 960 --seed 1

# 途中で止まった学習を、同じ名前の最新のチェックポイントから続ける(上書きした設定も、保存した値に戻る)。
# 学習の量を増やして同じように続ければ、学習を延ばせる。すでにある名前で --resume なしに始めると、混ざらないよう止まる
$ uv run scripts/train_gan.py --name <名前> --resume

# 学習の設定(スクリプトの冒頭の定数)を、この学習だけ変える(スイープ用。何度でも指定できる。使った値はチェックポイントに残る)
$ uv run scripts/train_gan.py --name <名前> --override LEARNING_RATE=1e-4 --override CAPACITY=16

# 保存ごとに、スナップショットを文字認識のモデルで評価のスクリプトと同じ方法で評価し、ログに1行出して evaluation.csv に足す
$ uv run scripts/train_gan.py --name <名前> --classifier data/glyph_classifier.pt

# 別の解像度の学習データで学習する(省略時はdata/glyphs_64.npz)
$ uv run scripts/train_gan.py --name <名前> --data data/glyphs_128.npz

# 生成した字の評価の物差しにする文字認識のモデルを学習し、data/glyph_classifier.ptへ保存する。
# 3書風を学習から外し、見たことのない書風でも読めるかを確かめる。学習のあと、実在字で特徴の距離の分布を測り、
# 評価に使うしきい値(同じ種類の字とみなす距離、別の字への急な切り替わりとみなす距離)も一緒に保存する
$ uv run scripts/train_classifier.py

# 学習データと保存先を指定する
$ uv run scripts/train_classifier.py --data data/glyphs_128.npz --output data/glyph_classifier_128.pt

# 学習したチェックポイント(推論に使う移動平均の版の重み)を評価する。新しさ(知らない字になっているか)、
# 2字の種類の数(同じ字ばかり出ていないか)、崩れの割合、精度・再現率・密度・網羅率(実在字の分布との重なり。
# 実在字どうしの値を並べる)、写しの割合(学習データの丸写しに近い字の割合)、なめらかさ(表示と同じsimplex noiseの軌跡での、
# 急な切り替わりの割合)を表示し、生成した字の一覧の画像をチェックポイントの隣に保存する。
# 関門(崩れ・崩壊・新しさ・写し・なめらかさ)を満たしたかも表示する。学習データの全画像をGPUに載せる(64pxで約4GB)
$ uv run scripts/evaluate_gan.py --checkpoint data/checkpoints/glyph_gan/<名前>/checkpoint_<歩数>.pt

# 学習の途中の複数の時点を、同じ実在字・同じsimplex noiseの位置で比べ、値をCSVに書き出す。
# 関門をすべて満たすもののうち、網羅率が最も高いものを示す
$ uv run scripts/evaluate_gan.py --checkpoint data/checkpoints/glyph_gan/<名前>/snapshot_*.pt --csv <CSVのパス>

# 字の数(省略時は生成物・実在字それぞれ1万字)と軌跡の数(省略時は64本)を減らして、手早く確かめる
$ uv run scripts/evaluate_gan.py --checkpoint <チェックポイント> --samples 1000 --walks 4

# 文字認識のモデル・実在字のデータ・乱数の種を指定する
$ uv run scripts/evaluate_gan.py --checkpoint <チェックポイント> --classifier <文字認識のモデルの重み> --data <学習データ> --seed 1

# 学習中の保存ごとの評価の値(evaluation.csv)を、複数の学習ぶん並べて、歩数ごとの推移を1枚の画像に描く
# (関門を満たさなかった時点は白抜きの点)。学習の名前か、evaluation.csv のパスを渡す
$ uv run scripts/plot_evaluations.py <名前1> <名前2> --output <画像のパス>

# 学習したチェックポイントの生成器を、data/onnx/glyph_generator.onnxへ書き出す(推論には移動平均の版の重みを使う)。
# 入力はsimplex noiseの値で、正規分布への変換 → 写像ネットワーク → 生成器 → インクの画像。
# 書き出したあと、同じ入力でPyTorchとONNX Runtimeの出力を比べる。書き出した生成器の表示・録画は [表示側](../display/README.md) で行う
$ uv run scripts/export_onnx.py --checkpoint data/checkpoints/glyph_gan/<名前>/checkpoint_<歩数>.pt

# 書き出し先を指定する
$ uv run scripts/export_onnx.py --checkpoint <チェックポイント> --output <ONNXのパス>
```

## RunPodへのデプロイ

- 外部GPU( [RunPod](https://www.runpod.io/) )でGANの学習(scripts/train_gan.py)や、文字認識のモデルの学習・評価などを実行するためのスクリプト
- 初回のみ、RunPodアカウントの作成、 `runpodctl` のセットアップ(APIキー・SSH鍵)が別途必要(Network Volumeを使う場合はその作成も)
- podの作成は課金が発生するので1つずつ行い、作れなかったときは `runpodctl pod list` で作られていないことを確かめてから、条件を変えて作り直す

```sh
# pod作成。SSH接続の確認と、ホストのCPUの速さの簡易ベンチマーク(学習の速さはCPU側で決まりやすく、同じGPUの
# 機種でもホストによって違う)まで行い、pod_id/ip/portを表示する。Network Volumeを使わない場合は、pod自身の
# ディスク(podを消すと中身も消える)を使い、空きのあるデータセンターをRunPodに選ばせる
$ ./scripts/runpod/create_pod.sh [--volume <network-volume-id>] [--gpu <gpu-id>]... [--name <pod-name>]
$ ./scripts/runpod/create_pod.sh
# GPU機種を指定する(省略時はRTX 4090)。複数指定すると、空きがないときに次の機種を順に試し、1つ作れたら止める
# (機種の名前は runpodctl gpu list で確かめる)
$ ./scripts/runpod/create_pod.sh --gpu "NVIDIA GeForce RTX 4090" --gpu "NVIDIA RTX PRO 6000 Blackwell Server Edition" --gpu "NVIDIA L40S"
# Network Volumeを/workspaceに付ける(そのVolumeのデータセンターで作るので、GPUの空きがないと作れないことがある)
$ ./scripts/runpod/create_pod.sh --volume <network-volume-id>

# コード(scripts・pyproject.toml・uv.lock)を転送しuv syncして、CUDAが使えるかを確かめる
$ ./scripts/runpod/deploy_code.sh <ip> <port> [--data <file>] [--classifier <file>] [--source <local-dir>] [--remote-dir <path>] [--link-venv <remote-dir>]
$ ./scripts/runpod/deploy_code.sh <ip> <port>
# 学習データ(training/data/配下のファイル)も送る(初回のみ)
$ ./scripts/runpod/deploy_code.sh <ip> <port> --data glyphs_64.npz
# 保存ごとの評価に使う文字認識のモデルの重み(training/data/配下のファイル)も送る
$ ./scripts/runpod/deploy_code.sh <ip> <port> --data glyphs_64.npz --classifier <文字認識のモデルの重み>
# 本番のtraining/scripts以外(アブレーション用のスクラッチコピーなど)を送る
$ ./scripts/runpod/deploy_code.sh <ip> <port> --source <local-dir>
# 同じpodに、別の配置先として並べて置く(配置先ごとにdata/を持つため--dataも必要)。
# 既に同期済みの配置先の.venvをシンボリックリンクすれば、依存関係が同じなら再ダウンロードなしで済む
$ ./scripts/runpod/deploy_code.sh <ip> <port> --data glyphs_64.npz --remote-dir /workspace/ghosts/training_b --link-venv /workspace/ghosts/training

# 学習をバックグラウンドで始める(SSHを切っても止まらない)。ログは学習の名前ごとにtrain_<名前>.logへ書くので、
# 1つのpodで複数の学習を同時に回せる(中身は runpod/launch_job.sh で、ジョブの名前を train_<名前> にして起動する)
$ ./scripts/runpod/launch_training.sh <ip> <port> --name <名前> [--resume] [--remote-dir <path>] [-- <train_gan.pyの引数>...]
$ ./scripts/runpod/launch_training.sh <ip> <port> --name <名前>
# train_gan.pyに渡す追加の引数を、-- のあとにそのまま並べる(空白を含む値も、引用符で囲めばそのまま渡る)
$ ./scripts/runpod/launch_training.sh <ip> <port> --name <名前> -- --kimg 960 --seed 1 --classifier data/glyph_classifier.pt
# 止まった学習を、最新のチェックポイントから続ける(ログには追記する)。学習の量を増やして続ければ、学習を延ばせる
$ ./scripts/runpod/launch_training.sh <ip> <port> --name <名前> --resume
$ ./scripts/runpod/launch_training.sh <ip> <port> --name <名前> --resume -- --classifier data/glyph_classifier.pt --kimg 2560

# 学習以外のスクリプト(文字認識のモデルの学習・評価など)を、ジョブの名前をつけてバックグラウンドで始める。
# ログは<ジョブの名前>.log、プロセスの番号は<ジョブの名前>.pidへ書き、終わるとログの最後に「exit code <n>」を足す
$ ./scripts/runpod/launch_job.sh <ip> <port> --job <ジョブの名前> [--append] [--remote-dir <path>] -- <スクリプト> [引数...]
$ ./scripts/runpod/launch_job.sh <ip> <port> --job train_classifier -- scripts/train_classifier.py
$ ./scripts/runpod/launch_job.sh <ip> <port> --job evaluate -- scripts/evaluate_gan.py --checkpoint <チェックポイント> --csv <CSVのパス>

# 学習が動いているか(終わっていれば終了コード)・ログの直近n行・最新の指標(スタイルの散らばり、崩れの割合など)・
# 最新の評価(--classifier を指定したとき)を確認する
$ ./scripts/runpod/check_progress.sh <ip> <port> (--name <名前> | --job <ジョブの名前>) [--lines <n>] [--follow [--interval <秒>]] [--remote-dir <path>]
$ ./scripts/runpod/check_progress.sh <ip> <port> --name <名前>
# 表示する行数を指定する
$ ./scripts/runpod/check_progress.sh <ip> <port> --name <名前> --lines 50
# 学習以外のジョブ(runpod/launch_job.sh で始めたもの)を確認する
$ ./scripts/runpod/check_progress.sh <ip> <port> --job <ジョブの名前>
# 終わるまで新しい行を流し続け、ジョブの終了コードで終わる(学習は評価・終わり・失敗の行だけ、ほかのジョブはすべての行)
$ ./scripts/runpod/check_progress.sh <ip> <port> --name <名前> --follow

# 学習の結果(チェックポイント・スナップショット・評価の値のディレクトリとログ)を、ローカルのdata/checkpoints/glyph_gan/<名前>/へ落とす。
# ファイルの数と中身(SHA-256)をpodの上と比べ、すべて一致したときだけpodの上のチェックポイントとスナップショットを消す。
# podを消す前に必ず実行する(Network Volumeを使わないpodでは、podを消すと中身も消えるため)
$ ./scripts/runpod/download_results.sh <ip> <port> --name <名前> [--keep-remote] [--remote-dir <path>]
$ ./scripts/runpod/download_results.sh <ip> <port> --name <名前>
# podの上のファイルを消さずに落とす(学習を延ばす前など)
$ ./scripts/runpod/download_results.sh <ip> <port> --name <名前> --keep-remote

# podを削除して課金を止める(Network Volumeは残る)
$ ./scripts/runpod/terminate.sh <pod-id>
```

## ライセンス

- このディレクトリのコードが使う、第三者のコード・データ・フォントと、そのライセンス

|対象|ライセンス|利用箇所|
|---|---|---|
|[lucidrains/stylegan2-pytorch](https://github.com/lucidrains/stylegan2-pytorch) の、写像ネットワーク・生成器・判別器のコード(部分的にコピーし、改変したもの)|MIT License(全文は [licenses/stylegan2-pytorch.txt](licenses/stylegan2-pytorch.txt))|`glyph/gan.py`|
|[DiffAugment](https://github.com/mit-han-lab/data-efficient-gans) の増強(位置ずれ・切り抜き)(コードは含まず、同じ動きになるよう書き直したもの)|BSD 2-Clause License|`scripts/train_gan.py`|
|[Google Fonts](https://github.com/google/fonts) の25書風(Kosugi・Kosugi Maru以外)|SIL Open Font License 1.1|学習データ( `scripts/download_fonts.py` で取得)|
|Google Fonts の2書風(Kosugi・Kosugi Maru)|Apache License 2.0|学習データ(同上)|

- フォントの各ファミリーのライセンスの全文は、 `scripts/download_fonts.py` がフォントと一緒に `data/fonts/` へ取得する( `<ファミリーのディレクトリ名>_OFL.txt` / `<ファミリーのディレクトリ名>_LICENSE.txt` )
