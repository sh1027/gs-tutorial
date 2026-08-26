# Step-by-step Gaussian Splatting Tutorial

動画または多視点画像から、Gaussian Splattingを用いた3D再構築をJupyter NotebookまたはCLIで実行するチュートリアルです。

## Setup

```bash
git clone https://github.com/sh1027/gs-tutorial.git
cd gs-tutorial
```

既定ではリポジトリ内の`.venv`へ依存パッケージをインストールします。

```bash
bash scripts/setup.sh
source scripts/activate.sh
gs-tutorial env
```

既存環境の確認だけを行う場合は、次を実行します。

```bash
bash scripts/setup.sh --check
```

動画入力には`ffmpeg`が必要です。標準ではPyCOLMAPを使用します。COLMAP CLIを使う場合は、OS側にCOLMAPをインストールしてください。

## Configurations

動画では`configs/video.yaml`、多視点画像では`configs/images.yaml`を使用します。
実行前に設定ファイルの`input.source`と`scene_name`を変更してください。

## Run in Jupyter Notebook

次の順番で実行します。
デフォルトでは、`configs/video.yaml`を使用しています。多視点画像の場合は、該当箇所を`configs/images.yaml`に変更してください。

1. [01_preprocess.ipynb](notebooks/01_preprocess.ipynb): 入力画像の抽出と選別
2. [02_colmap.ipynb](notebooks/02_colmap.ipynb): 特徴抽出、マッチング、カメラ姿勢推定
3. [03_gaussian.ipynb](notebooks/03_gaussian.ipynb): Gaussianの学習、評価、表示

```bash
jupyter lab
```

## Run in CLI

次の順番で実行します。
デフォルトでは、`configs/video.yaml`を使用しています。多視点画像の場合は、該当箇所を`configs/images.yaml`に変更してください。

```bash
gs-tutorial prepare --config configs/video.yaml
gs-tutorial reconstruct --config configs/video.yaml
gs-tutorial inspect --config configs/video.yaml
gs-tutorial train --config configs/video.yaml
gs-tutorial render-views --config configs/video.yaml
gs-tutorial viewer outputs/my_video/training/point_cloud/final.ply --config configs/video.yaml
```

COLMAP CLIを使用する場合は、再構築時にbackendを指定します。

```bash
gs-tutorial reconstruct --config configs/video.yaml --backend cli
```

処理結果は`outputs/<scene_name>/`へ保存されます。既存出力は自動で上書きされません。
前処理、学習、評価をやり直す場合のみ、対応するコマンドに`--overwrite`を指定してください。
