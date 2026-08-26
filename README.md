# Step-by-step Gaussian Splatting Tutorial

動画または多視点画像から、Gaussian Splattingを用いた3D再構築をJupyter NotebookまたはCLIで実行するチュートリアルです。

## Setup

```bash
git clone https://github.com/sh1027/gs-tutorial.git
cd gs-tutorial
```

OS側の依存パッケージをインストールします。Debian/Ubuntu環境向けのスクリプトを用意しています。

```bash
bash scripts/install-system-deps.sh
```

既定ではシステムの`python3`を使い、リポジトリ内の`.venv`へ依存パッケージをインストールします。Python 3.10から3.12までをサポートしています。

```bash
bash scripts/setup.sh
source scripts/activate.sh
gs-tutorial env
```

使用するPythonを明示する場合は、両方のスクリプトへ同じ`PYTHON_VERSION`を指定します。

```bash
PYTHON_VERSION=3.10 bash scripts/install-system-deps.sh
PYTHON_VERSION=3.10 bash scripts/setup.sh
```

`scripts/setup.sh`はJupyterカーネルも次の名前で自動登録します。

- カーネル名: `gs-tutorial`
- 表示名: `Python (gs-tutorial)`

カーネルだけを手動で再登録する場合は、次を実行します。

```bash
./.venv/bin/python -m pip install ipykernel
./.venv/bin/python -m ipykernel install \
    --user \
    --name gs-tutorial \
    --display-name "Python (gs-tutorial)"
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
各Notebookのカーネルに`Python (gs-tutorial)`を選択してください。

1. [01_preprocess.ipynb](notebooks/01_preprocess.ipynb): 入力画像の抽出と選別
2. [02_colmap.ipynb](notebooks/02_colmap.ipynb): 特徴抽出、マッチング、カメラ姿勢推定
3. [03_gaussian.ipynb](notebooks/03_gaussian.ipynb): Gaussianの学習、評価、表示

### Viewerを開く

学習完了後、別のターミナルでviewerを起動します。このターミナルはviewerを表示している間、そのまま起動しておきます。

```bash
cd gs-tutorial
source scripts/activate.sh
gs-tutorial viewer \
    outputs/my_video/training/point_cloud/final.ply \
    --config configs/video.yaml \
    --port 8000
```

viewerをリモートサーバーで起動した場合は、手元のPCからSSHでport 8000をforwardします。`<username>`と`<remote-host>`は利用環境に合わせて置き換えてください。

```bash
ssh -L 8000:localhost:8000 <username>@<remote-host>
```

viewerと同じマシンから見る場合はSSH forwardingは不要です。viewerの起動後、ブラウザで[http://localhost:8000](http://localhost:8000)を開きます。

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
