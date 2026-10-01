#!/bin/bash
cd "$(dirname "$0")"

# 1. 仮想環境を作成 (ここでは 'myenv' という名前)
python3 -m venv myenv

# 2. 仮想環境を有効化 (左端に (myenv) と表示されます)
source myenv/bin/activate

# 3. 依存ライブラリと pyworld をインストール
pip install --upgrade pip setuptools wheel
pip install cython numpy
pip install pyworld

# すでに setuptools の最新バージョン（84.0.0）がインストールされていますね。それにもかかわらず pkg_resources が見つからない原因は、setuptools バージョン 84.0.0 以降で pkg_resources モジュールが完全に削除（廃止）されたためです。pyworld というライブラリの内部コードが古い仕様（pkg_resources）に依存しているため、最新の setuptools と互換性がなくなってしまっています。pkg_resources がまだ含まれていた少し古いバージョンの setuptools にダウングレードすることで解決します。ターミナルで以下のコマンドを実行してください。
python -m pip install --upgrade pip setuptools
pip install "setuptools<70.0.0"

sudo apt-get update
sudo apt-get install libsndfile1
# 2. 仮想環境に soundfile パッケージをインストール
pip install soundfile
