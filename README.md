# ylab-curling

カーリング AI ゲームクライアント/サーバープロジェクト。MCTS アルゴリズムを使用して戦略的なカーリングの手を決定する AI システムです。

## プロジェクト概要

このプロジェクトは、カーリングゲームシミュレーションのためのクライアント・サーバーシステムです：

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone --recursive https://github.com/your-username/ylab-curling.git
cd ylab-curling
```

### 2. 依存関係のインストール

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install cmake build-essential libboost-all-dev qt6-base-dev qt6-svg-dev
```

### 3. プロジェクトのビルド

#### Linux/macOS

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## ゲームの実行方法

### サーバーの起動

1. サーバー設定ファイルの確認:

```bash
# デフォルト設定を確認
cat digitalcurling3_server/bin/config.json
```

2. サーバーの起動:

```bash
cd digitalcurling3_server/bin
./digitalcurling3_server.exe
```

サーバーは以下のポートでクライアント接続を待機します：

- **Team 0**: ポート 10000
- **Team 1**: ポート 10001

### クライアント（AI プレイヤー）の起動

#### 自分でビルドしたクライアントを使用

```bash
# Debugビルド
cd build/Debug
./ylab_client.exe localhost 10000

# Releaseビルド
cd build/Release
./ylab_client.exe localhost 10000
```

#### 引数の説明

- `localhost`: サーバーのホスト名または IP アドレス
- `10000`: 接続ポート（Team 0 の場合 10000、Team 1 の場合 10001）

### GUI での観戦

ゲームの進行を GUI で観戦できます：

```bash
cd digitalcurling3_gui/bin
./digitalcurling3_gui.exe
```

## 対戦の開始手順

### 1. サーバーの起動

```bash
cd digitalcurling3_server/bin
./digitalcurling3_server.exe
```

### 2. 両チームのクライアント接続

**Team 0 (先攻)**:

```bash
cd build/Release
./ylab_client.exe localhost 10000
```

**Team 1 (後攻)**:

```bash
cd build/Release
./ylab_client.exe localhost 10001
```

### 3. GUI 観戦（オプション）

```bash
cd digitalcurling3_gui/bin
./digitalcurling3_gui.exe
```

両方のクライアントが接続されると、自動的にゲームが開始されます。
