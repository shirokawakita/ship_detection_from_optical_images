# 他PCでの hold-out 学習の再現手順（同一データセット・同一手法）

**GitHub に push する前の確認**や、**ソース取得後にデータ取得〜hold-out で `best.pt` まで進める手順**は **[github-clone-kit/README.md](github-clone-kit/README.md)**（入口）と **[github-clone-kit/clone-and-implement.md](github-clone-kit/clone-and-implement.md)**（実装フロー中心）にあります。

このドキュメントは、**本リポジトリで用いている ShipRSImageNet（Roboflow 等）＋ Ultralytics YOLO OBB の hold-out 学習**を、**別の PC 上で同じ前提に近づけて**実行し、`best.pt` などの学習済みモデルを得るためのチェックリストです。

> **hold-out の意味（本プロジェクト）**  
> `data.yaml` の **`train`** で学習し、学習ループ中の検証は **`val`**（Roboflow エクスポートでは多くの場合 `valid/images`）に対して行います。**test** は学習に使わず、必要に応じて [`scripts/evaluate.py`](../scripts/evaluate.py) の `--split test` で最終評価します（K-fold は使いません）。

---

## Cursor で別PCから実装する（推奨ワークフロー）

Markdown だけではコマンドは走りません。**Cursor のチャットを Agent モードにし**、下記のように **このファイルを `@` で添付**してから依頼すると、エージェントがリポジトリ内のコマンドを実行しながら手順を進めやすくなります。

### 手順の概要

1. **別PCで** 本リポジトリを clone し、Cursor で **フォルダを開く**（開くのは必ず **リポジトリルート**＝`pyproject.toml` がある階層）。
2. チャットで **`@docs/holdout_training_on_another_pc.md`** を指定し、次の「コピペ用プロンプト」のいずれか（または順に）を送る。
3. **あなたが手で行うこと**: `.env` に API キーを書く、USB 等でデータフォルダをコピーする、Roboflow の Web でバージョン確認する、など（エージェントに秘密を貼らない運用を推奨）。

### コピペ用プロンプト（そのまま Agent に送れる）

**A. 環境だけ先に整える（clone 済み前提）**

```text
@docs/holdout_training_on_another_pc.md の「2. 新しい PC での初期セットアップ」に従ってください。
プロジェクトルートで uv sync（--extra dev 可）、続けて GPU 確認用の python -c コマンドを実行し、結果を要約してください。エラーが出たら原因を切り分けて修正案を出してください。
```

**B. Roboflow からデータを取る（`.env` はユーザーが作成済みとする）**

```text
@docs/holdout_training_on_another_pc.md に従い、Roboflow 経由でデータを取得してください。
scripts/download_data.py を実行し、出力された data.yaml のパスを確認したうえで、configs/train_config.yaml の data がそのパスと一致しているか確認してください。
続けて「4. 必須: data.yaml の path」に該当する場合は data.yaml を修正し、学習が通る状態にしてください。
```

**C. データは別PCからコピー済み（フォルダだけある）**

```text
@docs/holdout_training_on_another_pc.md の「3. 方法 B」と「4. data.yaml の path」に従ってください。
data/datasets/shiprsimagenet-39/data.yaml を開き、誤った path 行があれば削除または絶対パスに直してください。configs/train_config.yaml の data を確認してください。
```

**D. hold-out 学習を実行**

```text
@docs/holdout_training_on_another_pc.md の「6. 学習の実行」に従い、configs/train_config.yaml で uv run python scripts/train.py を実行してください。
終了後、runs/detect/<name>/weights/best.pt のパスを明示してください。
```

**E. 評価まで（任意）**

```text
@docs/holdout_training_on_another_pc.md の「7. 任意: 評価」に従い、学習で得た best.pt で evaluate.py を --split val と --split test の両方実行してください。出力ディレクトリを教えてください。
```

### Cursor ルール（任意）

Agent が方針を外しにくくするため、[`.cursor/rules/holdout-remote-setup.mdc`](../.cursor/rules/holdout-remote-setup.mdc) をチャットに **`@holdout-remote-setup`** で添付してから上記プロンプトを送っても構いません（`alwaysApply: false` のため、**自動では読み込まれません**）。

### 注意（Cursor / エージェントの限界）

- **ターミナルのカレントディレクトリ**はリポジトリルートにしてください（`cd` 先がずれると相対パスが壊れます）。
- **API キー**はチャットに貼らず、`.env` をエディタで編集する運用を推奨します。
- **大容量データ**の USB コピーは人間が行い、コピー後にエージェントに「データを置いたので検証して」と依頼する形が確実です。
- 長時間学習は **バックグラウンド実行**や **別ターミナル**を検討してください（Cursor のタイムアウト設定に依存します）。

---

## 1. 前提のそろえ方（再現性）

| 項目 | 推奨 |
|------|------|
| ソースコード | **同一の Git コミット**（または同一タグ）を checkout |
| Python 依存 | リポジトリの **`uv.lock`** を残したまま **`uv sync`**（`--extra dev` は任意） |
| データセット | **同一のバージョン**（例: Roboflow `ROBOFLOW_VERSION=39`）または、既存 PC の **`data/datasets/shiprsimagenet-39/` を丸ごとコピー** |
| 事前学習重み | [`configs/train_config.yaml`](../configs/train_config.yaml) の `model: yolo11s-obb.pt` と一致させる（別名にした場合は YAML も変更） |
| 乱数 | 完全同一の再現には Ultralytics 側の `seed` 固定や同一 GPU/CUDA 版が影響します。**実務上は** `uv.lock` とデータ・YAML を揃えることを優先してください。 |

---

## 2. 新しい PC での初期セットアップ

リポジトリルートで実行します（Windows の例は PowerShell 想定）。

### 2.1 リポジトリと仮想環境

```bash
git clone <このリポジトリの URL> ship_detection_optical_images
cd ship_detection_optical_images
uv sync --extra dev
```

### 2.2 GPU（CUDA）の確認

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`torch.cuda.is_available()` が `False` のときは、`pyproject.toml` の **`[tool.uv.sources]`**（CUDA 12.4 用インデックス）が効いた `uv sync` になっているか、ドライバを確認してください。CPU のみでも学習は可能ですが時間がかかります。

### 2.3 環境変数（データを Roboflow から取る場合）

```bash
cp .env.example .env
```

`.env` に **`ROBOFLOW_API_KEY`** を設定します。ShipRS の例は [`.env.example`](../.env.example) のとおりです（`ROBOFLOW_WORKSPACE` / `ROBOFLOW_PROJECT` / `ROBOFLOW_VERSION` など）。

---

## 3. データセットの配置（いずれか一方）

### 方法 A: この PC で `download_data.py` を使う（推奨・キーが必要）

```bash
uv run python scripts/download_data.py
```

完了後、表示される `data.yaml` の場所を確認し、[ `configs/train_config.yaml`](../configs/train_config.yaml) の **`data:`** が、**その `data.yaml` を指す相対パス**になっているか確認します（既定例: `data/datasets/shiprsimagenet-39/data.yaml`）。

### 方法 B: 既に学習に使った PCからフォルダをコピーする（キー不要・オフライン可）

次のディレクトリ構造を維持したまま、**丸ごと**新 PCの同じ相対パスに置きます。

```text
data/datasets/shiprsimagenet-39/
  data.yaml
  train/images/   train/labels/
  valid/images/   valid/labels/
  test/images/    test/labels/
```

そのうえで [`configs/train_config.yaml`](../configs/train_config.yaml) の `data:` を上記 `data.yaml` に合わせます。

---

## 4. 必須: `data.yaml` の `path`（トラブル回避）

Roboflow が出力する `data.yaml` に、誤った

```yaml
path: ../datasets/roboflow
```

のような **`path` 行が含まれている**場合、Ultralytics は画像パスを誤解決し、学習が失敗することがあります。

**推奨いずれか:**

1. **`path` キー行を削除する**（`train` / `val` / `test` / `names` だけにする）  
   → Ultralytics は `data.yaml` があるディレクトリをデータルートとして解決します。
2. または **`path` をデータセットの絶対パス**に明示する。

`train` / `val` / `test` の値は、従来どおり `train/images` や `valid/images` 等で構いません（Roboflow の `valid` は YAML 上 `val: valid/images` 表記のことが多いです）。

---

## 5. 学習設定の確認（[`configs/train_config.yaml`](../configs/train_config.yaml)）

他PCで変えたいことが多い項目:

| キー | 内容 |
|------|------|
| `data` | 上記 `data.yaml` へのパス（プロジェクトルートからの相対で可） |
| `model` | 例: `yolo11s-obb.pt`（ルートに無い場合は Ultralytics の解決規則に従い取得されます） |
| `epochs` | 本番学習では十分な値に（疎通なら小さく） |
| `batch` | GPU メモリに合わせて調整 |
| `device` | 例: `"0"`（CPU なら `"cpu"`） |
| `workers` | CPU 数に合わせて減らすと安定することがあります |
| `project` / `name` | 出力先 `runs/detect/<name>/` を識別しやすい名前に |

**hold-out の分割自体**は `data.yaml` の `train` / `val` / `test` に依存します。ここを変えなければ、**同一データ・同一分割**で学習できます。

---

## 6. 学習の実行

リポジトリルートで:

```bash
uv run python scripts/train.py --config configs/train_config.yaml
```

成功すると、学習済みモデルは通常次に出力されます。

```text
runs/detect/<train_config.yaml の name>/weights/best.pt
runs/detect/<name>/weights/last.pt
```

既定の `name` が `yolo11s-obb-shiprs-v39` の場合の例:

```text
runs/detect/yolo11s-obb-shiprs-v39/weights/best.pt
```

`runs/` は `.gitignore` 対象のことが多く、**Git には含まれません**。他PCへ持ち運ぶ場合は、この `weights` フォルダや `best.pt` を別途コピーしてください。

---

## 7. 任意: 評価（valid / test）

同一の `data.yaml` を指定し、学習済み `best.pt` で評価します。

```bash
# 検証分割（val）— 学習中の指標と同系
uv run python scripts/evaluate.py --weights runs/detect/yolo11s-obb-shiprs-v39/weights/best.pt --data data/datasets/shiprsimagenet-39/data.yaml --split val

# テスト分割（test）— ホールドアウトの最終確認用
uv run python scripts/evaluate.py --weights runs/detect/yolo11s-obb-shiprs-v39/weights/best.pt --data data/datasets/shiprsimagenet-39/data.yaml --split test
```

`--weights` / `--data` は、実際のパスに読み替えてください。レポートは `runs/eval/` 以下に出力されます。

---

## 8. 任意: loss・精度の推移図

```bash
uv run python scripts/plot_training_curves.py --results runs/detect/<name>/results.csv --output docs/images/my_training_curves.png
```

---

## 9. よくある問題

| 現象 | 確認すること |
|------|----------------|
| `images not found` / パスエラー | **`data.yaml` の `path`**（[セクション 4](#4-必須-datayaml-のパストラブル回避)） |
| CUDA OOM | `batch` を下げる、`imgsz` を下げる |
| Roboflow 401 | `.env` の API キー、プロジェクト／バージョン番号 |
| 依存の食い違い | **`uv.lock` をコミット済みのブランチで `uv sync`** |

---

## 10. 参考（リポジトリ内）

- GitHub 公開・clone キット: [`github-clone-kit/README.md`](github-clone-kit/README.md)
- メイン README: [`README.md`](../README.md)
- Cursor Agent 向けルール（任意・`@` 添付用）: [`.cursor/rules/holdout-remote-setup.mdc`](../.cursor/rules/holdout-remote-setup.mdc)
- 学習エントリ: [`scripts/train.py`](../scripts/train.py)、[`src/ship_detector/train/trainer.py`](../src/ship_detector/train/trainer.py)
- 評価エントリ: [`scripts/evaluate.py`](../scripts/evaluate.py)

K-fold 用の [`scripts/train_kfold.py`](../scripts/train_kfold.py) は **hold-out 再現には不要**です。通常の hold-out のみ行う場合は **`scripts/train.py`** だけで足ります。
