# 他PCで `git clone` して実装・学習まで進める

## 1. 前提

- **Git** と **uv** がその PCに入っていること（[uv 公式](https://docs.astral.sh/uv/getting/installation/)）。
- リポジトリは **GitHub 等に push 済み**で、手元には **秘密が含まれていない**状態であること（`.env` は各自が作る）。

## 2. clone と依存関係

```bash
git clone https://github.com/<YOUR_USER_OR_ORG>/<REPO_NAME>.git
cd <REPO_NAME>
uv sync --extra dev
```

GPU 確認（CUDA 利用時）:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## 3. データと環境変数

GitHub には **`data/` が含まれない**ことが多いです（`.gitignore` 参照）。次のいずれかで用意します。

- **Roboflow**: `.env.example` を `.env` にコピーし API キーを設定 → `uv run python scripts/download_data.py`
- **別PCからコピー**: 既存環境の `data/datasets/...` を同じ相対パスに置く

詳細・`data.yaml` の **`path` 修正**・hold-out 学習は次を正としてください。

→ **[../holdout_training_on_another_pc.md](../holdout_training_on_another_pc.md)**

## 4. 学習（hold-out）

```bash
uv run python scripts/train.py --config configs/train_config.yaml
```

重みの出力先の例: `runs/detect/<configs の name>/weights/best.pt`（`runs/` も通常は Git 対象外のため、**成果物は各自の PC にのみ**存在します）。

## 5. Cursor で進める場合

[../holdout_training_on_another_pc.md](../holdout_training_on_another_pc.md) の **「Cursor で別PCから実装する」** に、Agent 用の `@` 添付とコピペプロンプトがあります。任意で [`.cursor/rules/holdout-remote-setup.mdc`](../../.cursor/rules/holdout-remote-setup.mdc) をチャットに添付してください。

## 6. このキットの入口

一覧とリンクは [README.md](README.md) を参照してください。
