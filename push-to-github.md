# GitHub にコード・実装を push する前に

## 1. このリポジトリに「載る」もの（想定）

次は **通常 Git にコミットする**対象です。

- `src/`、`scripts/`、`configs/`、`tests/`、`docs/`（本キット含む）
- `pyproject.toml`、`uv.lock`（**再現性の要**。必ずコミット推奨）
- `.env.example`（**秘密を書かないテンプレート**）
- `.gitignore`、`.cursor/rules/`、`README.md` など

## 2. `.gitignore` により「載らない」もの（clone 側で用意）

ルートの [`.gitignore`](../../.gitignore) により、次は **リポジトリに含まれません**。GitHub 上には現れず、**各PCで別途用意**します。

| 対象 | 理由 | clone 後の対応 |
|------|------|------------------|
| `.env` | API キー等の秘密 | `.env.example` をコピーして `.env` を作成し、キーを設定 |
| `data/` | データセットが大容量 | Roboflow の `download_data.py` または別PCからフォルダコピー |
| `runs/` | 学習ログ・中間生成物 | 学習実行で再生成 |
| `.venv/` | 仮想環境 | `uv sync` で再生成 |
| `*.pt` 等 | 重みファイルが大きい | 初回学習時に Ultralytics が取得する場合や、手動配置 |

> **注意:** `*.pt` が ignore されているため、**カスタムで学習した `best.pt` を GitHub に載せたい**場合は、**Git LFS** の利用や `.gitignore` の見直し（チーム方針）が必要です。既定のままでは学習済み重みは push されません。

## 3. push 前チェックリスト

- [ ] `git status` で **意図しない秘密ファイル**（`.env`、キー付き JSON、個人パス）が含まれていない  
- [ ] `uv lock` を実行済みで **`uv.lock` が最新**（依存変更時）  
- [ ] `uv run pytest` が通る（可能な範囲）  
- [ ] `README.md` の手順が現在の `configs` と矛盾していない  
- [ ] 大きなバイナリを誤って add していない（`git check-ignore -v <file>` で確認可）

## 4. GitHub 上でリポジトリを作成したあと（例）

リポジトリ URL を自分のものに置き換えてください。

```bash
# 未初期化の場合のみ
git init
git add .
git status   # 確認してから
git commit -m "chore: initial import ship-detector pipeline"

git branch -M main
git remote add origin https://github.com/<YOUR_USER_OR_ORG>/<REPO_NAME>.git
git push -u origin main
```

既に `origin` がある場合は `git remote set-url origin ...` で更新します。

## 5. 参考

- メイン README: [../../README.md](../../README.md)
- clone 後の手順: [clone-and-implement.md](clone-and-implement.md)
