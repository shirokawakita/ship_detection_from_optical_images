# GitHub 公開・clone 用キット

このディレクトリは、**ソースや実装を GitHub に載せ、他PCで `git clone` して同じ環境を再現する**ときの手順を **一箇所に集約**したものです。コード本体や設定はリポジトリルートのままです（ここにコピーはしません）。

## 含まれるファイル

| ファイル | 内容 |
|----------|------|
| [push-to-github.md](push-to-github.md) | リモート作成前の確認、`.gitignore` の意味、初回 `push` の例 |
| [clone-and-implement.md](clone-and-implement.md) | `clone` 後の `uv sync`、データ、学習・Cursor までの導線 |

## すぐ読む順番

1. **初めて GitHub に上げる側** → [push-to-github.md](push-to-github.md)  
2. **別PCで clone して実装する側** → [clone-and-implement.md](clone-and-implement.md)  
3. **hold-out 学習の詳細**（データ YAML の `path` 修正など）→ [../holdout_training_on_another_pc.md](../holdout_training_on_another_pc.md)

## リポジトリの正

- **本番のコード・設定**: ワークスペース**ルート**（`pyproject.toml`、`src/`、`scripts/`、`configs/`）。
- この `docs/github-clone-kit/` は **手順書のみ**（実行コードは含みません）。
