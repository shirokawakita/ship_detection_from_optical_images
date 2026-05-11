# GitHub 公開・clone 用キット

このディレクトリは、**GitHub に載せたソースを他PCで再現する**ときの手順を **一箇所に集約**したものです。コード本体はリポジトリルートのままです。

## 含まれるファイル

| ファイル | 内容 |
|----------|------|
| [push-to-github.md](push-to-github.md) | リモート作成前の確認、`.gitignore` の意味、初回 `push` の例 |
| [clone-and-implement.md](clone-and-implement.md) | **データ取得・`data.yaml` 整備・hold-out 学習で `best.pt` まで**（clone は最小限の入口として最後に記載） |

## すぐ読む順番

1. **初めて GitHub に上げる側** → [push-to-github.md](push-to-github.md)  
2. **同じ実装（データセット取得 → hold-out 学習モデル）を別環境でやる側** → [clone-and-implement.md](clone-and-implement.md)（メイン手順）  
3. **細目・Cursor プロンプト・トラブルシュート** → [../holdout_training_on_another_pc.md](../holdout_training_on_another_pc.md)

## リポジトリの正

- **本番のコード・設定**: ワークスペース**ルート**（`pyproject.toml`、`src/`、`scripts/`、`configs/`）。
- この `docs/github-clone-kit/` は **手順書のみ**（実行コードは含みません）。
