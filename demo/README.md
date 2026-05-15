# CPU デモ（Streamlit）

学習済み `best.pt` を `demo/weights/best.pt` に置いたうえで:

```bash
cd demo
uv sync
uv run streamlit run streamlit_app.py
```

`sample_images/` の JPG をアップロードして動作確認できます。重みの学習手順はリポジトリ直下の [README.md](../README.md) を参照してください。
