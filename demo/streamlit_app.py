"""Streamlit UI (CPU): upload a satellite image, auto SAHI when larger than training imgsz, show confidences."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

from ship_sat.infer_sahi import export_prediction_visual, predict_sahi


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    st.set_page_config(page_title="Ship OBB (CPU + SAHI)", layout="wide")
    st.title("光学衛星画像 — 船舶検出（CPU / YOLO-OBB + SAHI）")
    st.caption(
        "推論は **CPU のみ**で実行します。学習時の入力サイズより大きい画像では **SAHI タイル推論** を自動で使います。"
    )

    root = _project_root()
    cfg_path = root / "configs/default.yaml"
    cfg = _load_config(cfg_path)
    train_imgsz = int(cfg.get("imgsz", 768))
    sahi_cfg = cfg.get("sahi") or {}

    default_w = os.environ.get("SHIP_OBB_WEIGHTS", str(root / "weights" / "best.pt"))
    weights = st.text_input("重みパス (best.pt)", value=default_w)

    with st.expander("推論パラメータ（変更したら「設定で再推論」）", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            conf = st.slider(
                "信頼度閾値",
                0.05,
                0.95,
                float(sahi_cfg.get("confidence_threshold", 0.25)),
                0.05,
            )
        with col2:
            sw = st.number_input(
                "タイル幅（SAHI 時）",
                min_value=128,
                max_value=2048,
                value=int(sahi_cfg.get("slice_width", 640)),
                step=32,
            )
        with col3:
            sh = st.number_input(
                "タイル高さ（SAHI 時）",
                min_value=128,
                max_value=2048,
                value=int(sahi_cfg.get("slice_height", 640)),
                step=32,
            )
        owr = st.slider(
            "横オーバーラップ比（SAHI 時）",
            0.0,
            0.5,
            float(sahi_cfg.get("overlap_width_ratio", 0.2)),
            0.05,
        )
        ohr = st.slider(
            "縦オーバーラップ比（SAHI 時）",
            0.0,
            0.5,
            float(sahi_cfg.get("overlap_height_ratio", 0.2)),
            0.05,
        )
        force_sahi = st.checkbox("大きさに関わらず常に SAHI を使う", value=False)
        force_full = st.checkbox("常に単発推論（デバッグ用・大画像では非推奨）", value=False)
        imgsz_backend = st.number_input(
            "バックエンド imgsz（学習に合わせる）",
            min_value=0,
            max_value=2048,
            value=train_imgsz,
            step=32,
            help="0 のときはモデル既定。学習が 768 の場合は 768 を推奨。",
        )
    imgsz_opt = int(imgsz_backend) if imgsz_backend > 0 else None

    uploaded = st.file_uploader(
        "新規画像を選択",
        type=["png", "jpg", "jpeg", "tif", "tiff", "webp"],
        help="新しいファイルを選ぶと自動で推論します。閾値やタイルを変えたときは「設定で再推論」を押してください。",
    )

    rerun = st.button("設定で再推論", type="secondary")

    if uploaded is None:
        for k in ("ship_last_file_hash", "ship_result_key", "ship_png", "ship_rows"):
            st.session_state.pop(k, None)
        st.info("上から画像をアップロードしてください。")
        return

    raw = uploaded.getvalue()
    with Image.open(io.BytesIO(raw)) as im:
        img_w, img_h = im.size

    max_side = max(img_w, img_h)
    if force_full:
        use_slices = False
    elif force_sahi:
        use_slices = True
    else:
        use_slices = max_side > train_imgsz

    mode_label = "SAHI タイル推論" if use_slices else "単発推論（画像の長辺 ≤ 学習 imgsz）"
    st.metric("入力画像サイズ", f"{img_w} × {img_h} px")
    st.info(
        f"学習時 **imgsz = {train_imgsz}**（`configs/default.yaml`）と比較し、長辺 **{max_side} px** のため **{mode_label}** を使用します。"
    )

    wpath = Path(weights).expanduser()
    if not wpath.is_file():
        st.error(f"重みが見つかりません: {wpath}")
        return

    file_hash = hashlib.sha256(raw).hexdigest()
    last_file_hash = st.session_state.get("ship_last_file_hash")
    is_new_file = file_hash != last_file_hash

    param_key = (
        file_hash,
        str(wpath.resolve()),
        float(conf),
        int(sw),
        int(sh),
        float(owr),
        float(ohr),
        use_slices,
        imgsz_opt,
        force_sahi,
        force_full,
    )

    should_run = is_new_file or rerun
    last_result_key = st.session_state.get("ship_result_key")

    if not should_run and last_result_key != param_key:
        st.warning("パラメータが変わっています。「設定で再推論」で結果を更新してください。")
        st.subheader("入力プレビュー")
        st.image(io.BytesIO(raw), caption=uploaded.name, use_container_width=True)
        return

    if should_run:
        suffix = Path(uploaded.name).suffix or ".png"
        with st.spinner("CPU で推論中…（大画像・SAHI 時は数分かかることがあります）"):
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                src = td_path / f"input{suffix}"
                src.write_bytes(raw)
                out = td_path / "prediction.png"
                try:
                    result = predict_sahi(
                        wpath,
                        src,
                        device="cpu",
                        confidence_threshold=float(conf),
                        slice_width=int(sw),
                        slice_height=int(sh),
                        overlap_width_ratio=float(owr),
                        overlap_height_ratio=float(ohr),
                        use_slices=use_slices,
                        image_size=imgsz_opt,
                    )
                    out_path = export_prediction_visual(result, out, hide_conf=False)
                    png_bytes = out_path.read_bytes()
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    return

        preds = result.object_prediction_list
        rows = []
        for i, o in enumerate(
            sorted(preds, key=lambda p: p.score.value, reverse=True),
            start=1,
        ):
            rows.append(
                {
                    "順位": i,
                    "クラス": o.category.name,
                    "信頼度": round(float(o.score.value), 4),
                }
            )
        st.session_state["ship_last_file_hash"] = file_hash
        st.session_state["ship_result_key"] = param_key
        st.session_state["ship_png"] = png_bytes
        st.session_state["ship_rows"] = rows
    else:
        png_bytes = st.session_state.get("ship_png")
        rows = st.session_state.get("ship_rows", [])
        if png_bytes is None:
            st.error("表示用の推論結果がありません。画像を選び直すか、ページを再読み込みしてください。")
            return

    st.success(f"検出数: **{len(rows)}**（画像上は `クラス 信頼度` 形式で表示）")
    st.image(png_bytes, caption="推論結果（信頼度付き）", use_container_width=True)

    if rows:
        st.subheader("信頼度一覧")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("閾値を下げるか、別の画像を試してください（この閾値では検出がありません）。")

    st.download_button(
        "PNG をダウンロード",
        data=png_bytes,
        file_name="ship_obb_prediction.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
