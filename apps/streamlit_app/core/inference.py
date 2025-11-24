from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from shapely.geometry.base import BaseGeometry

from apps.streamlit_app.core.geo import bounds_to_polygon
from apps.streamlit_app.core.metrics import calc_pr_metrics


def _extract_features_all_valid(x_tb_hw: np.ndarray, valid_hw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten tile into per-pixel feature vectors, keep only valid pixels."""
    T, B, H, W = x_tb_hw.shape
    flat = x_tb_hw.reshape(T * B, H * W).T
    valid_idx = np.flatnonzero((valid_hw > 0.5).reshape(-1))
    return flat[valid_idx].astype(np.float32), valid_idx


def run_inference(
    geom: BaseGeometry,
    ds,
    model,
    tiles_idx: List[Dict[str, Any]],
    prob_th: float,
    pixels_cap: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    tile_polys = [bounds_to_polygon(r["bounds"]) for r in tiles_idx]
    selected: List[int] = [i for i, poly in enumerate(tile_polys) if geom.intersects(poly)]

    if not selected:
        st.error("⚠️ **No data tiles found in the selected region.**")
        st.info("Draw your shape so that it overlaps the blue tile rectangles – only those areas have data.")
        return [], []

    cover_rows: List[Dict[str, Any]] = []

    prog = st.progress(0.0, text="Running inference…")
    cap = int(pixels_cap) if pixels_cap and pixels_cap > 0 else None

    canceled = False
    for k, idx in enumerate(selected):
        if st.session_state.get("cancel_inference"):
            canceled = True
            break
        rec = tiles_idx[idx]
        item = ds[idx]

        x = item["x"].numpy()
        valid = item["valid_mask"].numpy()[0] > 0.5
        n_valid_total = int(valid.sum())

        flat, valid_idx = _extract_features_all_valid(x, valid)

        if cap is not None and len(valid_idx) > cap:
            rng = np.random.default_rng(42)
            sampled = rng.choice(len(valid_idx), cap, replace=False)
            valid_idx = valid_idx[sampled]
            flat = flat[sampled]

        coverage_gt = None
        coverage_delta = None
        precision = None
        recall = None
        iou = None
        tp = fp = fn = tn = 0
        has_gt = False

        if len(flat) == 0:
            cov = 0.0
        else:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(flat)[:, -1]
            else:
                logits = model.decision_function(flat)
                proba = 1.0 / (1.0 + np.exp(-logits))
            pred = (proba >= prob_th).astype(np.uint8)
            cov = float(pred.mean())

            if "wheat_mask" in item:
                wheat = item["wheat_mask"].numpy()[0] > 0.5
                gt_flat = wheat.reshape(-1)[valid_idx]
                if gt_flat.size > 0:
                    has_gt = True
                    coverage_gt = float(gt_flat.mean())
                    coverage_delta = round(cov - coverage_gt, 4)
                    metrics = calc_pr_metrics(pred, gt_flat)
                    precision = metrics["precision"]
                    recall = metrics["recall"]
                    iou = metrics["iou"]
                    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]

        cover_rows.append(
            {
                "region": rec["region"],
                "tile_id": rec["tile_id"],
                "tile_index": idx,
                "bounds": rec["bounds"],
                "coverage_pred": round(cov, 4),
                "coverage_gt": None if coverage_gt is None else round(coverage_gt, 4),
                "coverage_delta": coverage_delta,
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "has_gt": has_gt,
                "n_pixels": len(flat),
                "n_valid_total": n_valid_total,
                "sample_rate": (len(flat) / n_valid_total) if n_valid_total > 0 else None,
            }
        )
        prog.progress((k + 1) / len(selected), text=f"Processing tile {k + 1}/{len(selected)}")

    prog.empty()
    if canceled:
        st.warning("Inference canceled.")
    else:
        st.success("✅ Inference finished! Open **3️⃣ Results & Analysis** in the sidebar.")
    return cover_rows, selected
