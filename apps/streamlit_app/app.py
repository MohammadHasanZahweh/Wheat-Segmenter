from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import requests
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape, Polygon
from shapely.geometry.base import BaseGeometry
import plotly.graph_objects as go

import rasterio
from rasterio.warp import transform_bounds

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from wheat_segmenter import WheatTilesDataset

DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"C:\Users\Administrator\Desktop\preprocessed_data"))


# -----------------------------------------------------------------------------
# Dataclass for initial CLI config
# -----------------------------------------------------------------------------
@dataclass
class AppConfig:
    root: str
    year: str
    months: tuple[int, ...]


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def inject_global_styles() -> None:
    """Inject CSS for the dark UI."""
    st.markdown(
        """
        <style>
        :root {
            --bg1: #0a1526;
            --bg2: #07101d;
            --panel: rgba(11, 20, 34, 0.9);
            --accent: #5ee6a0;
            --accent-2: #74f1ff;
            --border: rgba(115, 161, 255, 0.3);
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background: linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 50%, #050915 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            background: transparent;
        }
        .main-block {
            padding: 1.5rem 1.75rem;
            border-radius: 1rem;
            background: var(--panel);
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }
        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            background: linear-gradient(120deg, var(--accent-2), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .main-subtitle {
            color: #9ca3af;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
        }
        .instruction-list li {
            margin-bottom: 0.25rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c1830 0%, #081122 100%);
        }
        .sidebar-title {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
            color: var(--accent);
        }
        .metrics-row {
            padding: 0.75rem 1rem 0.25rem 1rem;
            border-radius: 0.9rem;
            background: linear-gradient(120deg, rgba(12,25,43,0.95), rgba(9,18,34,0.95));
            border: 1px solid var(--border);
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.25);
        }
        .stat-card {
            background: rgba(13, 24, 41, 0.9);
            border-left: 4px solid var(--accent);
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .highlight-positive {
            color: var(--accent);
            font-weight: 600;
        }
        .highlight-warning {
            color: #ffba66;
            font-weight: 600;
        }
        /* Fix white header bars */
        header[data-testid="stHeader"], .stToolbar {
            background: transparent;
        }
        /* Tabs text color */
        .stTabs [data-baseweb="tab"] {
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Geo / dataset helpers
# -----------------------------------------------------------------------------
def _tile_bounds_latlon(month_paths: Dict[int, str]) -> Tuple[float, float, float, float] | None:
    """Get bounds of a tile in EPSG:4326 (lon/lat)."""
    for _, p in month_paths.items():
        try:
            with rasterio.open(p) as ds:
                b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            return b
        except Exception:
            continue
    return None


def load_tiles_index(ds: WheatTilesDataset) -> List[Dict[str, Any]]:
    """Create index of tiles with lat/lon bounds."""
    idx: List[Dict[str, Any]] = []
    for rec in ds.index:
        bounds_ll = _tile_bounds_latlon(rec["month_paths"])
        if bounds_ll is None:
            continue
        idx.append(
            {
                "region": rec["region"],
                "tile_id": rec["tile_id"],
                "bounds": bounds_ll,
            }
        )
    return idx


def _bounds_to_polygon(b: Tuple[float, float, float, float]) -> Polygon:
    minx, miny, maxx, maxy = b
    return Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])


def _extract_features_all_valid(x_tb_hw: np.ndarray, valid_hw: np.ndarray):
    """Flatten tile into per-pixel feature vectors, keep only valid pixels."""
    T, B, H, W = x_tb_hw.shape
    flat = x_tb_hw.reshape(T * B, H * W).T  # [H*W, T*B]
    valid_idx = np.flatnonzero((valid_hw > 0.5).reshape(-1))
    return flat[valid_idx].astype(np.float32), valid_idx


def _parse_regions(text: str) -> List[str] | None:
    tokens = [tok.strip() for tok in text.replace(",", " ").split()]
    regions = [tok for tok in tokens if tok]
    return regions or None


def cov_color(v: float) -> str:
    """Color for coverage: green (high) → red (low)."""
    v = max(0.0, min(1.0, float(v)))
    if v > 0.7:
        return "#00FF00"
    elif v > 0.5:
        return "#7FFF00"
    elif v > 0.3:
        return "#FFFF00"
    elif v > 0.1:
        return "#FFA500"
    else:
        return "#FF0000"


# -----------------------------------------------------------------------------
# Export / analytics helpers
# -----------------------------------------------------------------------------
def _export_geojson(
    cover_rows: List[Dict[str, Any]],
    selected: List[int],
    tiles_idx: List[Dict[str, Any]],
) -> str:
    """Export results as GeoJSON FeatureCollection."""
    features = []
    for rec, row in zip([tiles_idx[i] for i in selected], cover_rows):
        poly = _bounds_to_polygon(rec["bounds"])
        features.append(
            {
                "type": "Feature",
                "geometry": poly.__geo_interface__,
                "properties": {
                    "tile_id": row["tile_id"],
                    "region": row["region"],
                    "wheat_coverage": row["coverage_pred"],
                    "wheat_coverage_gt": row.get("coverage_gt"),
                    "coverage_delta": row.get("coverage_delta"),
                    "pixels_analyzed": row["n_pixels"],
                    "n_valid_total": row.get("n_valid_total"),
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "iou": row.get("iou"),
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)


def _create_coverage_distribution_chart(cover_rows: List[Dict[str, Any]]) -> go.Figure:
    """Interactive histogram of coverage distribution."""
    coverages = [r["coverage_pred"] for r in cover_rows]
    fig = go.Figure(
        data=[
            go.Histogram(
                x=coverages,
                nbinsx=15,
                name="Tiles",
                marker_color="rgba(127, 255, 0, 0.7)",
                hovertemplate="<b>Coverage:</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="📊 Wheat Coverage Distribution",
        xaxis_title="Coverage (0–1)",
        yaxis_title="Number of Tiles",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _create_coverage_by_region_chart(cover_rows: List[Dict[str, Any]]) -> go.Figure:
    """Bar chart of average coverage by region."""
    df = pd.DataFrame(cover_rows)
    region_stats = df.groupby("region").agg({"coverage_pred": ["mean", "count"]}).round(4)
    region_stats.columns = ["avg_coverage", "tile_count"]
    region_stats = region_stats.reset_index().sort_values("avg_coverage", ascending=False)

    fig = go.Figure(
        data=[
            go.Bar(
                x=region_stats["region"],
                y=region_stats["avg_coverage"],
                marker=dict(
                    color=region_stats["avg_coverage"],
                    colorscale="RdYlGn",
                    line=dict(color="rgba(255,255,255,0.2)", width=1),
                ),
                text=region_stats["avg_coverage"].apply(lambda x: f"{x:.1%}"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Avg Coverage: %{y:.1%}<br>Tiles: %{customdata}<extra></extra>",
                customdata=region_stats["tile_count"],
            )
        ]
    )
    fig.update_layout(
        title="🌾 Average Coverage by Region",
        xaxis_title="Region",
        yaxis_title="Average Coverage",
        template="plotly_dark",
        height=400,
        hovermode="x",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def _create_tile_ranking_chart(cover_rows: List[Dict[str, Any]]) -> go.Figure:
    """Line chart ranking tiles by coverage."""
    df = pd.DataFrame(cover_rows)
    df = df.sort_values("coverage_pred", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["rank"],
                y=df["coverage_pred"],
                mode="lines+markers",
                marker=dict(size=6),
                hovertemplate="Rank %{x}<br>Tile: %{text}<br>Coverage: %{y:.2%}<extra></extra>",
                text=df["tile_id"],
            )
        ]
    )
    fig.update_layout(
        title="📈 Tile Coverage Ranking",
        xaxis_title="Tile Rank (highest coverage → lowest)",
        yaxis_title="Coverage",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _create_summary_statistics(cover_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate descriptive statistics."""
    coverages = np.array([r["coverage_pred"] for r in cover_rows])
    return {
        "mean": float(np.mean(coverages)),
        "median": float(np.median(coverages)),
        "std": float(np.std(coverages)),
        "min": float(np.min(coverages)),
        "max": float(np.max(coverages)),
        "q25": float(np.percentile(coverages, 25)),
        "q75": float(np.percentile(coverages, 75)),
    }


def _calc_pr_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """Precision/recall/IoU over boolean vectors, returns counts for aggregation."""
    if pred.size == 0 or target.size == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": None, "recall": None, "iou": None}

    pred_b = pred.astype(bool)
    tgt_b = target.astype(bool)

    tp = int(np.logical_and(pred_b, tgt_b).sum())
    fp = int(np.logical_and(pred_b, ~tgt_b).sum())
    fn = int(np.logical_and(~pred_b, tgt_b).sum())
    tn = int(np.logical_and(~pred_b, ~tgt_b).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
    }


def _aggregate_verification(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-tile verification metrics."""
    has_gt = [r for r in rows if r.get("coverage_gt") is not None]
    if not has_gt:
        return {
            "avg_gt": None,
            "avg_delta": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "iou": None,
        }

    avg_gt = float(np.mean([r["coverage_gt"] for r in has_gt]))
    avg_delta = float(np.mean([r["coverage_delta"] for r in has_gt]))

    tp = sum(r.get("tp", 0) for r in has_gt)
    fp = sum(r.get("fp", 0) for r in has_gt)
    fn = sum(r.get("fn", 0) for r in has_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else None
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None

    return {
        "avg_gt": avg_gt,
        "avg_delta": avg_delta,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


# -----------------------------------------------------------------------------
# Main Streamlit app
# -----------------------------------------------------------------------------
def main_streamlit(app_cfg: AppConfig) -> None:
    st.set_page_config(page_title="Wheat Map (Lebanon)", layout="wide")
    inject_global_styles()
    st.title("🌾 Wheat Coverage Map (Lebanon)")
    st.markdown("""
    **Instructions:**
    1. Load dataset and model using the sidebar → Click "🔄 Load Dataset & Model"
    2. **IMPORTANT:** You can only analyze regions where data tiles exist (visible rectangles on map)
    3. Draw a polygon/rectangle/hexagon **over the visible tile boundaries**
    4. Click **Run inference** to see wheat coverage predictions
    5. Results persist - scroll down to see colored map, table, and statistics
    
    ⚠️ **Data Coverage:** The rectangles on the map show where your satellite data exists. 
    You cannot analyze areas outside these tiles!
    """)

    with st.expander("🧠 Training Jobs (API)", expanded=False):
        job_id = st.session_state.get("train_job_id")
        job_status = st.session_state.get("train_job_status")
        if job_id:
            st.markdown(f"**Current job ID:** `{job_id}`")
            if job_status:
                st.json(job_status)
            else:
                st.info("No job status yet. Use the sidebar to refresh.")
        else:
            st.info("Configure and launch a training job from the sidebar to track it here.")

    nav = st.sidebar.radio(
        "Navigation",
        ["1️⃣ Welcome", "2️⃣ Configure & Select Region", "3️⃣ Results & Analysis"],
        index=1,
    )

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Model settings
    st.sidebar.markdown('<div class="sidebar-title">Model Settings</div>', unsafe_allow_html=True)
    model_path = st.sidebar.text_input("Model .joblib path", value="runs/xgb_2020.joblib")
    prob_th = st.sidebar.slider("Probability threshold", 0.0, 1.0, 0.5, 0.05)
    pixels_cap = st.sidebar.number_input(
        "Pixels cap per tile (0 = all)", min_value=0, value=2000, step=500
    )

    st.sidebar.markdown("---")

    # Dataset settings
    st.sidebar.markdown('<div class="sidebar-title">Dataset Settings</div>', unsafe_allow_html=True)
    root = st.sidebar.text_input(
        "Root (contains data/ and label/)",
        value=app_cfg.root,
        help="Example: C:/Users/user/Desktop/preprocessed_data",
    )
    year = st.sidebar.text_input("Year", value=app_cfg.year)
    months_text = st.sidebar.text_input(
        "Months (space-separated)", value=" ".join(map(str, app_cfg.months))
    )

    if months_text.strip():
        try:
            months_sequence = tuple(int(m) for m in months_text.strip().split())
        except ValueError:
            st.sidebar.error("Invalid months input; falling back to defaults.")
            months_sequence = app_cfg.months
    else:
        months_sequence = app_cfg.months

    if "train_job_id" not in st.session_state:
        st.session_state["train_job_id"] = None
    if "train_job_status" not in st.session_state:
        st.session_state["train_job_status"] = None

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">🧠 Train Model via API</div>', unsafe_allow_html=True)
    api_url = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
    job_name = st.sidebar.text_input("Job name", value="streamlit_job")
    algo_labels = {
        "xgboost": "XGBoost",
        "hist_gradient_boosting": "HistGradientBoosting",
        "random_forest": "Random Forest",
        "svm": "Support Vector Machine",
    }
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        options=list(algo_labels.keys()),
        format_func=lambda key: algo_labels.get(key, key.title()),
    )
    regions_text = st.sidebar.text_input("Regions (optional)", placeholder="e.g. 0 1 2")
    train_fraction = st.sidebar.number_input(
        "Train fraction", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f"
    )
    test_fraction = st.sidebar.number_input(
        "Test fraction", min_value=0.0, max_value=1.0, value=0.25, step=0.05, format="%.2f"
    )
    pixels_per_tile = st.sidebar.number_input("Pixels per tile", min_value=256, value=4096, step=512)
    balance_pixels = st.sidebar.checkbox("Balance sampled pixels", value=False)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
    save_model = st.sidebar.checkbox("Save trained model", value=True)
    output_path = st.sidebar.text_input("Save path override (optional)", value="")

    model_params: Dict[str, Any] = {}
    with st.sidebar.expander(f"{algo_labels.get(algorithm, algorithm.title())} Hyperparameters", expanded=False):
        if algorithm == "xgboost":
            xgb_estimators = st.number_input("n_estimators", min_value=10, value=400, step=10, key="xgb_estimators")
            xgb_depth = st.number_input("max_depth", min_value=1, value=8, step=1, key="xgb_depth")
            xgb_lr = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.05, step=0.01, format="%.3f", key="xgb_lr")
            xgb_subsample = st.number_input("subsample", min_value=0.1, max_value=1.0, value=0.8, step=0.05, format="%.2f", key="xgb_subsample")
            xgb_colsample = st.number_input("colsample_bytree", min_value=0.1, max_value=1.0, value=0.8, step=0.05, format="%.2f", key="xgb_colsample")
            model_params = {
                "n_estimators": int(xgb_estimators),
                "max_depth": int(xgb_depth),
                "learning_rate": float(xgb_lr),
                "subsample": float(xgb_subsample),
                "colsample_bytree": float(xgb_colsample),
            }
        elif algorithm == "hist_gradient_boosting":
            hgb_depth = st.number_input("max_depth (0=None)", min_value=0, value=8, step=1, key="hgb_depth")
            hgb_iter = st.number_input("max_iter", min_value=10, value=400, step=10, key="hgb_iter")
            hgb_lr = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.05, step=0.01, format="%.3f", key="hgb_lr")
            hgb_l2 = st.number_input("l2_regularization", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="hgb_l2")
            model_params = {
                "max_depth": int(hgb_depth),
                "max_iter": int(hgb_iter),
                "learning_rate": float(hgb_lr),
                "l2_regularization": float(hgb_l2),
            }
        elif algorithm == "random_forest":
            rf_estimators = st.number_input("n_estimators", min_value=10, value=200, step=10, key="rf_estimators")
            rf_depth = st.number_input("max_depth (0=None)", min_value=0, value=0, step=1, key="rf_depth")
            model_params = {
                "rf_estimators": int(rf_estimators),
                "rf_max_depth": int(rf_depth) if rf_depth > 0 else None,
            }
        elif algorithm == "svm":
            svm_kernel = st.selectbox("kernel", options=["rbf", "linear", "poly", "sigmoid"], index=0, key="svm_kernel")
            svm_c = st.number_input("C", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="svm_c")
            svm_gamma = st.text_input("gamma (scale, auto, or float)", value="scale", key="svm_gamma")
            model_params = {
                "svm_kernel": svm_kernel,
                "svm_C": float(svm_c),
                "svm_gamma": svm_gamma.strip() or "scale",
            }

    start_training = st.sidebar.button("🚀 Start Training Job", use_container_width=True)
    refresh_training = st.sidebar.button(
        "🔁 Refresh Job Status",
        use_container_width=True,
        disabled=not bool(st.session_state.get("train_job_id")),
    )
    clear_training = st.sidebar.button(
        "🗑 Clear Job Info",
        use_container_width=True,
        disabled=not bool(st.session_state.get("train_job_id")),
    )
    training_feedback = st.sidebar.empty()

    api_url_clean = api_url.rstrip("/")
    if start_training:
        if not api_url_clean:
            training_feedback.error("Provide a valid API base URL.")
        elif not root or not year:
            training_feedback.error("Root and year are required to submit a training job.")
        else:
            dataset_cfg = {
                "root": root,
                "year": year,
                "regions": _parse_regions(regions_text),
                "months": list(months_sequence),
                "train_fraction": float(train_fraction),
                "test_fraction": float(test_fraction),
                "pixels_per_tile": int(pixels_per_tile),
                "balance_pixels": bool(balance_pixels),
                "seed": int(seed),
            }
            payload = {
                "job_name": job_name.strip() or f"streamlit_job_{int(time.time())}",
                "algorithm": algorithm,
                "dataset": dataset_cfg,
                "save_model": bool(save_model),
                "output_path": output_path.strip() or None,
                "model_params": model_params or None,
            }
            try:
                resp = requests.post(f"{api_url_clean}/train", json=payload, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                job_id = data.get("job_id")
                if not job_id:
                    raise ValueError(f"Unexpected response: {data}")
                st.session_state["train_job_id"] = job_id
                st.session_state["train_job_status"] = {"status": "running", **data}
                training_feedback.success(f"Job started with id {job_id}")
            except requests.RequestException as exc:
                training_feedback.error(f"Failed to start job: {exc}")
            except Exception as exc:
                training_feedback.error(f"Unexpected error: {exc}")

    if refresh_training and st.session_state.get("train_job_id"):
        try:
            resp = requests.get(
                f"{api_url_clean}/train/status",
                params={"id": st.session_state["train_job_id"]},
                timeout=10,
            )
            resp.raise_for_status()
            status_data = resp.json()
            st.session_state["train_job_status"] = status_data
            training_feedback.info(f"Status: {status_data.get('status', 'unknown')}")
        except requests.RequestException as exc:
            training_feedback.error(f"Failed to fetch status: {exc}")

    if clear_training and st.session_state.get("train_job_id"):
        st.session_state["train_job_id"] = None
        st.session_state["train_job_status"] = None
        training_feedback.info("Cleared stored job info.")

    # Load data button
    load_clicked = st.sidebar.button("🔄 Load Dataset & Model", type="primary", use_container_width=True)
    if load_clicked:
        for key in ("results_data", "results_selected", "results_tiles_idx"):
            st.session_state.pop(key, None)

    # MODEL LOADING
    if "model" not in st.session_state or load_clicked:
        st.session_state["model"] = None
        if model_path:
            try:
                st.session_state["model"] = joblib.load(model_path)
                st.sidebar.success(f"✅ Model loaded: {Path(model_path).name}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load model: {e}")

    # DATASET LOADING
    if "dataset" not in st.session_state or load_clicked:
        st.session_state["dataset"] = None
        st.session_state["tiles_index"] = []
        if root and year:
            try:
                with st.spinner("Loading dataset..."):
                    ds = WheatTilesDataset(
                        root_preprocessed=root,
                        year=year,
                        regions=None,
                        month_order=months_sequence,
                        temporal_layout=True,
                        normalize=True,
                        band_stats=None,
                        require_complete=True,
                        target_bands=None,
                        target_size=(64, 64),
                        size_policy="pad",
                        probe_limit=12,
                    )
                    st.session_state["dataset"] = ds
                    st.session_state["tiles_index"] = load_tiles_index(ds)
                    st.sidebar.success(f"✅ Loaded {len(ds)} tiles")
            except Exception as e:
                st.sidebar.error(f"❌ Dataset load failed: {e}")
                st.session_state["dataset"] = None

    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">📊 Status</div>', unsafe_allow_html=True)
    model_status = "✅ Loaded" if st.session_state.get("model") is not None else "❌ Not loaded"
    dataset_status = "✅ Loaded" if st.session_state.get("dataset") is not None else "❌ Not loaded"
    tiles_count = len(st.session_state.get("tiles_index", []))
    st.sidebar.markdown(
        f"""
        - **Model:** {model_status}  
        - **Dataset:** {dataset_status}  
        - **Tiles with bounds:** {tiles_count}
        """
    )

    # =====================================================================
    # STEP 1 – WELCOME
    # =====================================================================
    if nav == "1️⃣ Welcome":
        st.markdown(
            """
            ### Welcome to the wheat segmentation dashboard

            This tool uses **satellite imagery** and **machine learning models**  
            to estimate how much of a selected area is covered by wheat.

            #### What you can do here
            - Load a pre-trained model (**XGBoost** or **HistGradientBoosting**).
            - Select a preprocessed dataset (tiles for a given year).
            - Draw a region of interest directly on the map.
            - Generate coverage maps colored by estimated wheat percentage.
            - Analyze distributions, compare regions, and export results.

            #### Workflow
            1. Open **2️⃣ Configure & Select Region** in the sidebar.  
            2. Choose the dataset root, year and model, then draw your polygon/rectangle.  
            3. Run inference, then switch to **3️⃣ Results & Analysis** to explore charts and exports.
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "ℹ️ The app expects preprocessed data under **`data/` and `label/`** "
            "inside your root folder, with `{year}/{region}/{month}` hierarchies."
        )
        return

    # =====================================================================
    # STEP 2 – CONFIGURE & SELECT REGION
    # =====================================================================
    if nav == "2️⃣ Configure & Select Region":
        st.subheader("📍 Step 2 – Select Region")

        # Base map
        m = folium.Map(
            location=[33.9, 35.9],
            zoom_start=8,
            control_scale=True,
            tiles="OpenStreetMap",
            zoom_control=True,
        )

        draw = Draw(
            export=False,
            position="topleft",
            draw_options={
                "polyline": False,
                "rectangle": True,
                "polygon": True,
                "circle": False,
                "marker": False,
                "circlemarker": False,
            },
            edit_options={"edit": False},
        )
        draw.add_to(m)

        # Draw tile outlines
        if st.session_state.get("tiles_index"):
            tile_count = len(st.session_state["tiles_index"])
            st.info(f"📍 **{tile_count} data tiles loaded** – draw your shape over these rectangles.")
            for rec in st.session_state["tiles_index"]:
                poly = _bounds_to_polygon(rec["bounds"]).exterior.coords[:]
                folium.PolyLine(
                    locations=[(y, x) for x, y in poly],
                    color="#0066FF",
                    weight=2,
                    opacity=0.6,
                    tooltip=f"Tile: {rec['tile_id']} | Region: {rec['region']}",
                ).add_to(m)
        else:
            st.warning("No tile boundaries available – check your dataset root and year.")

        map_output = st_folium(
            m,
            width=None,
            height=500,
            use_container_width=True,
            returned_objects=["all_drawings"],
            key="main_map",
        )

        geom: BaseGeometry | None = None
        if map_output and map_output.get("all_drawings"):
            drawings = map_output["all_drawings"]
            if drawings:
                geom = shape(drawings[-1]["geometry"])
                st.info(f"✅ Region selected: {drawings[-1]['geometry']['type']}")

        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            run_clicked = st.button(
                "🚀 Run Inference",
                type="primary",
                use_container_width=True,
                disabled=(geom is None),
            )

        with col2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                for key in ("results_data", "results_selected", "results_tiles_idx"):
                    st.session_state.pop(key, None)
                st.experimental_rerun()

        # INFERENCE
        if run_clicked and geom is not None:
            model = st.session_state.get("model")
            ds: WheatTilesDataset | None = st.session_state.get("dataset")
            tiles_idx = st.session_state.get("tiles_index", [])

            if model is None:
                st.error("❌ Please load a model first from the sidebar.")
            elif ds is None:
                st.error("❌ Please load a dataset first from the sidebar.")
            else:
                selected: List[int] = []
                for i, rec in enumerate(tiles_idx):
                    poly = _bounds_to_polygon(rec["bounds"])
                    if geom.intersects(poly):
                        selected.append(i)

                if not selected:
                    st.error("⚠️ **No data tiles found in the selected region.**")
                    st.info(
                        "Draw your shape so that it overlaps the blue tile rectangles – only those areas have data."
                    )
                else:
                    cover_rows: List[Dict[str, Any]] = []
                    prog = st.progress(0.0, text="Running inference…")
                    cap = int(pixels_cap) if pixels_cap and pixels_cap > 0 else None

                    for k, idx in enumerate(selected):
                        rec = tiles_idx[idx]
                        item = ds[idx]

                        x = item["x"].numpy()  # (T,B,H,W)
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

                            # Ground-truth verification
                            if "wheat_mask" in item:
                                wheat = item["wheat_mask"].numpy()[0] > 0.5
                                gt_flat = wheat.reshape(-1)[valid_idx]
                                if gt_flat.size > 0:
                                    coverage_gt = float(gt_flat.mean())
                                    coverage_delta = round(cov - coverage_gt, 4)
                                    metrics = _calc_pr_metrics(pred, gt_flat)
                                    precision = metrics["precision"]
                                    recall = metrics["recall"]
                                    iou = metrics["iou"]
                                    tp, fp, fn, tn = (
                                        metrics["tp"],
                                        metrics["fp"],
                                        metrics["fn"],
                                        metrics["tn"],
                                    )

                        cover_rows.append(
                            {
                                "region": rec["region"],
                                "tile_id": rec["tile_id"],
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
                                "n_pixels": len(flat),
                                "n_valid_total": n_valid_total,
                                "sample_rate": (len(flat) / n_valid_total) if n_valid_total > 0 else None,
                            }
                        )
                        prog.progress(
                            (k + 1) / len(selected),
                            text=f"Processing tile {k + 1}/{len(selected)}",
                        )

                    prog.empty()

                    st.session_state["results_data"] = cover_rows
                    st.session_state["results_selected"] = selected
                    st.session_state["results_tiles_idx"] = tiles_idx

                    st.success("✅ Inference finished! Open **3️⃣ Results & Analysis** in the sidebar.")
        return

    # =====================================================================
    # STEP 3 – RESULTS & ANALYSIS
    # =====================================================================
    if nav == "3️⃣ Results & Analysis":
        cover_rows = st.session_state.get("results_data")
        if not cover_rows:
            st.warning("No results yet. Run inference in **2️⃣ Configure & Select Region** first.")
            return

        selected = st.session_state["results_selected"]
        tiles_idx = st.session_state["results_tiles_idx"]

        st.subheader("🗺️ Step 3 – Results & Analysis")

        # Filters
        st.subheader("🔍 Filters")
        all_regions = sorted({r["region"] for r in cover_rows})
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

        with col_f1:
            selected_regions = st.multiselect(
                "Regions",
                options=all_regions,
                default=all_regions,
                help="Filter results to specific regions.",
            )

        with col_f2:
            cov_min, cov_max = st.slider(
                "Coverage range",
                0.0,
                1.0,
                (0.0, 1.0),
                step=0.05,
                help="Only keep tiles whose predicted coverage lies in this range.",
            )

        with col_f3:
            apply_filter_to_map = st.checkbox(
                "Filter map", value=False, help="Apply filters to the map as well."
            )

        filtered_rows = [
            r
            for r in cover_rows
            if r["region"] in selected_regions and cov_min <= r["coverage_pred"] <= cov_max
        ]

        if not filtered_rows:
            st.warning("Filters removed all tiles – adjust region/coverage filters.")
            filtered_rows = cover_rows

        avg_cov = float(np.mean([r["coverage_pred"] for r in filtered_rows]))
        max_cov = max([r["coverage_pred"] for r in filtered_rows])
        high_cov_count = sum(1 for r in filtered_rows if r["coverage_pred"] > 0.5)
        verification_summary = _aggregate_verification(filtered_rows)
        fmt_pct = lambda v: f"{v:.2%}" if v is not None else "—"
        fmt_delta = lambda v: f"{v:+.2%}" if v is not None else "—"

        st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Tiles (after filters)", len(filtered_rows))
        with mcol2:
            st.metric("Avg Coverage", f"{avg_cov:.2%}")
        with mcol3:
            st.metric("Max Coverage", f"{max_cov:.2%}")
        with mcol4:
            st.metric("High Coverage Tiles (>50%)", high_cov_count)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
        vcol1, vcol2, vcol3, vcol4 = st.columns(4)
        with vcol1:
            st.metric("Avg GT Coverage", fmt_pct(verification_summary["avg_gt"]))
        with vcol2:
            st.metric("Avg Coverage Δ (Pred-GT)", fmt_delta(verification_summary["avg_delta"]))
        with vcol3:
            st.metric("IoU (micro)", fmt_pct(verification_summary["iou"]))
        with vcol4:
            st.metric(
                "Precision / Recall",
                f"{fmt_pct(verification_summary['precision'])} / {fmt_pct(verification_summary['recall'])}",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🗺️ Map", "📊 Analytics", "📋 Data", "📈 Statistics", "💾 Export"]
        )

        # TAB 1: MAP
        with tab1:
            results_map = folium.Map(
                location=[33.9, 35.9],
                zoom_start=8,
                control_scale=True,
                tiles="OpenStreetMap",
            )

            if apply_filter_to_map:
                rows_for_map = filtered_rows
                idx_for_map = [tiles_idx[cover_rows.index(r)] for r in filtered_rows]
            else:
                rows_for_map = cover_rows
                idx_for_map = [tiles_idx[i] for i in selected]

            for rec, row in zip(idx_for_map, rows_for_map):
                poly = _bounds_to_polygon(rec["bounds"])
                color = cov_color(row["coverage_pred"])
                properties = {
                    "tile_id": row.get("tile_id"),
                    "region": row.get("region"),
                    "coverage_pred": row.get("coverage_pred"),
                    "coverage_gt": row.get("coverage_gt"),
                    "coverage_delta": row.get("coverage_delta"),
                    "n_pixels": row.get("n_pixels"),
                }

                folium.GeoJson(
                    data={
                        "type": "Feature",
                        "geometry": poly.__geo_interface__,
                        "properties": properties,
                    },
                    style_function=lambda feat, col=color: {
                        "fillColor": col,
                        "color": "#000",
                        "weight": 1.5,
                        "fillOpacity": 0.7,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["tile_id", "coverage_pred", "coverage_gt", "coverage_delta", "n_pixels"],
                        aliases=["Tile ID", "Wheat Coverage", "GT Coverage", "Delta", "Pixels Sampled"],
                        localize=True,
                    ),
                ).add_to(results_map)

            legend_html = """
            <div style="
                position: fixed;
                bottom: 50px;
                left: 50px;
                width: 220px;
                background-color: white;
                border: 1px solid grey;
                z-index: 9999;
                font-size: 13px;
                padding: 10px;
                border-radius: 8px;">
                <p style="margin:0; font-weight:bold;">🌾 Wheat Coverage Legend</p>
                <p style="margin:5px 0;"><span style="background-color:#00FF00; padding:2px 10px;">■</span> Very High (>70%)</p>
                <p style="margin:5px 0;"><span style="background-color:#7FFF00; padding:2px 10px;">■</span> High (50–70%)</p>
                <p style="margin:5px 0;"><span style="background-color:#FFFF00; padding:2px 10px;">■</span> Medium (30–50%)</p>
                <p style="margin:5px 0;"><span style="background-color:#FFA500; padding:2px 10px;">■</span> Low (10–30%)</p>
                <p style="margin:5px 0;"><span style="background-color:#FF0000; padding:2px 10px;">■</span> Very Low (<10%)</p>
            </div>
            """
            results_map.get_root().html.add_child(folium.Element(legend_html))

            st_folium(
                results_map,
                width=None,
                height=600,
                use_container_width=True,
                key="results_map_display",
            )

        # TAB 2: ANALYTICS
        with tab2:
            col_dist, col_region = st.columns(2)

            with col_dist:
                st.plotly_chart(
                    _create_coverage_distribution_chart(filtered_rows),
                    use_container_width=True,
                )

            with col_region:
                st.plotly_chart(
                    _create_coverage_by_region_chart(filtered_rows),
                    use_container_width=True,
                )

            st.markdown("---")
            st.plotly_chart(
                _create_tile_ranking_chart(filtered_rows),
                use_container_width=True,
            )

        # TAB 3: DATA TABLE
        with tab3:
            st.subheader("📋 Detailed Results (after filters)")
            df = pd.DataFrame(filtered_rows)
            df_display = df.copy()
            for missing_col in (
                "coverage_gt",
                "coverage_delta",
                "precision",
                "recall",
                "iou",
                "n_valid_total",
                "sample_rate",
            ):
                if missing_col not in df_display:
                    df_display[missing_col] = np.nan
            df_display["coverage_pred"] = df_display["coverage_pred"].apply(
                lambda x: f"{x:.1%}"
            )
            df_display["coverage_gt"] = df_display["coverage_gt"].apply(
                lambda x: f"{x:.1%}" if pd.notnull(x) else "—"
            )
            df_display["coverage_delta"] = df_display["coverage_delta"].apply(
                lambda x: f"{x:+.1%}" if pd.notnull(x) else "—"
            )
            for col in ("precision", "recall", "iou"):
                df_display[col] = df_display[col].apply(lambda v: f"{v:.1%}" if pd.notnull(v) else "—")
            df_display["sample_rate"] = df_display["sample_rate"].apply(
                lambda v: f"{v:.1%}" if pd.notnull(v) else "—"
            )
            ordered_cols = [
                "region",
                "tile_id",
                "coverage_pred",
                "coverage_gt",
                "coverage_delta",
                "precision",
                "recall",
                "iou",
                "n_pixels",
                "n_valid_total",
                "sample_rate",
            ]
            df_display = df_display[[c for c in ordered_cols if c in df_display.columns]]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        # TAB 4: STATISTICS
        with tab4:
            stats = _create_summary_statistics(filtered_rows)
            gt_values = [r["coverage_gt"] for r in filtered_rows if r.get("coverage_gt") is not None]
            delta_values = [r["coverage_delta"] for r in filtered_rows if r.get("coverage_delta") is not None]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    """
                    <div class="stat-card">
                        <strong>Central Tendency</strong><br>
                        Mean: <span class="highlight-positive">{mean:.1%}</span><br>
                        Median: <span class="highlight-positive">{median:.1%}</span>
                    </div>
                    """.format(**stats),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="stat-card">
                        <strong>Range</strong><br>
                        Min: <span class="highlight-warning">{min:.1%}</span><br>
                        Max: <span class="highlight-positive">{max:.1%}</span>
                    </div>
                    """.format(**stats),
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    """
                    <div class="stat-card">
                        <strong>Dispersion</strong><br>
                        Std Dev: <span class="highlight-warning">{std:.1%}</span><br>
                        Variance: <span class="highlight-warning">{var:.4f}</span>
                    </div>
                    """.format(std=stats["std"], var=stats["std"] ** 2),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="stat-card">
                        <strong>Quartiles</strong><br>
                        Q25: <span class="highlight-positive">{q25:.1%}</span><br>
                        Q75: <span class="highlight-positive">{q75:.1%}</span>
                    </div>
                    """.format(**stats),
                    unsafe_allow_html=True,
                )

            if gt_values:
                stats_gt_arr = np.array(gt_values)
                stats_gt = {
                    "mean": float(np.mean(stats_gt_arr)),
                    "median": float(np.median(stats_gt_arr)),
                    "min": float(np.min(stats_gt_arr)),
                    "max": float(np.max(stats_gt_arr)),
                }

                st.markdown("---")
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.markdown(
                        """
                        <div class="stat-card">
                            <strong>Ground Truth Coverage</strong><br>
                            Mean: <span class="highlight-positive">{mean:.1%}</span><br>
                            Median: <span class="highlight-positive">{median:.1%}</span><br>
                            Range: <span class="highlight-warning">{min:.1%}</span> → <span class="highlight-positive">{max:.1%}</span>
                        </div>
                        """.format(**stats_gt),
                        unsafe_allow_html=True,
                    )

                with gcol2:
                    ver = verification_summary
                    st.markdown(
                        """
                        <div class="stat-card">
                            <strong>Verification (micro)</strong><br>
                            Precision: <span class="highlight-positive">{precision}</span><br>
                            Recall: <span class="highlight-positive">{recall}</span><br>
                            IoU: <span class="highlight-positive">{iou}</span><br>
                            F1: <span class="highlight-positive">{f1}</span>
                        </div>
                        """.format(
                            precision=fmt_pct(ver.get("precision")),
                            recall=fmt_pct(ver.get("recall")),
                            iou=fmt_pct(ver.get("iou")),
                            f1=fmt_pct(ver.get("f1")),
                        ),
                        unsafe_allow_html=True,
                    )

                if delta_values:
                    delta_avg = float(np.mean(delta_values))
                    st.info(f"Average coverage delta (pred - GT): {fmt_delta(delta_avg)}")
            else:
                st.info("Ground-truth masks not available in the current results; verification skipped.")

        # TAB 5: EXPORT
        with tab5:
            stats_all = _create_summary_statistics(filtered_rows)
            regions = sorted({r["region"] for r in filtered_rows})
            regions_md = "\n".join(f"- {reg}" for reg in regions)
            high_cov_count_filtered = sum(
                1 for r in filtered_rows if r["coverage_pred"] > 0.5
            )
            high_percentage = 100 * high_cov_count_filtered / len(filtered_rows)

            col_csv, col_geojson, col_json = st.columns(3)

            # CSV
            with col_csv:
                fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.4f}"
                csv_headers = [
                    "region",
                    "tile_id",
                    "coverage_pred",
                    "coverage_gt",
                    "coverage_delta",
                    "precision",
                    "recall",
                    "iou",
                    "n_pixels",
                    "n_valid_total",
                    "sample_rate",
                ]
                csv_data = ",".join(csv_headers) + "\n"
                csv_rows = []
                for r in filtered_rows:
                    csv_rows.append(
                        ",".join(
                            [
                                str(r["region"]),
                                str(r["tile_id"]),
                                fmt(r.get("coverage_pred")),
                                fmt(r.get("coverage_gt")),
                                fmt(r.get("coverage_delta")),
                                fmt(r.get("precision")),
                                fmt(r.get("recall")),
                                fmt(r.get("iou")),
                                str(r.get("n_pixels", "")),
                                str(r.get("n_valid_total", "")),
                                fmt(r.get("sample_rate")),
                            ]
                        )
                    )
                csv_data += "\n".join(csv_rows)
                st.download_button(
                    "📥 CSV",
                    data=csv_data,
                    file_name=f"wheat_coverage_{year}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # GeoJSON
            with col_geojson:
                selected_indices = [cover_rows.index(r) for r in filtered_rows]
                geojson_data = _export_geojson(filtered_rows, selected_indices, tiles_idx)
                st.download_button(
                    "🗺️ GeoJSON",
                    data=geojson_data,
                    file_name=f"wheat_coverage_{year}.geojson",
                    mime="application/json",
                    use_container_width=True,
                )

            # JSON
            with col_json:
                json_data = json.dumps(
                    {
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "year": year,
                            "tiles_count": len(filtered_rows),
                            "statistics": stats_all,
                            "verification": verification_summary,
                            "filters": {
                                "regions": selected_regions,
                                "coverage_min": cov_min,
                                "coverage_max": cov_max,
                            },
                        },
                        "results": filtered_rows,
                    },
                    indent=2,
                )
                st.download_button(
                    "📄 JSON",
                    data=json_data,
                    file_name=f"wheat_coverage_{year}_full.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.markdown("---")

            # Summary report
            st.subheader("📄 Summary Report (after filters)")
            summary_text = f"""# Wheat Coverage Analysis Report
**Year:** {year}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Filters
- Regions: {', '.join(selected_regions)}
- Coverage range: {cov_min:.2f} – {cov_max:.2f}

    ## Overview
    - **Total Tiles Analyzed:** {len(filtered_rows)}
    - **Average Coverage:** {stats_all["mean"]:.2%}
    - **Coverage Range:** {stats_all["min"]:.2%} – {stats_all["max"]:.2%}
    - **High Coverage Tiles (>50%):** {high_cov_count_filtered} ({high_percentage:.1f}%)

## Verification (micro)
- **Ground Truth Avg Coverage:** {fmt_pct(verification_summary.get("avg_gt"))}
- **Avg Delta (Pred - GT):** {fmt_delta(verification_summary.get("avg_delta"))}
- **Precision / Recall:** {fmt_pct(verification_summary.get("precision"))} / {fmt_pct(verification_summary.get("recall"))}
- **IoU:** {fmt_pct(verification_summary.get("iou"))}
- **F1:** {fmt_pct(verification_summary.get("f1"))}

## Statistics
- **Median:** {stats_all["median"]:.2%}
- **Std Dev:** {stats_all["std"]:.2%}
- **Q1 (25th percentile):** {stats_all["q25"]:.2%}
- **Q3 (75th percentile):** {stats_all["q75"]:.2%}

## Regions Covered (after filters)
{regions_md}
"""
            st.markdown(summary_text)
            st.download_button(
                "📥 Download Report (Markdown)",
                data=summary_text,
                file_name=f"wheat_coverage_{year}_report.md",
                mime="text/markdown",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# CLI parser (for defaults)
# -----------------------------------------------------------------------------
def parse_cli() -> AppConfig:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--year", default="2020")
    ap.add_argument(
        "--months",
        nargs="*",
        type=int,
        default=[11, 12, 1, 2, 3, 4, 5, 6, 7],
    )
    try:
        args, _ = ap.parse_known_args()
    except SystemExit:
        class _Args: pass
        args = _Args(); args.root = str(DEFAULT_DATA_ROOT); args.year = "2020"; args.months = [11,12,1,2,3,4,5,6,7]
    return AppConfig(root=str(args.root), year=str(args.year), months=tuple(int(m) for m in args.months))


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = parse_cli()
    main_streamlit(cfg)
