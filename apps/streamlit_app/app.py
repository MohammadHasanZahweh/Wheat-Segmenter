from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import joblib
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape, Polygon
from shapely.geometry.base import BaseGeometry

import rasterio
from rasterio.warp import transform_bounds

import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Make sure we can import wheat_segmenter
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wheat_segmenter import WheatTilesDataset  # noqa: E402


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
        .stApp {
            background: #020617;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .main-block {
            padding: 1.5rem 1.75rem;
            border-radius: 1rem;
            background: rgba(15, 23, 42, 0.98);
            border: 1px solid rgba(148, 163, 184, 0.4);
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            background: linear-gradient(135deg, #7FFF00, #00FF00);
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
            background: #020617;
        }
        .sidebar-title {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
            color: #7FFF00;
        }
        .metrics-row {
            padding: 0.75rem 1rem 0.25rem 1rem;
            border-radius: 0.9rem;
            background: rgba(15,23,42,0.95);
            border: 1px solid rgba(51,65,85,0.9);
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .stat-card {
            background: rgba(30, 41, 59, 0.8);
            border-left: 4px solid #7FFF00;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .highlight-positive {
            color: #00FF00;
            font-weight: 600;
        }
        .highlight-warning {
            color: #FFA500;
            font-weight: 600;
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
                    "pixels_analyzed": row["n_pixels"],
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


# -----------------------------------------------------------------------------
# Main Streamlit app
# -----------------------------------------------------------------------------
def main_streamlit(app_cfg: AppConfig) -> None:
    st.set_page_config(page_title="Wheat Map (Lebanon)", layout="wide")
    inject_global_styles()

    # -------------------------------------------------
    # Navigation state (wizard)
    # -------------------------------------------------
    if "nav" not in st.session_state:
        st.session_state["nav"] = "1️⃣ Welcome"

    with st.sidebar:
        st.markdown("### Navigation")
        nav = st.radio(
            "",
            ["1️⃣ Welcome", "2️⃣ Configure & Select Region", "3️⃣ Results & Analysis"],
            key="nav",
        )

    # -------------------------------------------------
    # Global header
    # -------------------------------------------------
    with st.container():
        st.markdown(
            """
            <div class="main-block">
                <div class="main-title">🌾 Wheat Coverage Map (Lebanon)</div>
                <div class="main-subtitle">
                    Interactive dashboard to estimate wheat coverage from satellite tiles over Lebanese agricultural regions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------
    # Sidebar configuration (used in steps 2 & 3)
    # -------------------------------------------------
    st.sidebar.markdown("---")
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

    load_clicked = st.sidebar.button(
        "🔄 Load Dataset & Model", type="primary", use_container_width=True
    )
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
                if months_text.strip():
                    months = tuple(int(m) for m in months_text.strip().split())
                else:
                    months = app_cfg.months

                with st.spinner("📦 Loading dataset..."):
                    ds = WheatTilesDataset(
                        root_preprocessed=root,
                        year=year,
                        regions=None,
                        month_order=months,
                        temporal_layout=True,
                        normalize=True,
                        band_stats=None,
                        require_complete=True,
                        target_bands=None,
                        target_size=(64, 64),
                        size_policy="pad",
                        probe_limit=12,
                    )
                    tiles_index = load_tiles_index(ds)
                    st.session_state["dataset"] = ds
                    st.session_state["tiles_index"] = tiles_index
                    st.sidebar.success(f"✅ Loaded {len(ds)} tiles ({len(tiles_index)} with bounds)")
            except Exception as e:
                st.sidebar.error(f"❌ Dataset load failed: {e}")

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

                        flat, valid_idx = _extract_features_all_valid(x, valid)

                        if cap is not None and len(valid_idx) > cap:
                            rng = np.random.default_rng(42)
                            sampled = rng.choice(len(valid_idx), cap, replace=False)
                            valid_idx = valid_idx[sampled]
                            flat = flat[sampled]

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

                        cover_rows.append(
                            {
                                "region": rec["region"],
                                "tile_id": rec["tile_id"],
                                "coverage_pred": round(cov, 4),
                                "n_pixels": len(flat),
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

                folium.GeoJson(
                    data={
                        "type": "Feature",
                        "geometry": poly.__geo_interface__,
                        "properties": row,
                    },
                    style_function=lambda feat, col=color: {
                        "fillColor": col,
                        "color": "#000",
                        "weight": 1.5,
                        "fillOpacity": 0.7,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["tile_id", "coverage_pred", "n_pixels"],
                        aliases=["Tile ID", "Wheat Coverage", "Pixels Sampled"],
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
            df_display["coverage_pred"] = df_display["coverage_pred"].apply(
                lambda x: f"{x:.1%}"
            )
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        # TAB 4: STATISTICS
        with tab4:
            stats = _create_summary_statistics(filtered_rows)

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
                csv_data = "region,tile_id,coverage_pred,n_pixels\n"
                csv_data += "\n".join(
                    f"{r['region']},{r['tile_id']},{r['coverage_pred']:.4f},{r['n_pixels']}"
                    for r in filtered_rows
                )
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
    default_root = "C:/Users/user/Desktop/preprocessed_data"

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--root", default=default_root)
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
        class _Args:
            pass

        args = _Args()
        args.root = default_root
        args.year = "2020"
        args.months = [11, 12, 1, 2, 3, 4, 5, 6, 7]

    return AppConfig(
        root=str(args.root),
        year=str(args.year),
        months=tuple(int(m) for m in args.months),
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = parse_cli()
    main_streamlit(cfg)
