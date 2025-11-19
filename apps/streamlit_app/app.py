from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Add the project root to Python path to import wheat_segmenter
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import joblib
import requests
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape, Polygon
from shapely.geometry.base import BaseGeometry

import rasterio
from rasterio.warp import transform_bounds

from wheat_segmenter import WheatTilesDataset

DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"C:\Users\Administrator\Desktop\preprocessed_data"))


@dataclass
class AppConfig:
    root: str
    year: str
    months: tuple[int, ...]


def _tile_bounds_latlon(month_paths: Dict[int, str]) -> Tuple[float, float, float, float] | None:
    for m, p in month_paths.items():
        try:
            with rasterio.open(p) as ds:
                b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            return b
        except Exception:
            continue
    return None


def load_tiles_index(ds: WheatTilesDataset) -> List[Dict[str, Any]]:
    idx: List[Dict[str, Any]] = []
    for rec in ds.index:
        bounds_ll = _tile_bounds_latlon(rec["month_paths"])  # (minx,miny,maxx,maxy) lon/lat
        if bounds_ll is None:
            continue
        idx.append({
            "region": rec["region"],
            "tile_id": rec["tile_id"],
            "bounds": bounds_ll,
        })
    return idx


def _bounds_to_polygon(b: Tuple[float, float, float, float]) -> Polygon:
    minx, miny, maxx, maxy = b
    return Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])


def _extract_features_all_valid(x_tb_hw: np.ndarray, valid_hw: np.ndarray):
    T, B, H, W = x_tb_hw.shape
    flat = x_tb_hw.reshape(T * B, H * W).T
    valid_idx = np.flatnonzero((valid_hw > 0.5).reshape(-1))
    return flat[valid_idx].astype(np.float32), valid_idx


def _parse_regions(text: str) -> List[str] | None:
    tokens = [tok.strip() for tok in text.replace(",", " ").split()]
    regions = [tok for tok in tokens if tok]
    return regions or None


def main_streamlit(app_cfg: AppConfig) -> None:
    st.set_page_config(page_title="Wheat Map (Lebanon)", layout="wide")
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

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("Model Settings")
    model_path = st.sidebar.text_input("Model .joblib path", value="runs/xgb_2020.joblib")
    prob_th = st.sidebar.slider("Probability threshold", 0.0, 1.0, 0.5, 0.05)
    pixels_cap = st.sidebar.number_input("Pixels cap per tile (0=all)", min_value=0, value=2000, step=500)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Settings")
    root = st.sidebar.text_input("Root (contains data/ and label/)", value=app_cfg.root)
    year = st.sidebar.text_input("Year", value=app_cfg.year)
    months_text = st.sidebar.text_input("Months (space-separated)", value=" ".join(map(str, app_cfg.months)))

    # Common months tuple for dataset/training requests
    if months_text.strip():
        try:
            months_sequence = tuple(int(m) for m in months_text.strip().split())
        except ValueError:
            st.sidebar.error("Invalid months input; falling back to defaults.")
            months_sequence = app_cfg.months
    else:
        months_sequence = app_cfg.months

    # Initialize training job state containers
    if "train_job_id" not in st.session_state:
        st.session_state["train_job_id"] = None
    if "train_job_status" not in st.session_state:
        st.session_state["train_job_status"] = None

    # Training controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Train Model via API")
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
        "Train fraction",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.3f",
    )
    test_fraction = st.sidebar.number_input(
        "Test fraction",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        format="%.2f",
    )
    pixels_per_tile = st.sidebar.number_input(
        "Pixels per tile",
        min_value=256,
        value=4096,
        step=512,
    )
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
            hgb_depth = st.number_input("max_depth (None for unlimited)", min_value=0, value=8, step=1, key="hgb_depth")
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
                resp = requests.post(f"{api_url_clean}/train", json=payload, timeout=15)
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
        if "results_map" in st.session_state:
            del st.session_state["results_map"]
        if "results_data" in st.session_state:
            del st.session_state["results_data"]

    # Load model
    if "model" not in st.session_state or load_clicked:
        if model_path:
            try:
                st.session_state["model"] = joblib.load(model_path)
                st.sidebar.success(f"✅ Model loaded: {Path(model_path).name}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load model: {e}")
                st.session_state["model"] = None

    # Load dataset
    if "dataset" not in st.session_state or load_clicked:
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
                        target_size=(64,64),
                        size_policy="pad",
                        probe_limit=12,
                    )
                st.session_state["dataset"] = ds
                st.session_state["tiles_index"] = load_tiles_index(ds)
                st.sidebar.success(f"✅ Loaded {len(ds)} tiles")
            except Exception as e:
                st.sidebar.error(f"❌ Dataset load failed: {e}")
                st.session_state["dataset"] = None
                st.session_state["tiles_index"] = []

    # Status indicators
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status")
    model_status = "✅ Loaded" if st.session_state.get("model") else "❌ Not loaded"
    dataset_status = "✅ Loaded" if st.session_state.get("dataset") else "❌ Not loaded"
    tiles_count = len(st.session_state.get("tiles_index", []))
    
    st.sidebar.markdown(f"""
    - **Model:** {model_status}
    - **Dataset:** {dataset_status}
    - **Tiles:** {tiles_count}
    """)

    # Main map for drawing
    st.subheader("📍 Select Region")
    
    # Create map with drawing tools
    m = folium.Map(
        location=[33.9, 35.9], 
        zoom_start=8, 
        control_scale=True,
        tiles="OpenStreetMap",
        zoom_control=True,
    )
    
    # Add draw control with multiple shape options
    draw = Draw(
        export=False,
        position="topleft",
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)

    # Show tile boundaries if loaded
    if "tiles_index" in st.session_state and st.session_state["tiles_index"]:
        tile_count = len(st.session_state["tiles_index"])
        st.info(f"📍 **{tile_count} data tiles loaded** - Draw your shape over these rectangles")
        for rec in st.session_state["tiles_index"]:
            poly = _bounds_to_polygon(rec["bounds"]).exterior.coords[:]
            folium.PolyLine(
                locations=[(y, x) for x, y in poly], 
                color="#0066FF", 
                weight=2, 
                opacity=0.6,
                tooltip=f"Tile: {rec['tile_id']} | Region: {rec['region']}"
            ).add_to(m)

    # Render map and capture drawings
    map_output = st_folium(
        m, 
        width=None, 
        height=500, 
        use_container_width=True, 
        returned_objects=["all_drawings"],
        key="main_map"
    )
    
    # Extract drawn geometry
    geom: BaseGeometry | None = None
    if map_output and map_output.get("all_drawings"):
        drawings = map_output["all_drawings"]
        if drawings:
            geom = shape(drawings[-1]["geometry"])
            st.info(f"✅ Region selected: {drawings[-1]['geometry']['type']}")

    # Inference button and results
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        run_clicked = st.button("🚀 Run Inference", type="primary", use_container_width=True, disabled=(geom is None))
    
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            if "results_data" in st.session_state:
                del st.session_state["results_data"]
            if "results_selected" in st.session_state:
                del st.session_state["results_selected"]
            if "results_tiles_idx" in st.session_state:
                del st.session_state["results_tiles_idx"]
            st.rerun()
    
    with col3:
        if geom:
            st.success("✅ Region selected - Ready to run inference")
        else:
            st.warning("👆 Draw a shape over the blue tile boundaries first")

    # Run inference
    if run_clicked and geom is not None:
        model = st.session_state.get("model")
        ds = st.session_state.get("dataset")
        tiles_idx = st.session_state.get("tiles_index", [])
        
        if model is None:
            st.error("❌ Please load a model first!")
        elif ds is None:
            st.error("❌ Please load dataset first!")
        else:
            # Find intersecting tiles
            selected: List[int] = []
            for i, rec in enumerate(tiles_idx):
                poly = _bounds_to_polygon(rec["bounds"])
                if geom.intersects(poly):
                    selected.append(i)
            
            if not selected:
                st.error("⚠️ **No data tiles found in selected region!**")
                st.info("""
                **Why this happens:**
                - The rectangles you see on the map are your actual satellite image tiles
                - You can ONLY analyze regions where tiles exist
                - Your dataset covers specific regions of Lebanon (not the entire country)
                
                **Solution:** Draw your shape over the visible tile boundaries on the map
                """)
            else:
                st.success(f"🔍 Processing {len(selected)} tiles...")
                
                cover_rows: List[Dict[str, Any]] = []
                prog = st.progress(0.0, text="Running inference...")
                cap = int(pixels_cap) if pixels_cap and pixels_cap > 0 else None
                
                for k, idx in enumerate(selected):
                    rec = tiles_idx[idx]
                    item = ds[idx]
                    x = item["x"].numpy()
                    valid = item["valid_mask"].numpy()[0] > 0.5
                    T, B, H, W = x.shape
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
                            proba = 1 / (1 + np.exp(-logits))
                        pred = (proba >= prob_th).astype(np.uint8)
                        cov = float(pred.mean())
                    
                    cover_rows.append({
                        "region": rec["region"],
                        "tile_id": rec["tile_id"],
                        "coverage_pred": round(cov, 4),
                        "n_pixels": len(flat),
                    })
                    prog.progress((k + 1) / len(selected), text=f"Processing tile {k+1}/{len(selected)}")
                
                prog.empty()
                
                # Store results in session state to persist them
                st.session_state["results_data"] = cover_rows
                st.session_state["results_selected"] = selected
                st.session_state["results_tiles_idx"] = tiles_idx

    # Display results if they exist (persists across reruns)
    if "results_data" in st.session_state and st.session_state["results_data"]:
        cover_rows = st.session_state["results_data"]
        selected = st.session_state["results_selected"]
        tiles_idx = st.session_state["results_tiles_idx"]
        
        # Create results map with colored tiles
        st.subheader("🗺️ Results: Wheat Coverage Visualization")
        
        results_map = folium.Map(
            location=[33.9, 35.9], 
            zoom_start=8, 
            control_scale=True,
            tiles="OpenStreetMap"
        )
        
        # Color function: Green = high wheat, Red = low wheat
        def cov_color(v: float) -> str:
            """Green for high coverage, yellow-orange for medium, red for low."""
            v = max(0.0, min(1.0, float(v)))
            if v > 0.7:
                # High coverage: bright green
                return "#00FF00"
            elif v > 0.5:
                # Medium-high: yellow-green
                return "#7FFF00"
            elif v > 0.3:
                # Medium: yellow
                return "#FFFF00"
            elif v > 0.1:
                # Low-medium: orange
                return "#FFA500"
            else:
                # Very low: red
                return "#FF0000"
        
        # Add colored tiles
        for rec, row in zip([tiles_idx[i] for i in selected], cover_rows):
            poly = _bounds_to_polygon(rec["bounds"])
            color = cov_color(row["coverage_pred"])
            
            folium.GeoJson(
                data={"type": "Feature", "geometry": poly.__geo_interface__, "properties": row},
                style_function=lambda feat, col=color: {
                    "fillColor": col, 
                    "color": "#000", 
                    "weight": 2, 
                    "fillOpacity": 0.7
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["tile_id", "coverage_pred", "n_pixels"], 
                    aliases=["Tile ID", "Wheat Coverage", "Pixels Sampled"],
                    localize=True
                ),
            ).add_to(results_map)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 180px; 
             background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px">
        <p style="margin:0; font-weight:bold;">🌾 Wheat Coverage Legend</p>
        <p style="margin:5px 0;"><span style="background-color:#00FF00; padding:2px 10px;">■</span> Very High (>70%)</p>
        <p style="margin:5px 0;"><span style="background-color:#7FFF00; padding:2px 10px;">■</span> High (50-70%)</p>
        <p style="margin:5px 0;"><span style="background-color:#FFFF00; padding:2px 10px;">■</span> Medium (30-50%)</p>
        <p style="margin:5px 0;"><span style="background-color:#FFA500; padding:2px 10px;">■</span> Low (10-30%)</p>
        <p style="margin:5px 0;"><span style="background-color:#FF0000; padding:2px 10px;">■</span> Very Low (<10%)</p>
        </div>
        '''
        results_map.get_root().html.add_child(folium.Element(legend_html))
        
        st_folium(results_map, width=None, height=600, use_container_width=True, key="results_map_display")
        
        # Show data table
        st.subheader("📋 Detailed Results")
        import pandas as pd
        df = pd.DataFrame(cover_rows)
        st.dataframe(df, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tiles", len(cover_rows))
        with col2:
            avg_cov = np.mean([r["coverage_pred"] for r in cover_rows])
            st.metric("Avg Coverage", f"{avg_cov:.2%}")
        with col3:
            max_cov = max([r["coverage_pred"] for r in cover_rows])
            st.metric("Max Coverage", f"{max_cov:.2%}")
        with col4:
            high_cov_count = sum(1 for r in cover_rows if r["coverage_pred"] > 0.5)
            st.metric("High Coverage Tiles", high_cov_count)
        
        # Download button
        csv_data = "region,tile_id,coverage_pred,n_pixels\n"
        csv_data += "\n".join([f"{r['region']},{r['tile_id']},{r['coverage_pred']:.4f},{r['n_pixels']}" for r in cover_rows])
        st.download_button(
            "📥 Download Results (CSV)", 
            data=csv_data, 
            file_name=f"wheat_coverage_{year}.csv", 
            mime="text/csv",
            use_container_width=True
        )


def parse_cli() -> AppConfig:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--year", default="2020")
    ap.add_argument("--months", nargs="*", type=int, default=[11,12,1,2,3,4,5,6,7])
    try:
        args, _ = ap.parse_known_args()
    except SystemExit:
        class _Args: pass
        args = _Args(); args.root = str(DEFAULT_DATA_ROOT); args.year = "2020"; args.months = [11,12,1,2,3,4,5,6,7]
    return AppConfig(root=str(args.root), year=str(args.year), months=tuple(int(m) for m in args.months))


if __name__ == "__main__":
    cfg = parse_cli()
    main_streamlit(cfg)
