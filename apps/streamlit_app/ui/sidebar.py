from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import streamlit as st

from apps.streamlit_app.core.geo import parse_regions


@dataclass
class SidebarConfig:
    model_path: str
    prob_th: float
    pixels_cap: int
    root: str
    year: str
    months_sequence: Tuple[int, ...]
    api_url: str
    job_name: str
    algorithm: str
    model_params: Dict[str, Any]
    dataset_cfg: Dict[str, Any]
    use_meta_stats: bool
    meta_dir: str
    save_model: bool
    output_path: str
    start_training: bool
    refresh_training: bool
    clear_training: bool
    load_clicked: bool
    balance_pixels: bool
    pixels_per_tile: int
    train_fraction: float
    test_fraction: float


def render_sidebar(app_cfg) -> SidebarConfig:
    st.sidebar.header("⚙️ Configuration")

    # Pull settings from session (set in Settings page)
    settings = st.session_state.get(
        "settings",
        {
            "model_path": "runs/xgb_2020.joblib",
            "prob_th": 0.5,
            "pixels_cap": 2000,
            "root": app_cfg.root,
            "year": app_cfg.year,
            "months_text": " ".join(map(str, app_cfg.months)),
            "use_meta_stats": True,
            "meta_dir": "./meta",
        },
    )
    model_path = settings.get("model_path", "")
    prob_th = float(settings.get("prob_th", 0.5))
    pixels_cap = int(settings.get("pixels_cap", 2000))
    root = settings.get("root", app_cfg.root)
    year = settings.get("year", app_cfg.year)
    months_text = settings.get("months_text", " ".join(map(str, app_cfg.months)))
    use_meta_stats = bool(settings.get("use_meta_stats", True))
    meta_dir = settings.get("meta_dir", "./meta")

    if months_text.strip():
        try:
            months_sequence = tuple(int(m) for m in months_text.strip().split())
        except ValueError:
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
    load_clicked = st.sidebar.button("🔄 Load Dataset & Model", type="primary", use_container_width=True)
    if load_clicked:
        for key in ("results_data", "results_selected", "results_tiles_idx"):
            st.session_state.pop(key, None)

    dataset_cfg = {
        "root": root,
        "year": year,
        "regions": parse_regions(regions_text),
        "months": list(months_sequence),
        "train_fraction": float(train_fraction),
        "test_fraction": float(test_fraction),
        "pixels_per_tile": int(pixels_per_tile),
        "balance_pixels": bool(balance_pixels),
        "seed": int(seed),
    }
    if use_meta_stats:
        dataset_cfg["use_meta_stats"] = True
        dataset_cfg["meta_dir"] = meta_dir.strip() or "./meta"

    return SidebarConfig(
        model_path=model_path,
        prob_th=prob_th,
        pixels_cap=int(pixels_cap),
        root=root,
        year=year,
        months_sequence=months_sequence,
        api_url=api_url.rstrip("/"),
        job_name=job_name,
        algorithm=algorithm,
        model_params=model_params,
        dataset_cfg=dataset_cfg,
        use_meta_stats=use_meta_stats,
        meta_dir=meta_dir,
        save_model=bool(save_model),
        output_path=output_path,
        start_training=start_training,
        refresh_training=refresh_training,
        clear_training=clear_training,
        load_clicked=load_clicked,
        balance_pixels=balance_pixels,
        pixels_per_tile=int(pixels_per_tile),
        train_fraction=float(train_fraction),
        test_fraction=float(test_fraction),
    )
