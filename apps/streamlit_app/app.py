from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import streamlit as st

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.streamlit_app.core.api_client import TrainAPI
from apps.streamlit_app.core.loaders import load_dataset, load_model
from apps.streamlit_app.ui.pages import (
    render_config_select,
    render_instructions,
    render_settings,
    render_results,
    render_training_jobs,
    render_welcome,
)
from apps.streamlit_app.ui.sidebar import SidebarConfig, render_sidebar
from apps.streamlit_app.ui.styles import inject_global_styles


DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"C:\\Users\\Administrator\\Desktop\\preprocessed_data"))


@dataclass
class AppConfig:
    root: str
    year: str
    months: tuple[int, ...]


def handle_training_actions(api_url: str, sidebar_cfg: SidebarConfig):
    training_feedback = st.sidebar.empty()
    client = TrainAPI(api_url)

    if sidebar_cfg.start_training:
        if not client.base:
            training_feedback.error("Provide a valid API base URL.")
        elif not sidebar_cfg.root or not sidebar_cfg.year:
            training_feedback.error("Root and year are required to submit a training job.")
        else:
            payload = {
                "job_name": sidebar_cfg.job_name.strip() or f"streamlit_job_{int(time.time())}",
                "algorithm": sidebar_cfg.algorithm,
                "dataset": sidebar_cfg.dataset_cfg,
                "save_model": bool(sidebar_cfg.save_model),
                "output_path": sidebar_cfg.output_path.strip() or None,
                "model_params": sidebar_cfg.model_params or None,
            }
            try:
                data = client.start_job(payload)
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

    if sidebar_cfg.refresh_training and st.session_state.get("train_job_id"):
        try:
            status_data = client.status(st.session_state["train_job_id"])
            st.session_state["train_job_status"] = status_data
            training_feedback.info(f"Status: {status_data.get('status', 'unknown')}")
        except requests.RequestException as exc:
            training_feedback.error(f"Failed to fetch status: {exc}")

    if sidebar_cfg.clear_training and st.session_state.get("train_job_id"):
        st.session_state["train_job_id"] = None
        st.session_state["train_job_status"] = None
        training_feedback.info("Cleared stored job info.")


def main_streamlit(app_cfg: AppConfig) -> None:
    st.set_page_config(page_title="Wheat Map (Lebanon)", layout="wide")
    inject_global_styles()
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span style="font-size: 32px;">🌾</span>
            <h1 style="margin:0;">Wheat Coverage Map (Lebanon)</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = ["Welcome", "Instructions", "Settings", "Configure & Select Region", "Results & Analysis"]
    if "nav" not in st.session_state:
        st.session_state["nav"] = nav_options[0]

    st.markdown(
        """
        <style>
        .nav-row button[kind=\"secondary\"] {
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(115,161,255,0.3);
            color: #e2e8f0;
        }
        .nav-row button[kind=\"primary\"] {
            border-radius: 999px;
            background: linear-gradient(120deg, #74f1ff, #5ee6a0);
            color: #0a1526;
            border: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    nav_cols = st.columns(len(nav_options), gap="small")
    with st.container():
        for col, label in zip(nav_cols, nav_options):
            with col:
                active = st.session_state["nav"] == label
                if st.button(
                    label,
                    type="primary" if active else "secondary",
                    use_container_width=True,
                    key=f"nav_{label}",
                ):
                    st.session_state["nav"] = label
                    st.rerun()
    nav = st.session_state["nav"]

    sidebar_cfg = render_sidebar(app_cfg)
    handle_training_actions(sidebar_cfg.api_url, sidebar_cfg)
    render_training_jobs(sidebar_cfg.api_url)

    if "model" not in st.session_state or sidebar_cfg.load_clicked:
        st.session_state["model"] = load_model(sidebar_cfg.model_path)

    if "dataset" not in st.session_state or sidebar_cfg.load_clicked:
        ds, tiles_index = load_dataset(sidebar_cfg.root, sidebar_cfg.year, sidebar_cfg.months_sequence)
        st.session_state["dataset"] = ds
        st.session_state["tiles_index"] = tiles_index

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">?? Status</div>', unsafe_allow_html=True)
    model_status = "? Loaded" if st.session_state.get("model") is not None else "? Not loaded"
    dataset_status = "? Loaded" if st.session_state.get("dataset") is not None else "? Not loaded"
    tiles_count = len(st.session_state.get("tiles_index", []))
    st.sidebar.markdown(
        f"""
        - **Model:** {model_status}  
        - **Dataset:** {dataset_status}  
        - **Tiles with bounds:** {tiles_count}
        """
    )

    if nav == "Welcome":
        render_welcome(app_cfg)
        return
    if nav == "Instructions":
        render_instructions()
        return
    if nav == "Settings":
        render_settings()
        return
    if nav == "Configure & Select Region":
        render_config_select(
            sidebar_cfg=sidebar_cfg,
            model=st.session_state.get("model"),
            ds=st.session_state.get("dataset"),
            tiles_idx=st.session_state.get("tiles_index", []),
        )
        return
    if nav == "Results & Analysis":
        cover_rows = st.session_state.get("results_data")
        if not cover_rows:
            st.warning("No results yet. Run inference in **2?? Configure & Select Region** first.")
            return
        render_results(
            sidebar_cfg=sidebar_cfg,
            cover_rows=cover_rows,
            tiles_idx=st.session_state.get("results_tiles_idx", []),
            year=sidebar_cfg.year,
        )


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
        class _Args:
            pass

        args = _Args()
        args.root = str(DEFAULT_DATA_ROOT)
        args.year = "2020"
        args.months = [11, 12, 1, 2, 3, 4, 5, 6, 7]
    return AppConfig(root=str(args.root), year=str(args.year), months=tuple(int(m) for m in args.months))


if __name__ == "__main__":
    cfg = parse_cli()
    main_streamlit(cfg)
