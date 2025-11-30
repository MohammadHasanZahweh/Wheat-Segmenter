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
from apps.streamlit_app.ui.pages import render_config_select, render_instructions, render_results, render_welcome
from apps.streamlit_app.ui.sidebar import SidebarConfig, render_sidebar
from apps.streamlit_app.ui.styles import inject_global_styles


DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"wheat"))


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


def handle_inference_actions(api_url: str, sidebar_cfg: SidebarConfig):
    inference_feedback = st.sidebar.empty()
    client = TrainAPI(api_url)

    project_name = "Wheat"
    region_name = "region_0"
    job_id = st.session_state.get("inference_job_id")

    base_request = {
        "project_name": project_name,
        "region_name": region_name,
        "model_name": sidebar_cfg.model_name,
        "year": int(sidebar_cfg.inference_year),
        "save_name": sidebar_cfg.save_name,
    }
    if st.session_state.get("inference_geometry"):
        base_request["geometry"] = st.session_state["inference_geometry"]

    if sidebar_cfg.start_inference:
        if not client.base:
            inference_feedback.error("Provide a valid API base URL.")
        elif not sidebar_cfg.model_name:
            inference_feedback.error("Model name is required.")
        elif "geometry" not in base_request:
            inference_feedback.error("Draw an area on the map before starting inference.")
        else:
            try:
                data = client.start_inference(base_request)
                job_id = data.get("job_id")
                if not job_id:
                    raise ValueError(f"Unexpected response: {data}")
                st.session_state["inference_job_id"] = job_id
                st.session_state["inference_request"] = base_request
                st.session_state["inference_status"] = {"status": data.get("status", "running"), **data}
                st.session_state["inference_result_b64"] = None
                st.session_state["inference_last_result_name"] = base_request["save_name"]
                inference_feedback.success(f"Inference job started with id {job_id}")
            except requests.RequestException as exc:
                inference_feedback.error(f"Failed to start inference: {exc}")
            except Exception as exc:
                inference_feedback.error(f"Unexpected error: {exc}")

    if sidebar_cfg.refresh_inference and st.session_state.get("inference_job_id"):
        try:
            status_data = client.inference_status(st.session_state["inference_job_id"])
            st.session_state["inference_status"] = status_data
            status_label = status_data.get("status", "unknown")
            detail = status_data.get("output_exists")
            suffix = f" | output ready: {detail}" if detail is not None else ""
            inference_feedback.info(f"Inference status: {status_label}{suffix}")
        except requests.RequestException as exc:
            inference_feedback.error(f"Failed to fetch inference status: {exc}")

    if sidebar_cfg.fetch_result:
        req = st.session_state.get("inference_request") or base_request
        project = req.get("project_name") or project_name
        run_name = req.get("save_name") or sidebar_cfg.save_name
        status_data = st.session_state.get("inference_status") or {}
        if job_id and status_data.get("status") and status_data["status"] != "completed":
            inference_feedback.warning("Inference is not completed yet. Refresh status and try again once finished.")
            return
        if status_data.get("output_name"):
            run_name = status_data["output_name"]
        if status_data.get("project"):
            project = status_data["project"]
        if not run_name:
            inference_feedback.error("Result name is required to fetch results.")
        else:
            try:
                result = client.fetch_result(project, run_name, job_id=job_id)
                st.session_state["inference_status"] = {**status_data, "status": result.get("status")}
                image_b64 = result.get("image_base64")
                if image_b64:
                    st.session_state["inference_result_b64"] = image_b64
                    st.session_state["inference_last_result_name"] = result.get("run", run_name)
                    inference_feedback.success(f"Fetched result for {project}/{st.session_state['inference_last_result_name']}")
                else:
                    inference_feedback.warning(result.get("status", "Result not ready yet"))
            except requests.RequestException as exc:
                inference_feedback.error(f"Failed to fetch result image: {exc}")


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

    nav_options = ["Welcome", "Instructions", "Inference", "Results"]
    if "nav" not in st.session_state:
        st.session_state["nav"] = nav_options[0]

    st.markdown(
        """
        <style>
        .nav-row button[kind=\"secondary\"] {
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,99,132,0.35);
            color: #e2e8f0;
        }
        .nav-row button[kind=\"primary\"] {
            border-radius: 999px;
            background: linear-gradient(120deg, #ff5f6d, #ffc371);
            color: #0a0f1f;
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
    handle_inference_actions(sidebar_cfg.api_url, sidebar_cfg)

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">Status</div>', unsafe_allow_html=True)
    inference_job = st.session_state.get("inference_status") or {}
    inference_status = inference_job.get("status", "not started")
    st.sidebar.markdown(
        f"""
        - **Inference job:** {inference_status}
        """
    )

    if nav == "Welcome":
        render_welcome(app_cfg)
        return
    if nav == "Instructions":
        render_instructions()
        return
    if nav == "Inference":
        render_config_select(sidebar_cfg=sidebar_cfg)
        return
    if nav == "Results":
        render_results(sidebar_cfg=sidebar_cfg)


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
