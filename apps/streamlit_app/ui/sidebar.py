from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import streamlit as st
import os


@dataclass
class SidebarConfig:
    api_url: str
    project_name: str
    model_name: str
    save_name: str
    inference_year: int
    start_inference: bool
    refresh_inference: bool
    fetch_result: bool


def render_sidebar(app_cfg) -> SidebarConfig:
    st.sidebar.header("Configuration")

    settings = st.session_state.get(
        "settings",
        {
            "project_name": app_cfg.root,
            "model_name": "xgb_2020.joblib",
            "save_name": "latest_run.tiff",
            "year": app_cfg.year,
        },
    )
    year = settings.get("year", app_cfg.year)
    project_name = settings.get("project_name", app_cfg.root)
    model_name = settings.get("model_name", "xgb_2020.joblib")
    save_name = settings.get("save_name", "latest_run.tiff")

    if "train_job_id" not in st.session_state:
        st.session_state["train_job_id"] = None
    if "train_job_status" not in st.session_state:
        st.session_state["train_job_status"] = None
    if "inference_job_id" not in st.session_state:
        st.session_state["inference_job_id"] = None
    if "inference_status" not in st.session_state:
        st.session_state["inference_status"] = None
    if "inference_result_b64" not in st.session_state:
        st.session_state["inference_result_b64"] = None

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">API</div>', unsafe_allow_html=True)
    api_url = st.sidebar.text_input("API_base_URL", value=os.getenv("API_URL","http://127.0.0.1:8000"))

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">Inference (server)</div>', unsafe_allow_html=True)
    project_name_input = st.sidebar.text_input("Project name", value=project_name)
    model_name_input = st.sidebar.text_input("Model (.joblib)", value=model_name)
    save_name_input = st.sidebar.text_input("Result name (.tiff)", value=save_name)
    inference_year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=int(year), step=1)
    start_inference = st.sidebar.button("Start Inference Job", use_container_width=True)
    refresh_inference = st.sidebar.button(
        "Refresh Inference Status",
        use_container_width=True,
        disabled=not bool(st.session_state.get("inference_job_id")),
    )
    fetch_result = st.sidebar.button("Fetch Result Image", use_container_width=True)

    return SidebarConfig(
        api_url=api_url.rstrip("/"),
        project_name=project_name_input,
        model_name=model_name_input,
        save_name=save_name_input,
        inference_year=int(inference_year),
        start_inference=start_inference,
        refresh_inference=refresh_inference,
        fetch_result=fetch_result,
    )
