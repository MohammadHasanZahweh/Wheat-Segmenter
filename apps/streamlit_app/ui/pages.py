from __future__ import annotations

import base64
import json
from typing import Any, Dict

import folium
import requests
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import shape
from streamlit_folium import st_folium


def render_training_jobs(api_url: str):
    has_status = st.session_state.get("train_job_status") is not None
    with st.expander("Training Jobs (API)", expanded=has_status):
        job_id = st.session_state.get("train_job_id")
        job_status = st.session_state.get("train_job_status")
        if job_id:
            st.markdown(f"**Current job ID:** `{job_id}`")
            if job_status:
                status = job_status.get("status", "unknown")

                if status == "completed":
                    st.success("Training completed.")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("F1 Score", f"{job_status.get('f1', 0):.4f}")
                    with col2:
                        st.metric("IoU", f"{job_status.get('iou', 0):.4f}")
                    with col3:
                        st.metric("Precision", f"{job_status.get('precision', 0):.4f}")
                    with col4:
                        st.metric("Recall", f"{job_status.get('recall', 0):.4f}")
                    st.json(job_status)
                elif status == "running":
                    st.info("Training in progress... Click 'Refresh Job Status' to update.")
                    if st.button("Auto-refresh every 10s", key="auto_refresh_btn"):
                        st.session_state["auto_refresh"] = True
                        st.rerun()
                    if st.session_state.get("auto_refresh"):
                        with st.spinner("Auto-refreshing..."):
                            import time

                            time.sleep(10)
                            try:
                                resp = requests.get(
                                    f"{api_url.rstrip('/')}/train/status",
                                    params={"id": job_id},
                                    timeout=10,
                                )
                                resp.raise_for_status()
                                updated_status = resp.json()
                                st.session_state["train_job_status"] = updated_status
                                if updated_status.get("status") != "running":
                                    st.session_state["auto_refresh"] = False
                                st.rerun()
                            except Exception:
                                pass
                elif status == "failed":
                    st.error(f"Training failed: {job_status.get('error', 'Unknown error')}")
                    st.json(job_status)
                else:
                    st.warning(f"Status: {status}")
                    st.json(job_status)
            else:
                st.info("No job status yet. Use the sidebar to refresh.")
        else:
            st.info("Configure and launch a training job from the sidebar to track it here.")


def render_welcome(app_cfg):
    st.markdown(
        """
        ### Welcome to the wheat segmentation dashboard

        The UI is now a thin client: it calls the FastAPI server for training and inference, while the server does the heavy lifting.

        #### What you can do here
        - Trigger model training jobs on the server and monitor their status.
        - Launch server-side inference runs for a given project/region/model.
        - Fetch rendered result images directly from the API.
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Set the API base URL in the sidebar, then configure inference parameters (project, region, model, output name, year) before starting jobs."
    )


def render_instructions():
    st.markdown(
        """
        ## Instructions
        1. Set the API base URL in the sidebar.
        2. Fill inference fields (project name, region, model file, result name, year).
        3. Click **Start Inference Job**; then use **Refresh Inference Status** and **Fetch Result Image**.
        4. To train, fill the training section in the sidebar and start a training job.
        5. Use the **Results** tab to view/download the latest fetched inference image.
        """
    )


def render_settings():
    defaults = st.session_state.get(
        "settings",
        {
            "root": "",
            "year": "2020",
            "months_text": "11 12 1 2 3 4 5 6 7",
            "use_meta_stats": True,
            "meta_dir": "./meta",
            "project_name": "wheat",
            "region_name": "region_0",
            "model_name": "xgb_2020.joblib",
            "save_name": "latest_run.tiff",
        },
    )
    st.subheader("Settings")
    root = st.text_input("Project root/name", value=defaults["root"])
    year = st.text_input("Default year", value=defaults["year"])
    months_text = st.text_input("Months (space-separated)", value=defaults["months_text"])
    project_name = st.text_input("Default project name", value=defaults["project_name"])
    region_name = st.text_input("Default region name", value=defaults["region_name"])
    model_name = st.text_input("Default model file (.joblib)", value=defaults["model_name"])
    save_name = st.text_input("Default result name (.tiff)", value=defaults["save_name"])

    use_meta_stats = st.checkbox("Use meta stats normalization", value=bool(defaults.get("use_meta_stats", True)))
    meta_dir = st.text_input("Meta stats directory", value=defaults.get("meta_dir", "./meta"))

    if st.button("Save Settings", type="primary"):
        st.session_state["settings"] = {
            "root": root,
            "year": year,
            "months_text": months_text,
            "use_meta_stats": use_meta_stats,
            "meta_dir": meta_dir,
            "project_name": project_name,
            "region_name": region_name,
            "model_name": model_name,
            "save_name": save_name,
        }
        st.success("Settings saved. Values will prefill the sidebar.")


def render_config_select(sidebar_cfg):
    st.subheader("Step 2 - Select an area on the map (for your reference)")
    st.caption(
        "Draw a polygon/rectangle below. The current server API uses the `region_name` directory you provide; "
        "the geometry is stored in the session for reference until the API accepts polygons."
    )

    m = folium.Map(location=[33.9, 35.9], zoom_start=8, control_scale=True, tiles="CartoDB Positron")
    draw = Draw(
        export=False,
        position="topleft",
        draw_options={"polyline": False, "rectangle": True, "polygon": True, "circle": False, "marker": False, "circlemarker": False},
        edit_options={"edit": False},
    )
    draw.add_to(m)

    map_output = st_folium(
        m,
        width=None,
        height=520,
        use_container_width=True,
        returned_objects=["all_drawings"],
        key="inference_map",
    )

    if map_output and map_output.get("all_drawings"):
        drawings = map_output["all_drawings"]
        if drawings:
            st.session_state["inference_geometries"] = [d["geometry"] for d in drawings]
            geom = shape(drawings[-1]["geometry"])
            st.success(f"Captured {len(drawings)} geometries. Latest: {drawings[-1]['geometry']['type']}")
            if geom.is_valid and geom.area:
                st.caption(f"Latest approx area (degrees^2): {geom.area:.4f}")
    else:
        st.info("No geometry selected yet.")
        st.session_state.pop("inference_geometries", None)

    request_template: Dict[str, Any] = st.session_state.get("inference_request") or {
        "project_name": sidebar_cfg.project_name,
        "region_name": sidebar_cfg.region_name,
        "model_name": sidebar_cfg.model_name,
        "year": sidebar_cfg.inference_year,
        "save_name": sidebar_cfg.save_name,
    }
    if st.session_state.get("inference_geometries"):
        request_template["geometries"] = st.session_state["inference_geometries"]
    st.markdown("**Current inference request template:**")
    st.code(json.dumps(request_template, indent=2), language="json")

    job_id = st.session_state.get("inference_job_id")
    status = st.session_state.get("inference_status")
    if job_id:
        st.markdown(f"**Job ID:** `{job_id}`")
    if status:
        st.markdown("**Latest status:**")
        st.json(status)
    else:
        st.info("No inference job started yet.")


def render_results(sidebar_cfg):
    st.subheader("Results")
    job_id = st.session_state.get("inference_job_id")
    status = st.session_state.get("inference_status")
    if job_id:
        st.markdown(f"**Job ID:** `{job_id}`")
    if status:
        st.markdown("**Latest status:**")
        st.json(status)

    img_b64 = st.session_state.get("inference_result_b64")
    run_name = st.session_state.get("inference_last_result_name", sidebar_cfg.save_name)
    req = st.session_state.get("inference_request") or {}
    project = req.get("project_name", sidebar_cfg.project_name)

    if not img_b64:
        st.info("No result image fetched yet. Use 'Fetch Result Image' from the sidebar after the job finishes.")
        return

    try:
        image_bytes = base64.b64decode(img_b64)
    except Exception as exc:  # pragma: no cover - display error to user
        st.error(f"Failed to decode image: {exc}")
        return

    st.markdown(f"**Project:** {project} | **Result file:** {run_name}")
    st.image(image_bytes, caption="Inference result", use_column_width=True)
    st.download_button(
        "Download result image",
        data=image_bytes,
        file_name=run_name or "result.tiff",
        mime="image/tiff",
        use_container_width=True,
    )
