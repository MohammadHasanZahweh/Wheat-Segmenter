from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from folium.plugins import Draw
from shapely.geometry import shape
from streamlit_folium import st_folium

from apps.streamlit_app.core.charts import (
    coverage_by_region_chart,
    coverage_distribution_chart,
    tile_ranking_chart,
)
from apps.streamlit_app.core.exports import export_csv, export_geojson, export_json, summary_report_md
from apps.streamlit_app.core.geo import bounds_to_polygon, cov_color
from apps.streamlit_app.core.inference import run_inference
from apps.streamlit_app.core.metrics import aggregate_verification, summary_stats
from apps.streamlit_app.ui.styles import legend_html


def render_training_jobs(api_url: str):
    has_status = st.session_state.get("train_job_status") is not None
    with st.expander("🧠 Training Jobs (API)", expanded=has_status):
        job_id = st.session_state.get("train_job_id")
        job_status = st.session_state.get("train_job_status")
        if job_id:
            st.markdown(f"**Current job ID:** `{job_id}`")
            if job_status:
                status = job_status.get("status", "unknown")

                if status == "completed":
                    st.success("✅ Training completed!")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("F1 Score", f"{job_status.get('f1', 0):.4f}")
                    with col2:
                        st.metric("IoU", f"{job_status.get('iou', 0):.4f}")
                    with col3:
                        st.metric("Precision", f"{job_status.get('precision', 0):.4f}")
                    with col4:
                        st.metric("Recall", f"{job_status.get('recall', 0):.4f}")

                    with st.expander("📊 Full Results", expanded=False):
                        st.json(job_status)

                elif status == "running":
                    st.info("⏳ Training in progress... Click 'Refresh Job Status' to update.")
                    if st.button("🔄 Auto-refresh every 10s", key="auto_refresh_btn"):
                        st.session_state["auto_refresh"] = True
                        st.rerun()

                    if st.session_state.get("auto_refresh"):
                        with st.spinner("Auto-refreshing..."):
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
                    st.error(f"❌ Training failed: {job_status.get('error', 'Unknown error')}")
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


def render_config_select(sidebar_cfg, model, ds, tiles_idx):
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
    if tiles_idx:
        tile_count = len(tiles_idx)
        st.info(f"📍 **{tile_count} data tiles loaded** – draw your shape over these rectangles.")
        for rec in tiles_idx:
            poly = bounds_to_polygon(rec["bounds"]).exterior.coords[:]
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

    geom = None
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

    if run_clicked and geom is not None:
        if model is None:
            st.error("❌ Please load a model first from the sidebar.")
        elif ds is None:
            st.error("❌ Please load a dataset first from the sidebar.")
        else:
            cover_rows, selected = run_inference(
                geom=geom,
                ds=ds,
                model=model,
                tiles_idx=tiles_idx,
                prob_th=sidebar_cfg.prob_th,
                pixels_cap=sidebar_cfg.pixels_cap,
            )
            if cover_rows:
                st.session_state["results_data"] = cover_rows
                st.session_state["results_selected"] = selected
                st.session_state["results_tiles_idx"] = tiles_idx


def render_results(sidebar_cfg, cover_rows, tiles_idx, year, selected):
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
    verification_summary = aggregate_verification(filtered_rows)
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

        rows_for_map = filtered_rows if apply_filter_to_map else cover_rows
        idx_for_map = [tiles_idx[r["tile_index"]] for r in rows_for_map]

        for rec, row in zip(idx_for_map, rows_for_map):
            poly = bounds_to_polygon(rec["bounds"])
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

        results_map.get_root().html.add_child(folium.Element(legend_html()))

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
                coverage_distribution_chart(filtered_rows),
                use_container_width=True,
            )

        with col_region:
            st.plotly_chart(
                coverage_by_region_chart(filtered_rows),
                use_container_width=True,
            )

        st.markdown("---")
        st.plotly_chart(
            tile_ranking_chart(filtered_rows),
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
            "has_gt",
            "tile_index",
        ):
            if missing_col not in df_display:
                df_display[missing_col] = np.nan
        df_display["coverage_pred"] = df_display["coverage_pred"].apply(lambda x: f"{x:.1%}")
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
            "tile_index",
            "coverage_pred",
            "coverage_gt",
            "coverage_delta",
            "precision",
            "recall",
            "iou",
            "n_pixels",
            "n_valid_total",
            "sample_rate",
            "has_gt",
        ]
        df_display = df_display[[c for c in ordered_cols if c in df_display.columns]]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # TAB 4: STATISTICS
    with tab4:
        stats = summary_stats(filtered_rows)
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
        filters = {"regions": selected_regions, "coverage_min": cov_min, "coverage_max": cov_max}
        csv_data, csv_name = export_csv(filtered_rows, year)
        geojson_data = export_geojson(filtered_rows, tiles_idx)
        json_data = export_json(filtered_rows, year, verification_summary, filters)

        col_csv, col_geojson, col_json = st.columns(3)
        with col_csv:
            st.download_button(
                "📥 CSV",
                data=csv_data,
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
            )

        with col_geojson:
            st.download_button(
                "🗺️ GeoJSON",
                data=geojson_data,
                file_name=f"wheat_coverage_{year}.geojson",
                mime="application/json",
                use_container_width=True,
            )

        with col_json:
            st.download_button(
                "📄 JSON",
                data=json_data,
                file_name=f"wheat_coverage_{year}_full.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")

        st.subheader("📄 Summary Report (after filters)")
        summary_text = summary_report_md(
            cover_rows=filtered_rows,
            year=year,
            verification=verification_summary,
            selected_regions=selected_regions,
            cov_min=cov_min,
            cov_max=cov_max,
        )
        st.markdown(summary_text)
        st.download_button(
            "📥 Download Report (Markdown)",
            data=summary_text,
            file_name=f"wheat_coverage_{year}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
