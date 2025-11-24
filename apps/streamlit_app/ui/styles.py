from __future__ import annotations

import streamlit as st


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
        header[data-testid="stHeader"], .stToolbar {
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def legend_html() -> str:
    """Inline legend for folium map."""
    return """
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
