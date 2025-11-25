from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import joblib
import streamlit as st

from apps.streamlit_app.core.geo import load_tiles_index
from server.dataset.PatchDataset import WheatTilesDataset


@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_dataset_cached(root: str, year: str, months_sequence: Tuple[int, ...]):
    return WheatTilesDataset(
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


def load_model(path: str):
    model = None
    if not path:
        st.sidebar.error("❌ Provide a model path.")
        return None
    try:
        model = load_model_cached(path)
        st.sidebar.success(f"✅ Model loaded: {Path(path).name}")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")
    return model


def load_dataset(root: str, year: str, months_sequence: Tuple[int, ...]):
    ds = None
    tiles_index = []
    if not root or not year:
        st.sidebar.error("❌ Root and year are required.")
        return ds, tiles_index

    try:
        with st.spinner("Loading dataset..."):
            ds = load_dataset_cached(root, year, months_sequence)
            tiles_index = load_tiles_index(ds)
            st.sidebar.success(f"✅ Loaded {len(ds)} tiles")
    except Exception as e:
        st.sidebar.error(f"❌ Dataset load failed: {e}")
        ds = None
        tiles_index = []

    return ds, tiles_index
