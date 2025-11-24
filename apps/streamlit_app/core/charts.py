from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go


def coverage_distribution_chart(cover_rows: List[Dict[str, Any]]) -> go.Figure:
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


def coverage_by_region_chart(cover_rows: List[Dict[str, Any]]) -> go.Figure:
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


def tile_ranking_chart(cover_rows: List[Dict[str, Any]]) -> go.Figure:
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
