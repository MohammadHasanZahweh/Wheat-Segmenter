from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from apps.streamlit_app.core.geo import bounds_to_polygon
from apps.streamlit_app.core.metrics import summary_stats


def export_geojson(cover_rows: List[Dict[str, Any]], tiles_idx: List[Dict[str, Any]]) -> str:
    features = []
    for row in cover_rows:
        rec = tiles_idx[row["tile_index"]]
        poly = bounds_to_polygon(rec["bounds"])
        features.append(
            {
                "type": "Feature",
                "geometry": poly.__geo_interface__,
                "properties": {
                    "tile_id": row["tile_id"],
                    "region": row["region"],
                    "wheat_coverage": row["coverage_pred"],
                    "wheat_coverage_gt": row.get("coverage_gt"),
                    "coverage_delta": row.get("coverage_delta"),
                    "pixels_analyzed": row["n_pixels"],
                    "n_valid_total": row.get("n_valid_total"),
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "iou": row.get("iou"),
                    "has_gt": row.get("has_gt"),
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)


def export_csv(cover_rows: List[Dict[str, Any]], year: str) -> tuple[str, str]:
    fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.4f}"
    csv_headers = [
        "region",
        "tile_id",
        "coverage_pred",
        "coverage_gt",
        "coverage_delta",
        "precision",
        "recall",
        "iou",
        "n_pixels",
        "n_valid_total",
        "sample_rate",
        "tile_index",
    ]
    csv_data = ",".join(csv_headers) + "\n"
    csv_rows = []
    for r in cover_rows:
        csv_rows.append(
            ",".join(
                [
                    str(r["region"]),
                    str(r["tile_id"]),
                    fmt(r.get("coverage_pred")),
                    fmt(r.get("coverage_gt")),
                    fmt(r.get("coverage_delta")),
                    fmt(r.get("precision")),
                    fmt(r.get("recall")),
                    fmt(r.get("iou")),
                    str(r.get("n_pixels", "")),
                    str(r.get("n_valid_total", "")),
                    fmt(r.get("sample_rate")),
                    str(r.get("tile_index")),
                ]
            )
        )
    csv_data += "\n".join(csv_rows)
    return csv_data, f"wheat_coverage_{year}.csv"


def export_json(cover_rows: List[Dict[str, Any]], year: str, verification: Dict[str, Any], filters: Dict[str, Any]) -> str:
    stats_all = summary_stats(cover_rows)
    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "year": year,
            "tiles_count": len(cover_rows),
            "statistics": stats_all,
            "verification": verification,
            "filters": filters,
        },
        "results": cover_rows,
    }
    return json.dumps(payload, indent=2)


def summary_report_md(
    cover_rows: List[Dict[str, Any]],
    year: str,
    verification: Dict[str, Any],
    selected_regions: List[str],
    cov_min: float,
    cov_max: float,
) -> str:
    stats_all = summary_stats(cover_rows)
    regions_md = "\n".join(f"- {reg}" for reg in sorted({r['region'] for r in cover_rows}))
    high_cov_count_filtered = sum(1 for r in cover_rows if r["coverage_pred"] > 0.5)
    high_percentage = 100 * high_cov_count_filtered / len(cover_rows)
    fmt_pct = lambda v: f"{v:.2%}" if v is not None else "—"
    fmt_delta = lambda v: f"{v:+.2%}" if v is not None else "—"

    summary_text = f"""# Wheat Coverage Analysis Report
**Year:** {year}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Filters
- Regions: {', '.join(selected_regions)}
- Coverage range: {cov_min:.2f} – {cov_max:.2f}

## Overview
- **Total Tiles Analyzed:** {len(cover_rows)}
- **Average Coverage:** {stats_all['mean']:.2%}
- **Coverage Range:** {stats_all['min']:.2%} – {stats_all['max']:.2%}
- **High Coverage Tiles (>50%):** {high_cov_count_filtered} ({high_percentage:.1f}%)

## Verification (micro)
- **Ground Truth Avg Coverage:** {fmt_pct(verification.get('avg_gt'))}
- **Avg Delta (Pred - GT):** {fmt_delta(verification.get('avg_delta'))}
- **Precision / Recall:** {fmt_pct(verification.get('precision'))} / {fmt_pct(verification.get('recall'))}
- **IoU:** {fmt_pct(verification.get('iou'))}
- **F1:** {fmt_pct(verification.get('f1'))}

## Statistics
- **Median:** {stats_all['median']:.2%}
- **Std Dev:** {stats_all['std']:.2%}
- **Q1 (25th percentile):** {stats_all['q25']:.2%}
- **Q3 (75th percentile):** {stats_all['q75']:.2%}

## Regions Covered (after filters)
{regions_md}
"""
    return summary_text
