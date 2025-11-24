from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def calc_pr_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """Precision/recall/IoU over boolean vectors, returns counts for aggregation."""
    if pred.size == 0 or target.size == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": None, "recall": None, "iou": None}

    pred_b = pred.astype(bool)
    tgt_b = target.astype(bool)

    tp = int(np.logical_and(pred_b, tgt_b).sum())
    fp = int(np.logical_and(pred_b, ~tgt_b).sum())
    fn = int(np.logical_and(~pred_b, tgt_b).sum())
    tn = int(np.logical_and(~pred_b, ~tgt_b).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
    }


def aggregate_verification(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-tile verification metrics."""
    has_gt = [r for r in rows if r.get("coverage_gt") is not None]
    if not has_gt:
        return {
            "avg_gt": None,
            "avg_delta": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "iou": None,
        }

    avg_gt = float(np.mean([r["coverage_gt"] for r in has_gt]))
    avg_delta = float(np.mean([r["coverage_delta"] for r in has_gt]))

    tp = sum(r.get("tp", 0) for r in has_gt)
    fp = sum(r.get("fp", 0) for r in has_gt)
    fn = sum(r.get("fn", 0) for r in has_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else None
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None

    return {
        "avg_gt": avg_gt,
        "avg_delta": avg_delta,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def summary_stats(cover_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    coverages = np.array([r["coverage_pred"] for r in cover_rows])
    return {
        "mean": float(np.mean(coverages)),
        "median": float(np.median(coverages)),
        "std": float(np.std(coverages)),
        "min": float(np.min(coverages)),
        "max": float(np.max(coverages)),
        "q25": float(np.percentile(coverages, 25)),
        "q75": float(np.percentile(coverages, 75)),
    }
