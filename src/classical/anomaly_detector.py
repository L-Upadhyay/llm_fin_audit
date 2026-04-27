"""
Earnings anomaly detector.

Three classical signals over a quarterly EPS series:
  - statistical anomaly detection (>2 standard deviations from the mean)
  - linear search for the single worst quarter
  - trend classification (improving / declining / stable)

Inputs are lists of EPS values ordered newest-first (index 0 = most recent
quarter, last index = oldest), matching the loader's output convention.
"""

import numpy as np


def detect_earnings_anomaly(earnings_history):
    """
    Flag any quarter whose EPS lies more than 2 standard deviations from
    the historical mean.

    Returns:
        {
          "mean":          mean EPS,
          "std":           sample standard deviation (ddof=1),
          "anomalies":     [{"index": i, "eps": v, "z_score": z}, ...],
          "worst_quarter": the anomaly with the largest |z|, or None,
          "severity":      "none" | "moderate" | "severe",
          "summary":       plain-English explanation,
        }

    Severity rules:
      - "none"     : no quarter exceeds 2 std deviations
      - "moderate" : at least one |z| > 2 but all < 3
      - "severe"   : at least one |z| >= 3 (looks like a different distribution)
    """
    n = len(earnings_history)
    if n < 2:
        return {
            "mean": None,
            "std": None,
            "anomalies": [],
            "worst_quarter": None,
            "severity": "none",
            "summary": "Not enough quarters to compute anomalies.",
        }

    arr = np.asarray(earnings_history, dtype=float)
    mean = float(arr.mean())
    # Sample standard deviation (ddof=1) — more conservative for small N
    # than the population std.
    std = float(arr.std(ddof=1))

    anomalies = []
    if std > 0:
        for i, v in enumerate(arr):
            z = (float(v) - mean) / std
            if abs(z) > 2.0:
                anomalies.append({
                    "index": i,
                    "eps": float(v),
                    "z_score": float(z),
                })

    worst = max(anomalies, key=lambda a: abs(a["z_score"])) if anomalies else None

    if not anomalies:
        severity = "none"
    elif any(abs(a["z_score"]) >= 3.0 for a in anomalies):
        severity = "severe"
    else:
        severity = "moderate"

    if severity == "none":
        summary = (
            f"No anomalies detected across {n} quarters "
            f"(mean EPS = {mean:.2f}, std = {std:.2f})."
        )
    else:
        summary = (
            f"{len(anomalies)} anomalous quarter(s) out of {n} "
            f"(mean EPS = {mean:.2f}, std = {std:.2f}). "
            f"Worst: index {worst['index']} with EPS {worst['eps']:.2f} "
            f"(z = {worst['z_score']:.2f}). Severity: {severity}."
        )

    return {
        "mean": mean,
        "std": std,
        "anomalies": anomalies,
        "worst_quarter": worst,
        "severity": severity,
        "summary": summary,
    }


def search_worst_period(earnings_history):
    """
    Linear search for the lowest-EPS quarter.

    Returns:
        {"index": i, "eps": v} for the worst quarter, or None if the input
        is empty. Index is into the original (newest-first) list.
    """
    if not earnings_history:
        return None

    worst_idx = 0
    worst_val = earnings_history[0]
    # Standard linear scan — O(n), no sorting needed.
    for i in range(1, len(earnings_history)):
        if earnings_history[i] < worst_val:
            worst_idx = i
            worst_val = earnings_history[i]

    return {"index": worst_idx, "eps": float(worst_val)}


def analyze_trend(earnings_history, threshold=0.05):
    """
    Classify the overall trend by splitting the series in half and comparing
    average EPS.

    The input is newest-first, so:
      - newer_half = earnings_history[:half]   (chronologically later)
      - older_half = earnings_history[half:]   (chronologically earlier)

    Returns "improving", "declining", or "stable".
    A relative change with magnitude below `threshold` (default 5% of the
    older mean) is considered stable.
    """
    n = len(earnings_history)
    if n < 2:
        return "stable"

    half = n // 2
    newer = np.asarray(earnings_history[:half], dtype=float)
    older = np.asarray(earnings_history[half:], dtype=float)

    if newer.size == 0 or older.size == 0:
        return "stable"

    newer_mean = float(newer.mean())
    older_mean = float(older.mean())

    # Compare relative change so that the threshold is scale-invariant.
    # Fall back to absolute comparison if the older mean is zero.
    if older_mean == 0:
        change = newer_mean - older_mean
        if abs(change) < threshold:
            return "stable"
        return "improving" if change > 0 else "declining"

    pct_change = (newer_mean - older_mean) / abs(older_mean)
    if abs(pct_change) < threshold:
        return "stable"
    return "improving" if pct_change > 0 else "declining"
