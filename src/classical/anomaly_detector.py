"""
Earnings anomaly detector.

Three classical signals over a quarterly EPS series:
  - robust statistical anomaly detection (modified z-score, median/MAD)
  - linear search for the single worst quarter
  - trend classification (improving / declining / stable)

Inputs are lists of EPS values ordered newest-first (index 0 = most recent
quarter, last index = oldest), matching the loader's output convention.
"""

import numpy as np


# Modified z-score cut-offs (Iglewicz & Hoaglin). 3.5 is the standard
# outlier threshold; 5.0 marks a quarter that looks like a different regime.
MODERATE_Z = 3.5
SEVERE_Z = 5.0

# Scale factors that make MAD / mean-absolute-deviation comparable to a
# standard deviation for normally distributed data.
_MAD_SCALE = 0.6745
_MEANAD_SCALE = 1.253314


def _robust_z_scores(arr):
    """
    Modified z-scores based on the median and MAD.

    The classic (x - mean) / std score is a poor fit for 8 quarters: the
    outlier inflates the std it is measured against, and with n = 8 no
    point can ever exceed |z| = (n - 1) / sqrt(n) ~= 2.47. Median and MAD
    are barely moved by a single outlier, so one bad quarter stands out.

    If more than half the values are identical MAD is 0; fall back to the
    mean absolute deviation around the median. If that is also 0 the
    series is constant and every score is 0.
    """
    median = float(np.median(arr))
    deviations = arr - median
    mad = float(np.median(np.abs(deviations)))
    if mad > 0:
        return _MAD_SCALE * deviations / mad, median, mad
    mean_ad = float(np.mean(np.abs(deviations)))
    if mean_ad > 0:
        return deviations / (_MEANAD_SCALE * mean_ad), median, mad
    return np.zeros_like(arr), median, mad


def detect_earnings_anomaly(earnings_history):
    """
    Flag any quarter whose modified z-score exceeds MODERATE_Z.

    Returns:
        {
          "mean":          mean EPS (for display),
          "std":           sample standard deviation (for display),
          "median":        median EPS,
          "mad":           median absolute deviation,
          "anomalies":     [{"index": i, "eps": v, "z_score": z}, ...],
          "worst_quarter": the anomaly with the largest |z|, or None,
          "severity":      "none" | "moderate" | "severe",
          "summary":       plain-English explanation,
        }

    `z_score` is the robust (modified) z-score, not (x - mean) / std.

    Severity rules:
      - "none"     : no quarter reaches |z| > MODERATE_Z
      - "moderate" : at least one |z| > MODERATE_Z, all < SEVERE_Z
      - "severe"   : at least one |z| >= SEVERE_Z

    Known limitation: EPS is not seasonally adjusted, so a business with a
    strong holiday quarter can show a recurring spike.
    """
    n = len(earnings_history)
    if n < 2:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "mad": None,
            "anomalies": [],
            "worst_quarter": None,
            "severity": "none",
            "summary": "Not enough quarters to compute anomalies.",
        }

    arr = np.asarray(earnings_history, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    z_scores, median, mad = _robust_z_scores(arr)

    anomalies = [
        {"index": i, "eps": float(v), "z_score": float(z)}
        for i, (v, z) in enumerate(zip(arr, z_scores))
        if abs(z) > MODERATE_Z
    ]

    worst = max(anomalies, key=lambda a: abs(a["z_score"])) if anomalies else None

    if not anomalies:
        severity = "none"
    elif any(abs(a["z_score"]) >= SEVERE_Z for a in anomalies):
        severity = "severe"
    else:
        severity = "moderate"

    if severity == "none":
        summary = (
            f"No anomalies detected across {n} quarters "
            f"(median EPS = {median:.2f}, mean = {mean:.2f})."
        )
    else:
        summary = (
            f"{len(anomalies)} anomalous quarter(s) out of {n} "
            f"(median EPS = {median:.2f}, mean = {mean:.2f}). "
            f"Worst: index {worst['index']} with EPS {worst['eps']:.2f} "
            f"(robust z = {worst['z_score']:.2f}). Severity: {severity}."
        )

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "mad": mad,
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
