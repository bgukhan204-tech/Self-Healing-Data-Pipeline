"""Reusable transformations for anomaly investigation views."""
from __future__ import annotations

import pandas as pd


def prepare_anomaly_investigation(
    anomalies: pd.DataFrame,
    detector: str = "All detectors",
    start_date=None,
    end_date=None,
    sort_by: str = "Newest first",
) -> pd.DataFrame:
    """Filter and sort safe anomaly metadata, adding a human-readable score band."""
    result = anomalies.copy()
    if "Detector" in result.columns and detector != "All detectors":
        result = result[result["Detector"].astype(str) == detector]

    if "Recorded At" in result.columns:
        timestamps = pd.to_datetime(result["Recorded At"], errors="coerce", utc=True)
        if start_date is not None:
            start = pd.Timestamp(start_date, tz="UTC")
            result = result.loc[timestamps >= start]
            timestamps = timestamps.loc[result.index]
        if end_date is not None:
            end_exclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
            result = result.loc[timestamps < end_exclusive]
            timestamps = timestamps.loc[result.index]
        result["Recorded At"] = timestamps

    if "Anomaly Score" in result.columns:
        scores = pd.to_numeric(result["Anomaly Score"], errors="coerce")
        result["Severity Band"] = "Unknown"
        result.loc[scores < 0.40, "Severity Band"] = "Low"
        result.loc[(scores >= 0.40) & (scores < 0.70), "Severity Band"] = "Moderate"
        result.loc[scores >= 0.70, "Severity Band"] = "High"

    if sort_by == "Highest score first" and "Anomaly Score" in result.columns:
        result["Anomaly Score"] = pd.to_numeric(result["Anomaly Score"], errors="coerce")
        result = result.sort_values("Anomaly Score", ascending=False, na_position="last")
    elif sort_by == "Oldest first" and "Recorded At" in result.columns:
        result = result.sort_values("Recorded At", ascending=True, na_position="last")
    elif "Recorded At" in result.columns:
        result = result.sort_values("Recorded At", ascending=False, na_position="last")
    return result


def build_anomaly_score_summary(anomalies: pd.DataFrame) -> pd.DataFrame:
    """Build a compact CSV-friendly summary for the supplied visible detections."""
    if "Anomaly Score" not in anomalies.columns:
        scores = pd.Series(dtype=float)
    else:
        scores = pd.to_numeric(anomalies["Anomaly Score"], errors="coerce").dropna()
    bands = anomalies.get("Severity Band", pd.Series(dtype=str)).value_counts()
    metrics = [
        ("Displayed detections", int(len(scores))),
        ("Average score", float(scores.mean()) if not scores.empty else None),
        ("Median score", float(scores.median()) if not scores.empty else None),
        ("95th percentile", float(scores.quantile(0.95)) if not scores.empty else None),
        ("Highest score", float(scores.max()) if not scores.empty else None),
        ("Low detections", int(bands.get("Low", 0))),
        ("Moderate detections", int(bands.get("Moderate", 0))),
        ("High detections", int(bands.get("High", 0))),
        ("Unknown-score detections", int(bands.get("Unknown", 0))),
    ]
    return pd.DataFrame(metrics, columns=["Metric", "Value"])
