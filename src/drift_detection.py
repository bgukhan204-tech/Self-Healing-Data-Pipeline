"""Data Drift Monitoring using Kolmogorov-Smirnov, Wasserstein Distance, and Chi-Square Tests."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp, wasserstein_distance
from config.settings import get_settings
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

DRIFT_DISCLAIMER = (
    "Drift vs Performance Degradation: Data drift measures distribution divergence in input features "
    "relative to the reference baseline. It indicates an environmental or population shift. However, "
    "input drift alone DOES NOT prove that predictive accuracy has degraded unless true ground-truth labels "
    "are collected and evaluated."
)


@dataclass
class FeatureDriftReport:
    """Drift test results for a single feature."""
    feature_name: str
    feature_type: str  # 'numeric' or 'categorical'
    test_method: str   # 'ks_test' or 'chi_square'
    statistic: float
    p_value: float
    distance_metric: float  # Wasserstein distance for numeric, Total Variation Distance for cat
    drift_detected: bool
    severity: str  # 'none', 'warning', 'critical'
    description: str


@dataclass
class DatasetDriftReport:
    """Aggregate drift monitoring report across all evaluated features."""
    overall_drift_status: str  # 'HEALTHY', 'WARNING', 'CRITICAL'
    drifted_features_count: int
    total_features_evaluated: int
    drift_percentage: float
    feature_reports: List[FeatureDriftReport]
    summary_df: pd.DataFrame
    disclaimer: str = DRIFT_DISCLAIMER


class DriftMonitor:
    """Monitors statistical distribution drift between reference and incoming batches."""

    def __init__(
        self,
        p_val_threshold: Optional[float] = None,
        critical_p_val: Optional[float] = None,
        wasserstein_threshold: Optional[float] = None,
    ):
        settings = get_settings()
        self.p_val_threshold = p_val_threshold or settings.drift_pvalue_threshold
        self.critical_p_val = critical_p_val or settings.drift_critical_pvalue
        self.wasserstein_threshold = wasserstein_threshold or settings.drift_wasserstein_threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self.num_cols = NUMERIC_FEATURES
        self.cat_cols = CATEGORICAL_FEATURES

    def set_reference_data(self, reference_df: pd.DataFrame) -> None:
        """Cache the reference baseline dataset."""
        if reference_df.empty:
            raise ValueError("Reference dataset cannot be empty.")
        self.reference_data = reference_df.copy()

    def evaluate_drift(self, incoming_df: pd.DataFrame) -> DatasetDriftReport:
        """
        Compare incoming batch distribution against reference dataset.
        """
        if self.reference_data is None:
            raise ValueError("Reference dataset must be set before evaluating drift.")
        if incoming_df.empty:
            return DatasetDriftReport(
                overall_drift_status="HEALTHY",
                drifted_features_count=0,
                total_features_evaluated=0,
                drift_percentage=0.0,
                feature_reports=[],
                summary_df=pd.DataFrame(),
            )

        feature_reports: List[FeatureDriftReport] = []

        # 1. Numeric Feature Drift (Two-sample KS-test & Wasserstein Distance)
        for col in self.num_cols:
            if col not in self.reference_data.columns or col not in incoming_df.columns:
                continue

            ref_series = self.reference_data[col].dropna()
            inc_series = incoming_df[col].dropna()

            if len(ref_series) < 5 or len(inc_series) < 5:
                continue

            # Two-sample KS-test
            ks_res = ks_2samp(ref_series, inc_series)
            stat = float(ks_res.statistic)
            p_val = float(ks_res.pvalue)

            # Standardized Wasserstein distance (divided by reference std for scale-invariance)
            ref_std = float(ref_series.std()) if ref_series.std() > 1e-6 else 1.0
            w_dist = float(wasserstein_distance(ref_series, inc_series) / ref_std)

            # Determine severity
            if p_val < self.critical_p_val or w_dist > (self.wasserstein_threshold * 1.5):
                drift_detected = True
                severity = "critical"
                desc = f"Critical distribution shift (p-val={p_val:.2e}, Wasserstein={w_dist:.2f})"
            elif p_val < self.p_val_threshold or w_dist > self.wasserstein_threshold:
                drift_detected = True
                severity = "warning"
                desc = f"Moderate distribution shift (p-val={p_val:.3f}, Wasserstein={w_dist:.2f})"
            else:
                drift_detected = False
                severity = "none"
                desc = f"Stable distribution (p-val={p_val:.3f}, Wasserstein={w_dist:.2f})"

            feature_reports.append(FeatureDriftReport(
                feature_name=col,
                feature_type="numeric",
                test_method="ks_test",
                statistic=stat,
                p_value=p_val,
                distance_metric=w_dist,
                drift_detected=drift_detected,
                severity=severity,
                description=desc,
            ))

        # 2. Categorical Feature Drift (Chi-Square Goodness-of-Fit / Frequency Comparison)
        for col in self.cat_cols:
            if col not in self.reference_data.columns or col not in incoming_df.columns:
                continue

            ref_counts = self.reference_data[col].value_counts(normalize=True)
            inc_counts = incoming_df[col].value_counts(normalize=True)

            # Align categories
            all_cats = list(set(ref_counts.index).union(set(inc_counts.index)))
            ref_freq = np.array([ref_counts.get(c, 1e-4) for c in all_cats])
            inc_freq = np.array([inc_counts.get(c, 1e-4) for c in all_cats])

            # Normalize frequencies to probabilities
            ref_freq = ref_freq / ref_freq.sum()
            inc_freq = inc_freq / inc_freq.sum()

            # Total Variation Distance (TVD) as distance metric [0, 1]
            tvd = float(0.5 * np.sum(np.abs(ref_freq - inc_freq)))

            # Chi-square test between observed incoming counts and expected from reference
            inc_raw_counts = np.array([incoming_df[col].value_counts().get(c, 0) for c in all_cats])
            expected_counts = ref_freq * len(incoming_df)
            
            # Avoid divide-by-zero
            valid_mask = expected_counts > 0
            if np.sum(valid_mask) > 1 and np.sum(inc_raw_counts) > 0:
                try:
                    chi_res = chisquare(f_obs=inc_raw_counts[valid_mask], f_exp=expected_counts[valid_mask])
                    chi_stat = float(chi_res.statistic)
                    chi_pval = float(chi_res.pvalue)
                except Exception:
                    chi_stat = float(tvd * 10.0)
                    chi_pval = 0.5
            else:
                chi_stat = 0.0
                chi_pval = 1.0

            if chi_pval < self.critical_p_val or tvd > 0.20:
                drift_detected = True
                severity = "critical"
                desc = f"Critical categorical frequency shift (TVD={tvd:.2f}, p-val={chi_pval:.2e})"
            elif chi_pval < self.p_val_threshold or tvd > 0.10:
                drift_detected = True
                severity = "warning"
                desc = f"Moderate categorical frequency shift (TVD={tvd:.2f}, p-val={chi_pval:.3f})"
            else:
                drift_detected = False
                severity = "none"
                desc = f"Stable categorical frequencies (TVD={tvd:.2f}, p-val={chi_pval:.3f})"

            feature_reports.append(FeatureDriftReport(
                feature_name=col,
                feature_type="categorical",
                test_method="chi_square",
                statistic=chi_stat,
                p_value=chi_pval,
                distance_metric=tvd,
                drift_detected=drift_detected,
                severity=severity,
                description=desc,
            ))

        # Overall Status
        drifted_count = sum(1 for r in feature_reports if r.drift_detected)
        total_feats = len(feature_reports)
        drift_pct = round((drifted_count / total_feats) * 100.0, 1) if total_feats > 0 else 0.0

        has_critical = any(r.severity == "critical" for r in feature_reports)
        if has_critical or drift_pct >= 40.0:
            overall_status = "CRITICAL"
        elif drifted_count > 0 or drift_pct >= 20.0:
            overall_status = "WARNING"
        else:
            overall_status = "HEALTHY"

        # Summary DataFrame
        summary_rows = []
        for r in feature_reports:
            summary_rows.append({
                "Feature": r.feature_name,
                "Type": r.feature_type,
                "Test": r.test_method,
                "Statistic": round(r.statistic, 4),
                "p-value": f"{r.p_value:.4f}" if r.p_value >= 0.0001 else f"{r.p_value:.2e}",
                "Distance Metric": round(r.distance_metric, 4),
                "Drift Detected": "YES" if r.drift_detected else "NO",
                "Severity": r.severity.upper(),
                "Description": r.description,
            })
        summary_df = pd.DataFrame(summary_rows)

        return DatasetDriftReport(
            overall_drift_status=overall_status,
            drifted_features_count=drifted_count,
            total_features_evaluated=total_feats,
            drift_percentage=drift_pct,
            feature_reports=feature_reports,
            summary_df=summary_df,
        )
