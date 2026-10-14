"""Anomaly Detection Module using Isolation Forest and Statistical Baseline Comparison."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from config.settings import get_settings
from src.preprocessing import NUMERIC_FEATURES

LIMITATIONS_DISCLAIMER = (
    "Note on Unsupervised Anomaly Detection: Anomaly detection identifies multivariate statistical "
    "outliers relative to the baseline training distribution. An anomaly is an unusual observation, "
    "not a confirmed pipeline error, system defect, or fraudulent transaction. Outliers may reflect "
    "novel, legitimate edge cases or natural environmental shifts. Do not treat raw anomaly flags "
    "as definitive defects without contextual or manual verification."
)


@dataclass
class AnomalyDetectionResult:
    """Output of anomaly detection on a batch of records."""
    is_anomaly: np.ndarray  # Boolean array: True if anomalous
    anomaly_scores: np.ndarray  # Continuous score: higher means more anomalous (0 to 1 scale)
    raw_decision_scores: np.ndarray  # Raw decision_function output (negative = outlier)
    statistical_baseline_flags: np.ndarray  # Boolean array from univariate Z-score baseline
    top_contributing_features: List[str]
    total_anomalies: int
    anomaly_percentage: float
    total_records: int
    anomalous_records_df: pd.DataFrame
    limitations_disclaimer: str = LIMITATIONS_DISCLAIMER


class AnomalyDetector:
    """Production-grade Isolation Forest Anomaly Detector with Statistical Baseline."""

    def __init__(
        self,
        features: Optional[List[str]] = None,
        contamination: float = 0.05,
        random_seed: int = 42,
    ):
        self.features = features or NUMERIC_FEATURES
        self.contamination = contamination
        self.random_seed = random_seed
        self.model: Optional[IsolationForest] = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, reference_df: pd.DataFrame) -> "AnomalyDetector":
        """Fit Isolation Forest and reference baseline statistics."""
        available_features = [f for f in self.features if f in reference_df.columns]
        if not available_features:
            raise ValueError(f"None of the required features {self.features} found in reference data.")
        
        self.features = available_features
        X_raw = reference_df[self.features].to_numpy()
        X_imputed = self.imputer.fit_transform(X_raw)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Compute univariate reference statistics for explanation and baseline comparison
        for i, feat in enumerate(self.features):
            col_data = X_imputed[:, i]
            mean = float(np.mean(col_data))
            std = float(np.std(col_data)) if np.std(col_data) > 1e-6 else 1.0
            q25 = float(np.percentile(col_data, 25))
            q75 = float(np.percentile(col_data, 75))
            self.reference_stats[feat] = {
                "mean": mean,
                "std": std,
                "q25": q25,
                "q75": q75,
                "iqr": q75 - q25 if (q75 - q25) > 1e-6 else 1.0,
            }

        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, incoming_df: pd.DataFrame) -> AnomalyDetectionResult:
        """
        Evaluate incoming records for anomalies using Isolation Forest and Z-score baseline.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("AnomalyDetector must be fitted before scoring incoming records.")

        if incoming_df.empty:
            empty_bool = np.array([], dtype=bool)
            empty_float = np.array([], dtype=float)
            return AnomalyDetectionResult(
                is_anomaly=empty_bool,
                anomaly_scores=empty_float,
                raw_decision_scores=empty_float,
                statistical_baseline_flags=empty_bool,
                top_contributing_features=[],
                total_anomalies=0,
                anomaly_percentage=0.0,
                total_records=0,
                anomalous_records_df=pd.DataFrame(),
            )

        X_raw = incoming_df[self.features].to_numpy()
        X_imputed = self.imputer.transform(X_raw)
        X_scaled = self.scaler.transform(X_imputed)

        # 1. Isolation Forest predictions
        # -1 = anomaly, 1 = normal
        raw_preds = self.model.predict(X_scaled)
        is_anomaly = (raw_preds == -1)

        # Decision function: lower values mean more anomalous
        raw_decisions = self.model.decision_function(X_scaled)
        # Normalize decision function to an anomaly score between 0.0 and 1.0 (higher = more anomalous)
        # Decision function typically spans [-0.5, 0.5]
        anomaly_scores = np.clip(0.5 - raw_decisions, 0.0, 1.0)

        # 2. Statistical Baseline (Z-score test where |z| > 3.0)
        baseline_flags = np.zeros(len(incoming_df), dtype=bool)
        top_contributing_features: List[str] = []

        for i in range(len(incoming_df)):
            row_vals = X_imputed[i]
            max_dev = -1.0
            top_feat = "none"

            for j, feat in enumerate(self.features):
                stats = self.reference_stats[feat]
                z_score = abs(row_vals[j] - stats["mean"]) / stats["std"]
                if z_score > 3.0:
                    baseline_flags[i] = True
                
                # Check relative deviation for explanation
                if z_score > max_dev:
                    max_dev = z_score
                    top_feat = f"{feat} (|z|={z_score:.1f})"

            top_contributing_features.append(top_feat if is_anomaly[i] else "normal")

        total_anomalies = int(np.sum(is_anomaly))
        total_records = len(incoming_df)
        anomaly_percentage = round((total_anomalies / total_records) * 100.0, 2) if total_records > 0 else 0.0

        # Construct anomalous dataframe subset with metadata
        anomalous_df = incoming_df[is_anomaly].copy()
        if not anomalous_df.empty:
            anomalous_df["anomaly_score"] = anomaly_scores[is_anomaly]
            anomalous_df["top_contributing_indicator"] = [
                top_contributing_features[i] for i in range(len(incoming_df)) if is_anomaly[i]
            ]
            anomalous_df["statistical_baseline_flag"] = baseline_flags[is_anomaly]

        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            anomaly_scores=anomaly_scores,
            raw_decision_scores=raw_decisions,
            statistical_baseline_flags=baseline_flags,
            top_contributing_features=top_contributing_features,
            total_anomalies=total_anomalies,
            anomaly_percentage=anomaly_percentage,
            total_records=total_records,
            anomalous_records_df=anomalous_df,
        )

    def save(self, filepath: str) -> None:
        """Save fitted detector artifact to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "AnomalyDetector":
        """Load fitted detector from disk."""
        return joblib.load(filepath)
