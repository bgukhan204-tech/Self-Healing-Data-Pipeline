"""Tests for Isolation Forest Anomaly Detector and Statistical Baseline."""
import numpy as np
import pandas as pd
import pytest
from src.anomaly_detection import AnomalyDetector
from src.ingestion import generate_synthetic_dataset


@pytest.fixture
def fitted_detector():
    """Fixture providing a fitted AnomalyDetector."""
    ref_df = generate_synthetic_dataset(n_samples=300, random_seed=42)
    detector = AnomalyDetector(contamination=0.05, random_seed=42)
    detector.fit(ref_df)
    return detector


def test_fit_and_predict_normal(fitted_detector):
    """Verify detector predictions on normal data."""
    test_df = generate_synthetic_dataset(n_samples=100, random_seed=123)
    res = fitted_detector.predict(test_df)
    
    assert res.total_records == 100
    assert len(res.is_anomaly) == 100
    assert len(res.anomaly_scores) == 100
    # Expected low anomaly count consistent with 5% contamination
    assert res.total_anomalies <= 15
    assert all(0.0 <= s <= 1.0 for s in res.anomaly_scores)


def test_predict_injected_outliers(fitted_detector):
    """Verify that extreme outliers are reliably flagged as anomalies."""
    outlier_df = generate_synthetic_dataset(n_samples=50, random_seed=999, outlier_ratio=0.50)
    res = fitted_detector.predict(outlier_df)
    
    assert res.total_anomalies >= 15
    assert not res.anomalous_records_df.empty
    assert "top_contributing_indicator" in res.anomalous_records_df.columns
    assert "statistical_baseline_flag" in res.anomalous_records_df.columns


def test_statistical_baseline_comparison(fitted_detector):
    """Verify comparison with univariate Z-score baseline."""
    df = generate_synthetic_dataset(n_samples=30, random_seed=42)
    # Inject 1 obvious massive income
    df.at[0, "annual_income"] = 5000000.0
    
    res = fitted_detector.predict(df)
    assert res.is_anomaly[0] == True
    assert res.statistical_baseline_flags[0] == True
    assert "annual_income" in res.top_contributing_features[0]
