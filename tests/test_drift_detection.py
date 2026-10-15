"""Tests for Data Drift Monitoring using Kolmogorov-Smirnov and Chi-Square Tests."""
import pytest
from src.drift_detection import DriftMonitor
from src.ingestion import generate_synthetic_dataset


@pytest.fixture
def monitor():
    """Fixture providing a DriftMonitor loaded with reference data."""
    ref_df = generate_synthetic_dataset(n_samples=500, random_seed=42)
    m = DriftMonitor(p_val_threshold=0.05, critical_p_val=0.01)
    m.set_reference_data(ref_df)
    return m


def test_drift_on_stable_data(monitor):
    """Verify that an identical or similar distribution does not trigger drift."""
    stable_batch = generate_synthetic_dataset(n_samples=300, random_seed=101, drift_factor=0.0)
    report = monitor.evaluate_drift(stable_batch)
    
    assert report.overall_drift_status == "HEALTHY"
    assert report.drifted_features_count <= 1  # May have at most 1 false positive at alpha=0.05
    assert not report.summary_df.empty


def test_drift_on_shifted_data(monitor):
    """Verify that heavily shifted distribution triggers warning or critical drift."""
    drifted_batch = generate_synthetic_dataset(n_samples=400, random_seed=999, drift_factor=0.8)
    report = monitor.evaluate_drift(drifted_batch)
    
    assert report.overall_drift_status in ["WARNING", "CRITICAL"]
    assert report.drifted_features_count >= 2
    
    # Check specific shifted numeric columns
    shifted_names = [r.feature_name for r in report.feature_reports if r.drift_detected]
    assert any(col in shifted_names for col in ["age", "annual_income", "credit_score", "debt_to_income"])


def test_categorical_frequency_drift(monitor):
    """Verify that severe categorical shift triggers categorical drift detection."""
    df = generate_synthetic_dataset(n_samples=250, random_seed=42)
    # Force single categorical category
    df["home_ownership"] = "RENT"
    report = monitor.evaluate_drift(df)
    
    cat_rep = next((r for r in report.feature_reports if r.feature_name == "home_ownership"), None)
    assert cat_rep is not None
    assert cat_rep.drift_detected is True
    assert cat_rep.test_method == "chi_square"
    assert cat_rep.distance_metric > 0.10


def test_practical_significance_distinction(monitor):
    """Verify that practical importance is distinguished in the summary table."""
    drifted_batch = generate_synthetic_dataset(n_samples=400, random_seed=999, drift_factor=0.9)
    report = monitor.evaluate_drift(drifted_batch)
    
    assert "Practical Importance" in report.summary_df.columns
    # Check that high drift has practical significance marked
    assert any(report.summary_df["Practical Importance"].isin(["HIGH", "MODERATE"]))


def test_drift_empty_and_missing_features(monitor):
    """Verify safe evaluation when incoming dataframe is empty or missing columns."""
    import pandas as pd
    empty_rep = monitor.evaluate_drift(pd.DataFrame())
    assert empty_rep.overall_drift_status == "HEALTHY"
    assert empty_rep.drifted_features_count == 0

    partial_df = pd.DataFrame({"age": [25, 30, 35, 40, 45, 50]})
    partial_rep = monitor.evaluate_drift(partial_df)
    assert partial_rep.total_features_evaluated == 1

