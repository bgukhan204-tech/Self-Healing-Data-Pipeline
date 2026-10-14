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
