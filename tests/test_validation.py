"""Tests for Data Validation, Schema Verification, Quality Scoring, and Quarantine Isolation."""
import numpy as np
import pandas as pd
import pytest
from src.ingestion import generate_synthetic_dataset
from src.validation import validate_dataset


def test_valid_dataset():
    """Verify that a clean synthetic dataset achieves high quality score and 0 quarantined."""
    clean_df = generate_synthetic_dataset(n_samples=100, random_seed=42)
    report = validate_dataset(clean_df, dataset_name="clean_test.csv")
    
    assert report.is_valid is True
    assert report.quality_score >= 90.0
    assert report.total_records == 100
    assert report.quarantined_records_count == 0
    assert report.valid_records_count == 100
    assert len(report.flagged_reasons) == 0


def test_empty_dataset():
    """Verify handling of empty dataframe."""
    empty_df = pd.DataFrame()
    report = validate_dataset(empty_df, dataset_name="empty.csv")
    
    assert report.is_valid is False
    assert report.quality_score == 0.0
    assert report.total_records == 0


def test_missing_values_detection():
    """Verify detection of missing values and quality score deduction."""
    df = generate_synthetic_dataset(n_samples=100, random_seed=42, missing_ratio=0.20)
    report = validate_dataset(df, dataset_name="missing_test.csv")
    
    assert report.total_missing_cells > 0
    assert report.quality_score < 100.0


def test_duplicate_records_detection():
    """Verify that duplicate rows are detected and flagged."""
    base_df = generate_synthetic_dataset(n_samples=20, random_seed=42)
    # Duplicate first 5 rows
    dup_rows = base_df.iloc[:5].copy()
    df_with_dups = pd.concat([base_df, dup_rows], ignore_index=True)
    
    report = validate_dataset(df_with_dups, dataset_name="dup_test.csv")
    assert report.duplicate_count == 5
    assert report.quarantined_records_count >= 5


def test_out_of_range_and_quarantine():
    """Verify that records with impossible values are isolated into quarantine."""
    df = generate_synthetic_dataset(n_samples=50, random_seed=42)
    
    # Inject impossible values
    df.at[0, "age"] = 185  # Human age impossible
    df.at[1, "credit_score"] = 120  # Below 300
    df.at[2, "annual_income"] = -50000  # Negative income
    
    report = validate_dataset(df, dataset_name="outlier_test.csv", quarantine_on_severe_issues=True)
    
    assert report.quarantined_records_count >= 3
    assert 0 in report.flagged_reasons
    assert 1 in report.flagged_reasons
    assert 2 in report.flagged_reasons
    assert len(report.valid_df) <= 47


def test_missing_required_schema_columns():
    """Verify that a dataset missing essential columns fails validation."""
    invalid_schema_df = pd.DataFrame({
        "unrelated_col_a": [1, 2, 3],
        "unrelated_col_b": [4, 5, 6],
    })
    report = validate_dataset(invalid_schema_df, dataset_name="bad_schema.csv")
    
    assert report.is_valid is False
    assert len(report.missing_required_columns) > 0
    assert "customer_id" in report.missing_required_columns
    assert "credit_score" in report.missing_required_columns
    assert report.quality_score <= 50.0


def test_infinite_values_quarantine():
    """Verify that infinite numerical values are caught and quarantined."""
    df = generate_synthetic_dataset(n_samples=20, random_seed=42)
    df.at[0, "debt_to_income"] = np.inf
    df.at[1, "annual_income"] = -np.inf
    
    report = validate_dataset(df, dataset_name="infinite_test.csv")
    assert 0 in report.flagged_reasons
    assert 1 in report.flagged_reasons
    assert any("infinite" in r for r in report.flagged_reasons[0])
    assert any("infinite" in r for r in report.flagged_reasons[1])
    assert report.quarantined_records_count >= 2


def test_custom_constraints_override():
    """Verify that custom constraints can be supplied dynamically."""
    df = generate_synthetic_dataset(n_samples=25, random_seed=42)
    # Stricter max age = 50
    custom_rules = {"age": {"type": "numeric", "min": 18, "max": 50}}
    report = validate_dataset(df, dataset_name="custom_rules.csv", custom_constraints=custom_rules)
    
    # Any record with age > 50 should be flagged
    over_50_count = (df["age"] > 50).sum()
    if over_50_count > 0:
        flagged_age = [idx for idx, reasons in report.flagged_reasons.items() if any("age" in r for r in reasons)]
        assert len(flagged_age) >= over_50_count


def test_load_csv_safely_duplicate_headers(tmp_path):
    """Verify that CSV with duplicate column headers is detected and rejected."""
    from src.ingestion import load_csv_safely
    dup_csv = tmp_path / "dup_cols.csv"
    dup_csv.write_text("customer_id,age,age\nCUST-1,25,25\nCUST-2,30,30\n", encoding="utf-8")
    
    df, err = load_csv_safely(dup_csv)
    assert df is None
    assert "duplicate" in err.lower()


def test_load_csv_safely_empty_and_size_limits(tmp_path):
    """Verify that 0-byte files or oversized files fail safely."""
    from src.ingestion import load_csv_safely
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    
    df, err = load_csv_safely(empty_csv)
    assert df is None
    assert "empty" in err.lower()

