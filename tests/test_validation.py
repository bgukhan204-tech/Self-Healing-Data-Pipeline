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
