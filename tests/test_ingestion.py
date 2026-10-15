"""Tests for Data Ingestion: Safe CSV Loading, Boundary Checks, and File Sanitization."""
import io
from pathlib import Path
import pandas as pd
import pytest
from src.ingestion import generate_synthetic_dataset, load_csv_safely, save_uploaded_batch


def test_generate_synthetic_dataset_defaults():
    """Verify synthetic dataset generator produces expected columns and rows."""
    df = generate_synthetic_dataset(n_samples=50, random_seed=42)
    assert len(df) == 50
    assert "customer_id" in df.columns
    assert "annual_income" in df.columns
    assert "credit_score" in df.columns
    assert "loan_status" in df.columns


def test_load_valid_csv(tmp_path):
    """Verify successfully loading a valid CSV file."""
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    df, err = load_csv_safely(csv_file)
    assert err is None
    assert df is not None
    assert len(df) == 2
    assert list(df.columns) == ["a", "b", "c"]


def test_load_empty_zero_byte_csv(tmp_path):
    """Verify that a 0-byte CSV file is rejected with a clear error."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    df, err = load_csv_safely(empty_file)
    assert df is None
    assert "empty" in err.lower()


def test_load_empty_bytes_io():
    """Verify that empty BytesIO buffer is rejected."""
    buf = io.BytesIO(b"")
    df, err = load_csv_safely(buf)
    assert df is None
    assert "empty" in err.lower()


def test_load_malformed_csv(tmp_path):
    """Verify that malformed CSV rows return a parsing error message."""
    malformed_file = tmp_path / "malformed.csv"
    malformed_file.write_text('col1,col2\n"unclosed quote,123\n456', encoding="utf-8")
    df, err = load_csv_safely(malformed_file)
    assert df is None
    assert "malformed" in err.lower() or "error" in err.lower()


def test_load_duplicate_column_headers(tmp_path):
    """Verify that duplicate headers are detected and rejected."""
    dup_file = tmp_path / "duplicates.csv"
    dup_file.write_text("user_id,amount,user_id\n1,100,1\n2,200,2\n", encoding="utf-8")
    df, err = load_csv_safely(dup_file)
    assert df is None
    assert "duplicate column headers" in err.lower()


def test_load_oversized_file_limit(tmp_path):
    """Verify that files exceeding the size limit are rejected cleanly."""
    large_file = tmp_path / "large.csv"
    large_file.write_text("x,y\n" + "1,2\n" * 1000, encoding="utf-8")
    # Setting max_size_mb to an impossibly small fraction of an MB (e.g. 0.0001 MB)
    df, err = load_csv_safely(large_file, max_size_mb=0)
    assert df is None
    assert "exceeds maximum allowed" in err.lower()


def test_save_uploaded_batch_sanitization(tmp_path, monkeypatch):
    """Verify that uploaded filenames are sanitized and given unique timestamps."""
    from config.settings import get_settings
    settings = get_settings()
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(settings, "uploads_dir", upload_dir)

    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    saved_path = save_uploaded_batch(df, original_filename="../../danger name.csv")

    assert saved_path.exists()
    assert saved_path.parent == upload_dir
    # Filename must not contain path traversal characters
    assert ".." not in saved_path.name
    assert " " not in saved_path.name
    assert saved_path.name.endswith("danger_name.csv")
