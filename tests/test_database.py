"""Tests for SQLite database schema, inserts, updates, and queries."""
import os
import tempfile
from pathlib import Path
import pytest
from src.database import (
    get_active_model,
    get_connection,
    get_latest_metrics_summary,
    get_model_history,
    get_pipeline_runs,
    get_recovery_actions,
    get_system_events,
    init_db,
    log_anomaly_events,
    log_drift_events,
    log_pipeline_run,
    log_recovery_action,
    log_system_event,
    register_model_run,
    update_model_status,
)


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing an isolated temporary database path."""
    db_file = tmp_path / "test_pipeline.db"
    init_db(db_file)
    return db_file


def test_init_db(temp_db):
    """Verify that all required tables and indexes are initialized."""
    with get_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        required = {
            "pipeline_runs",
            "anomaly_events",
            "drift_events",
            "model_registry",
            "recovery_actions",
            "system_events",
        }
        assert required.issubset(tables)


def test_system_event_logging(temp_db):
    """Verify immutable system event logging and filtering."""
    evt_id = log_system_event(
        event_type="TEST_EVENT",
        severity="WARNING",
        component="TEST_COMP",
        description="A test event was recorded.",
        db_path=temp_db,
    )
    assert evt_id.startswith("evt_")

    events = get_system_events(severity="WARNING", db_path=temp_db)
    assert len(events) == 1
    assert events[0]["event_type"] == "TEST_EVENT"
    assert events[0]["severity"] == "WARNING"


def test_pipeline_run_persistence(temp_db):
    """Verify storing and querying pipeline run metadata."""
    log_pipeline_run(
        run_id="run_test_01",
        dataset_name="test_data.csv",
        records_total=500,
        records_valid=480,
        records_quarantined=20,
        records_anomalous=15,
        quality_score=94.5,
        status="Healthy",
        execution_time_ms=120.5,
        db_path=temp_db,
    )
    runs = get_pipeline_runs(limit=10, db_path=temp_db)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_test_01"
    assert runs[0]["records_total"] == 500
    assert runs[0]["quality_score"] == 94.5


def test_model_registry_lifecycle(temp_db):
    """Verify model registration, promotion, demotion, and rollback state."""
    # Register candidate
    register_model_run(
        model_version="v1.0.0",
        algorithm="RandomForest",
        accuracy=0.88,
        precision_score=0.85,
        recall_score=0.82,
        f1_score=0.835,
        status="active",
        artifact_path="/tmp/model_v1.0.0.joblib",
        db_path=temp_db,
    )

    active = get_active_model(temp_db)
    assert active is not None
    assert active["model_version"] == "v1.0.0"

    # Register second candidate and promote it
    register_model_run(
        model_version="v1.1.0",
        algorithm="RandomForest",
        accuracy=0.91,
        precision_score=0.89,
        recall_score=0.86,
        f1_score=0.875,
        status="active",
        artifact_path="/tmp/model_v1.1.0.joblib",
        db_path=temp_db,
    )

    active_now = get_active_model(temp_db)
    assert active_now["model_version"] == "v1.1.0"

    # Previous v1.0.0 should be archived
    history = get_model_history(temp_db)
    v1_entry = next(m for m in history if m["model_version"] == "v1.0.0")
    assert v1_entry["status"] == "archived"


def test_recovery_action_logging(temp_db):
    """Verify recovery action persistence and query."""
    action_id = log_recovery_action(
        trigger_event="INVALID_INCOMING_DATA",
        action_type="QUARANTINE_RECORDS",
        status="SUCCESS",
        outcome_summary="Isolated 5 corrupt records.",
        run_id="run_001",
        db_path=temp_db,
    )
    assert action_id.startswith("rec_")
    actions = get_recovery_actions(db_path=temp_db)
    assert len(actions) == 1
    assert actions[0]["status"] == "SUCCESS"
