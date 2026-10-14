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
    get_recent_drift_events,
    get_recent_failed_recoveries,
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



def test_recovery_metrics_report_no_attempts_as_unavailable(temp_db):
    """No recovery history must not be reported as a perfect success rate."""
    summary = get_latest_metrics_summary(temp_db)

    assert summary["total_recoveries"] == 0
    assert summary["successful_recoveries"] == 0
    assert summary["failed_recoveries"] == 0
    assert summary["recovery_success_rate"] is None


def test_recovery_metrics_include_failed_attempts(temp_db):
    """Recovery aggregates should report successful and failed attempts accurately."""
    log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="SUCCESS",
        outcome_summary="Recovery succeeded.",
        run_id="run_metrics_1",
        db_path=temp_db,
    )
    log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="FAILED",
        outcome_summary="Recovery failed.",
        run_id="run_metrics_2",
        error_details="test failure",
        db_path=temp_db,
    )

    summary = get_latest_metrics_summary(temp_db)

    assert summary["total_recoveries"] == 2
    assert summary["successful_recoveries"] == 1
    assert summary["failed_recoveries"] == 1
    assert summary["recovery_success_rate"] == 50.0



def test_recovery_metrics_count_only_recent_failed_actions(temp_db):
    """Expose recent failed recoveries while excluding older audit records."""
    from datetime import datetime, timedelta, timezone

    log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="FAILED",
        outcome_summary="Recent recovery failure.",
        run_id="run_recent_failure",
        db_path=temp_db,
    )
    old_action_id = log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="FAILED",
        outcome_summary="Older recovery failure.",
        run_id="run_old_failure",
        db_path=temp_db,
    )
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    with get_connection(temp_db) as conn:
        conn.execute(
            "UPDATE recovery_actions SET created_at = ? WHERE action_id = ?",
            (old_timestamp, old_action_id),
        )
        conn.commit()

    summary = get_latest_metrics_summary(temp_db)

    assert summary["failed_recoveries"] == 2
    assert summary["failed_recoveries_24h"] == 1



def test_recent_failed_recoveries_include_actionable_details_and_exclude_successes(temp_db):
    """Failure inspection query should return useful fields and only failed actions."""
    log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="FAILED",
        outcome_summary="Candidate training crashed.",
        error_details="ValueError: target column missing",
        run_id="run_failed_detail",
        db_path=temp_db,
    )
    log_recovery_action(
        trigger_event="DRIFT_WARNING",
        action_type="RETRAIN_MODEL",
        status="SUCCESS",
        outcome_summary="Training succeeded.",
        run_id="run_success_detail",
        db_path=temp_db,
    )

    failures = get_recent_failed_recoveries(limit=10, db_path=temp_db)

    assert len(failures) == 1
    assert failures[0]["status"] == "FAILED"
    assert failures[0]["run_id"] == "run_failed_detail"
    assert failures[0]["trigger_event"] == "DRIFT_CRITICAL"
    assert failures[0]["action_type"] == "RETRAIN_MODEL"
    assert failures[0]["outcome_summary"] == "Candidate training crashed."
    assert failures[0]["error_details"] == "ValueError: target column missing"



def test_recent_drift_query_returns_detected_events_only(temp_db):
    """Recent drift inspection should include detected features and exclude clean ones."""
    log_pipeline_run(
        run_id="run_drift_details",
        dataset_name="drift.csv",
        records_total=10,
        records_valid=10,
        records_quarantined=0,
        records_anomalous=0,
        quality_score=100.0,
        status="Warning",
        execution_time_ms=10.0,
        db_path=temp_db,
    )
    from src.database import log_drift_events

    log_drift_events(
        run_id="run_drift_details",
        drift_results=[
            {
                "feature_name": "temperature",
                "test_method": "ks_test",
                "statistic": 0.42,
                "p_value": 0.01,
                "drift_detected": True,
                "severity": "CRITICAL",
            },
            {
                "feature_name": "humidity",
                "test_method": "ks_test",
                "statistic": 0.08,
                "p_value": 0.70,
                "drift_detected": False,
                "severity": "HEALTHY",
            },
        ],
        db_path=temp_db,
    )

    events = get_recent_drift_events(limit=10, db_path=temp_db)

    assert len(events) == 1
    assert events[0]["run_id"] == "run_drift_details"
    assert events[0]["feature_name"] == "temperature"
    assert events[0]["severity"] == "CRITICAL"
    assert events[0]["p_value"] == 0.01


def test_pipeline_run_query_bounds_limit_and_returns_run_details(temp_db):
    """Run history query should cap results and expose operational run metrics."""
    for index in range(3):
        log_pipeline_run(
            run_id=f"run_history_{index}",
            dataset_name=f"batch_{index}.csv",
            records_total=10 + index,
            records_valid=9 + index,
            records_quarantined=1,
            records_anomalous=index,
            quality_score=90.0 + index,
            status="Healthy" if index else "Warning",
            execution_time_ms=100.0 + index,
            db_path=temp_db,
        )

    runs = get_pipeline_runs(limit=2, db_path=temp_db)
    assert len(runs) == 2
    assert {run["run_id"] for run in runs}.issubset(
        {"run_history_0", "run_history_1", "run_history_2"}
    )
    assert {"dataset_name", "records_total", "records_quarantined", "quality_score", "status"}.issubset(runs[0])

    bounded_runs = get_pipeline_runs(limit=0, db_path=temp_db)
    assert len(bounded_runs) == 1


def test_pipeline_run_query_filters_by_status(temp_db):
    """Run history status filter should return only matching runs."""
    for run_id, status in [
        ("run_filter_healthy", "Healthy"),
        ("run_filter_warning", "Warning"),
        ("run_filter_error", "Error"),
    ]:
        log_pipeline_run(
            run_id=run_id,
            dataset_name=f"{run_id}.csv",
            records_total=5,
            records_valid=5,
            records_quarantined=0,
            records_anomalous=0,
            quality_score=100.0,
            status=status,
            execution_time_ms=25.0,
            db_path=temp_db,
        )

    warnings = get_pipeline_runs(limit=20, db_path=temp_db, status="Warning")
    assert len(warnings) == 1
    assert warnings[0]["run_id"] == "run_filter_warning"
    assert warnings[0]["status"] == "Warning"

    all_runs = get_pipeline_runs(limit=20, db_path=temp_db, status="ALL")
    assert len(all_runs) == 3



def test_pipeline_run_query_searches_run_id_or_dataset_and_combines_status(temp_db):
    """Run history search should match run IDs or dataset names and combine with status."""
    for run_id, dataset_name, status in [
        ("run_customer_import", "customers.csv", "Healthy"),
        ("run_orders_import", "orders.csv", "Warning"),
        ("run_customer_retry", "retry_customers.csv", "Warning"),
    ]:
        log_pipeline_run(
            run_id=run_id,
            dataset_name=dataset_name,
            records_total=5,
            records_valid=5,
            records_quarantined=0,
            records_anomalous=0,
            quality_score=100.0,
            status=status,
            execution_time_ms=25.0,
            db_path=temp_db,
        )

    dataset_matches = get_pipeline_runs(limit=20, db_path=temp_db, search="orders.csv")
    assert {run["run_id"] for run in dataset_matches} == {"run_orders_import"}

    run_id_matches = get_pipeline_runs(limit=20, db_path=temp_db, search="customer")
    assert {run["run_id"] for run in run_id_matches} == {
        "run_customer_import",
        "run_customer_retry",
    }

    filtered_matches = get_pipeline_runs(
        limit=20, db_path=temp_db, status="Warning", search="customer"
    )
    assert {run["run_id"] for run in filtered_matches} == {"run_customer_retry"}
