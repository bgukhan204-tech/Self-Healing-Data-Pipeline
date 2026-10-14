"""Regression tests for persistence of monitoring errors to the configured database."""
import pytest

from config.settings import get_settings
from src.database import get_system_events, init_db
from src.drift_detection import DriftMonitor
from src.ingestion import generate_synthetic_dataset
from src.monitoring import PipelineOrchestrator


def test_drift_monitoring_failure_is_audited_in_configured_database(tmp_path, monkeypatch):
    settings = get_settings()
    db_file = tmp_path / "monitoring_error.db"
    monkeypatch.setattr(settings, "database_path", db_file)
    monkeypatch.setattr(settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "reference_data_path", tmp_path / "reference.csv")
    monkeypatch.setattr(settings, "quarantine_dir", tmp_path / "quarantine")
    init_db(db_file)

    orchestrator = PipelineOrchestrator()

    def fail_drift(self, incoming_df):
        raise RuntimeError("simulated drift monitor outage")

    monkeypatch.setattr(DriftMonitor, "evaluate_drift", fail_drift)
    result = orchestrator.process_incoming_batch(
        generate_synthetic_dataset(n_samples=80, random_seed=912),
        dataset_name="drift_monitor_failure.csv",
    )

    assert result.status == "Warning"
    errors = get_system_events(severity="ERROR", limit=20, db_path=db_file)
    assert any(
        event["event_type"] == "DRIFT_DETECTION_ERROR"
        and "simulated drift monitor outage" in event["description"]
        for event in errors
    )
