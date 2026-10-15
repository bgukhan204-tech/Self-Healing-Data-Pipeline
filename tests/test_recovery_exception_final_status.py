"""End-to-end regression tests for recovery execution exceptions."""
from types import SimpleNamespace

import pandas as pd

from config.settings import get_settings
from src.monitoring import PipelineOrchestrator
from src.recovery_engine import RecoveryActionType, RecoveryDecision, RecoveryTrigger
from src.validation import ValidationReport


def test_retraining_exception_finalizes_run_as_warning(tmp_path, monkeypatch):
    import src.monitoring as monitoring

    settings = get_settings()
    monkeypatch.setattr(settings, "database_path", tmp_path / "pipeline.db")
    monkeypatch.setattr(settings, "auto_recovery_mode", "automatic")

    batch = pd.DataFrame({"age": [30, 31]})
    report = ValidationReport(
        is_valid=True, quality_score=100.0, total_records=2,
        valid_records_count=2, quarantined_records_count=0, duplicate_count=0,
        total_missing_cells=0, flagged_reasons={}, valid_df=batch,
        quarantined_df=pd.DataFrame(), column_summary=pd.DataFrame(),
    )
    monkeypatch.setattr(monitoring, "validate_dataset", lambda *a, **k: report)

    persisted_runs = []
    system_events = []
    recovery_actions = []
    monkeypatch.setattr(monitoring, "init_db", lambda *a, **k: None)
    monkeypatch.setattr(monitoring, "log_pipeline_run", lambda **kw: persisted_runs.append(kw))
    monkeypatch.setattr(monitoring, "log_system_event", lambda **kw: system_events.append(kw))
    monkeypatch.setattr(monitoring, "log_recovery_action", lambda **kw: recovery_actions.append(kw) or "persisted-recovery-id")
    monkeypatch.setattr(monitoring, "log_anomaly_events", lambda **kw: None)
    monkeypatch.setattr(monitoring, "log_drift_events", lambda **kw: None)

    decision = RecoveryDecision(
        trigger=RecoveryTrigger.DATA_DRIFT_DETECTED,
        recommended_action=RecoveryActionType.TRIGGER_RETRAINING,
        can_auto_execute=True,
        explanation="critical drift",
    )

    class FakeRecoveryEngine:
        def evaluate_pipeline_situation(self, **kwargs):
            return [decision]

        def execute_drift_retraining(self, **kwargs):
            raise RuntimeError("registry unavailable")

    class FakeAnomalyDetector:
        def predict(self, df):
            return SimpleNamespace(total_anomalies=0)

    class FakeDriftMonitor:
        def evaluate_drift(self, df):
            return SimpleNamespace(overall_drift_status="CRITICAL", feature_reports=[])

    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orchestrator.settings = settings
    orchestrator.recovery_engine = FakeRecoveryEngine()
    orchestrator.reference_data = batch
    orchestrator.anomaly_detector = FakeAnomalyDetector()
    orchestrator.drift_monitor = FakeDriftMonitor()

    result = orchestrator.process_incoming_batch(batch, dataset_name="failure.csv")

    assert result.status == "Warning"
    assert len(result.recovery_executions) == 1
    assert result.recovery_executions[0].status == "FAILED"
    assert result.recovery_executions[0].error_details == "registry unavailable"
    assert result.recovery_executions[0].action_id == "persisted-recovery-id"
    assert recovery_actions[0]["status"] == "FAILED"
    assert recovery_actions[0]["run_id"] == result.run_id
    assert recovery_actions[0]["db_path"] == settings.database_path
    assert persisted_runs[-1]["status"] == "Warning"
    assert any(event["event_type"] == "RECOVERY_EXECUTION_ERROR" for event in system_events)
