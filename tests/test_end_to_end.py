"""End-to-End integration test covering the complete self-healing pipeline workflow."""
import pytest
from src.database import (
    get_latest_metrics_summary,
    get_pipeline_runs,
    get_recovery_actions,
    get_system_events,
)
from src.ingestion import generate_synthetic_dataset
from src.monitoring import PipelineOrchestrator


def test_full_pipeline_end_to_end(tmp_path, monkeypatch):
    """Test full cycle: initialization, normal batch, corrupted batch, and audit trail."""
    db_file = tmp_path / "e2e_pipeline.db"
    model_dir = tmp_path / "e2e_models"
    ref_path = tmp_path / "e2e_ref.csv"
    quar_dir = tmp_path / "e2e_quarantine"
    
    # 1. Configure settings fixture
    from config.settings import get_settings
    from src.database import init_db
    settings = get_settings()
    monkeypatch.setattr(settings, "database_path", db_file)
    monkeypatch.setattr(settings, "model_dir", model_dir)
    monkeypatch.setattr(settings, "reference_data_path", ref_path)
    monkeypatch.setattr(settings, "quarantine_dir", quar_dir)
    init_db(db_file)
    
    orchestrator = PipelineOrchestrator()
    
    # 2. Process clean batch
    clean_batch = generate_synthetic_dataset(n_samples=150, random_seed=77)
    res_clean = orchestrator.process_incoming_batch(clean_batch, dataset_name="e2e_clean.csv")
    
    if res_clean.status != "Healthy":
        monitoring_errors = get_system_events(
            severity="ERROR",
            limit=20,
            db_path=db_file,
        )
        pytest.fail(
            "Clean batch was expected to be Healthy but got "
            f"{res_clean.status}. Recent system errors: {monitoring_errors}"
        )
    assert res_clean.records_total == 150
    assert res_clean.records_quarantined == 0
    assert res_clean.quality_score >= 90.0

    # 3. Process corrupted batch
    corrupt_batch = generate_synthetic_dataset(
        n_samples=150,
        random_seed=88,
        outlier_ratio=0.10,
        missing_ratio=0.10,
    )
    res_corrupt = orchestrator.process_incoming_batch(corrupt_batch, dataset_name="e2e_corrupt.csv")
    
    assert res_corrupt.records_quarantined > 0
    assert res_corrupt.validation_report.quarantine_file_path is not None
    assert len(res_corrupt.recovery_executions) > 0

    # 4. Check SQLite audit trail and runs
    runs = get_pipeline_runs(db_path=db_file)
    assert len(runs) >= 2

    events = get_system_events(db_path=db_file)
    assert len(events) >= 4

    metrics = get_latest_metrics_summary(db_path=db_file)
    assert metrics["total_runs"] >= 2
    assert metrics["total_records"] == 300
