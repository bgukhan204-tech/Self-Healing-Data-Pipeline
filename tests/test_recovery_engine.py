"""Tests for Autonomous Recovery Engine: Retries, Cooldown, Quarantine, and Rollback."""
from datetime import datetime, timezone
import pytest
from src.database import get_active_model, init_db, log_recovery_action
from src.ingestion import generate_synthetic_dataset
from src.recovery_engine import RecoveryActionType, RecoveryEngine, RecoveryTrigger


@pytest.fixture
def recovery_env(tmp_path, monkeypatch):
    """Setup isolated db and environment for recovery engine."""
    from config.settings import get_settings
    db_file = tmp_path / "recovery_test.db"
    model_dir = tmp_path / "recovery_models"
    init_db(db_file)
    
    settings = get_settings()
    monkeypatch.setattr(settings, "database_path", db_file)
    monkeypatch.setattr(settings, "model_dir", model_dir)
    monkeypatch.setattr(settings, "max_retry_attempts", 3)
    monkeypatch.setattr(settings, "retry_backoff_seconds", 0.05)
    monkeypatch.setattr(settings, "recovery_cooldown_seconds", 5)
    engine = RecoveryEngine()
    return engine, db_file


def test_bounded_retry_success(recovery_env):
    """Verify that bounded retry succeeds when transient failure resolves."""
    engine, db_file = recovery_env
    attempts_tracker = {"count": 0}

    def flaky_operation():
        attempts_tracker["count"] += 1
        if attempts_tracker["count"] < 2:
            raise ConnectionError("Simulated temporary network blip")
        return "SUCCESSFUL_PAYLOAD"

    success, result, exec_res = engine.execute_bounded_retry(
        operation=flaky_operation,
        operation_name="FlakyAPI",
        max_attempts=3,
        backoff_sec=0.01,
    )
    assert success is True
    assert result == "SUCCESSFUL_PAYLOAD"
    assert exec_res.attempts == 2
    assert exec_res.status == "SUCCESS"


def test_bounded_retry_exhaustion(recovery_env):
    """Verify that bounded retry fails cleanly after reaching maximum attempts."""
    engine, db_file = recovery_env

    def always_failing_operation():
        raise TimeoutError("Simulated permanent timeout")

    success, result, exec_res = engine.execute_bounded_retry(
        operation=always_failing_operation,
        operation_name="FailingDB",
        max_attempts=3,
        backoff_sec=0.01,
    )
    assert success is False
    assert result is None
    assert exec_res.attempts == 3
    assert exec_res.status == "FAILED"
    assert "permanent timeout" in exec_res.error_details


def test_recovery_cooldown(recovery_env):
    """Verify that recovery cooldown blocks repeated action triggers within the window."""
    engine, db_file = recovery_env
    action = RecoveryActionType.TRIGGER_RETRAINING.value

    # Initially not cooling down
    cooling, _ = engine.check_cooldown(action)
    assert cooling is False

    # Log successful recovery action
    log_recovery_action(
        trigger_event=RecoveryTrigger.DATA_DRIFT_DETECTED.value,
        action_type=action,
        status="SUCCESS",
        outcome_summary="Retrained v1.1.0",
        db_path=db_file,
    )

    # Now cooldown should be active
    cooling_now, remaining = engine.check_cooldown(action)
    assert cooling_now is True
    assert remaining > 0


def test_model_rollback(recovery_env):
    """Verify rolling back to a previously validated model artifact."""
    engine, db_file = recovery_env
    df = generate_synthetic_dataset(n_samples=300, random_seed=42)

    # 1. Register v1.0.0 (active)
    p1, _, meta1 = engine.registry.validate_and_register_candidate(df, algorithm="RandomForest")
    assert meta1["version"] == "v1.0.0"

    # 2. Register v1.1.0 (new active)
    p2, _, meta2 = engine.registry.validate_and_register_candidate(df, algorithm="RandomForest")
    assert meta2["version"] == "v1.1.0"
    assert get_active_model(db_file)["model_version"] == "v1.1.0"

    # 3. Trigger rollback to v1.0.0
    res = engine.execute_rollback(target_version="v1.0.0")
    assert res.status == "SUCCESS"
    assert get_active_model(db_file)["model_version"] == "v1.0.0"
