"""Regression tests for guarded automatic retraining after drift."""
import pandas as pd
import pytest

from config.settings import get_settings
from src.database import init_db
from src.recovery_engine import RecoveryActionType, RecoveryEngine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    settings = get_settings()
    db_path = tmp_path / "safe_retraining.db"
    init_db(db_path)
    monkeypatch.setattr(settings, "database_path", db_path)
    monkeypatch.setattr(settings, "auto_recovery_mode", "automatic")
    monkeypatch.setattr(settings, "recovery_cooldown_seconds", 60)
    return RecoveryEngine()


def test_warning_drift_does_not_auto_execute_retraining(engine):
    decisions = engine.evaluate_pipeline_situation(drift_status="WARNING")

    retraining = next(
        d for d in decisions
        if d.recommended_action == RecoveryActionType.TRIGGER_RETRAINING
    )
    assert retraining.can_auto_execute is False
    assert "monitored only" in retraining.explanation


def test_critical_drift_can_auto_execute_when_automatic_mode_enabled(engine):
    decisions = engine.evaluate_pipeline_situation(drift_status="CRITICAL")

    retraining = next(
        d for d in decisions
        if d.recommended_action == RecoveryActionType.TRIGGER_RETRAINING
    )
    assert retraining.can_auto_execute is True


def test_retraining_skips_unlabeled_data_and_preserves_model(engine, monkeypatch):
    called = {"train": False}

    def unexpected_train(**kwargs):
        called["train"] = True
        raise AssertionError("Training must not run on unlabeled data")

    monkeypatch.setattr(engine.registry, "validate_and_register_candidate", unexpected_train)
    result = engine.execute_drift_retraining(
        training_df=pd.DataFrame({"age": [25, 30, 40]}),
        run_id="test-run",
    )

    assert result.status == "SKIPPED"
    assert "no 'loan_status' labels" in result.outcome_summary
    assert called["train"] is False


def test_retraining_skips_too_small_or_single_class_data(engine, monkeypatch):
    monkeypatch.setattr(
        engine.registry,
        "validate_and_register_candidate",
        lambda **kwargs: pytest.fail("Candidate training should not run"),
    )
    small_labeled_df = pd.DataFrame({
        "age": list(range(20)),
        "loan_status": [0] * 20,
    })

    result = engine.execute_drift_retraining(
        training_df=small_labeled_df,
        run_id="test-run",
    )

    assert result.status == "SKIPPED"
    assert "at least 40 complete labeled rows" in result.outcome_summary
