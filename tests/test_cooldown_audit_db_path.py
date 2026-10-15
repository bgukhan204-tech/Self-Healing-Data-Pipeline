"""Regression tests for recovery cooldown audit persistence."""
import sqlite3

import pandas as pd
import pytest

from config.settings import get_settings
from src.database import init_db
from src.recovery_engine import RecoveryEngine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    settings = get_settings()
    db_path = tmp_path / "cooldown_audit.db"
    init_db(db_path)
    monkeypatch.setattr(settings, "database_path", db_path)
    monkeypatch.setattr(settings, "recovery_cooldown_seconds", 60)
    return RecoveryEngine(), db_path


def test_cooldown_block_is_audited_in_configured_database(engine, monkeypatch):
    recovery, db_path = engine
    monkeypatch.setattr(recovery, "check_cooldown", lambda action_type: (True, 23))
    monkeypatch.setattr(
        recovery.registry,
        "validate_and_register_candidate",
        lambda **kwargs: pytest.fail("Retraining must not run during cooldown"),
    )

    training_df = pd.DataFrame({
        "age": list(range(80)),
        "loan_status": [0, 1] * 40,
    })
    result = recovery.execute_drift_retraining(
        training_df=training_df,
        run_id="cooldown-audit-test",
    )

    assert result.status == "COOLED_DOWN"
    assert "23 seconds" in result.outcome_summary
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT status, outcome_summary, run_id
            FROM recovery_actions
            WHERE action_id = ?
            """,
            (result.action_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == "SKIPPED"
    assert "23 seconds" in row[1]
    assert row[2] == "cooldown-audit-test"
