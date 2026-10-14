"""Boundary tests for bounded retry configuration and backoff behavior."""
import pytest

import src.recovery_engine as recovery_module
from src.recovery_engine import RecoveryEngine


@pytest.fixture
def engine(monkeypatch):
    instance = RecoveryEngine()
    monkeypatch.setattr(
        recovery_module, "log_recovery_action", lambda **kwargs: "test-action"
    )
    monkeypatch.setattr(recovery_module, "log_system_event", lambda **kwargs: None)
    return instance


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_bounded_retry_rejects_non_positive_attempt_count(engine, max_attempts):
    called = {"operation": False}

    def operation():
        called["operation"] = True

    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        engine.execute_bounded_retry(operation, max_attempts=max_attempts)

    assert called["operation"] is False


@pytest.mark.parametrize("backoff_sec", [-0.1, -1])
def test_bounded_retry_rejects_negative_backoff(engine, backoff_sec):
    with pytest.raises(ValueError, match="backoff_sec must be non-negative"):
        engine.execute_bounded_retry(lambda: "ok", backoff_sec=backoff_sec)


def test_zero_backoff_is_respected_between_attempts(engine, monkeypatch):
    sleep_calls = []
    operation_calls = {"count": 0}
    monkeypatch.setattr(recovery_module.time, "sleep", sleep_calls.append)

    def flaky_operation():
        operation_calls["count"] += 1
        if operation_calls["count"] == 1:
            raise RuntimeError("temporary failure")
        return "recovered"

    succeeded, value, result = engine.execute_bounded_retry(
        flaky_operation,
        operation_name="test operation",
        max_attempts=2,
        backoff_sec=0,
        run_id="retry-boundary-test",
    )

    assert succeeded is True
    assert value == "recovered"
    assert result.status == "SUCCESS"
    assert result.attempts == 2
    assert sleep_calls == [0]
