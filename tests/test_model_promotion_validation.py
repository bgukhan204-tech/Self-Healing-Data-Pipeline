"""Tests for fair, fail-closed model promotion validation."""
from types import SimpleNamespace

import pandas as pd

import src.model_registry as registry_module


class FakeReport:
    def __init__(self, f1_score):
        self.accuracy = f1_score
        self.precision_score = f1_score
        self.recall_score = f1_score
        self.f1_score = f1_score
        self.roc_auc = f1_score
        self.confusion_matrix = [[1, 0], [0, 1]]
        self.sample_count = 4

    def to_dict(self):
        return {"f1_score": self.f1_score}


def _prepare_registry(monkeypatch, active_f1=0.90, evaluation_error=None):
    settings = SimpleNamespace(
        database_path="test.db",
        model_min_f1_score=0.65,
        model_degradation_tolerance=0.02,
        random_seed=42,
    )
    manager = registry_module.ModelRegistryManager()
    monkeypatch.setattr(manager, "settings", settings)
    monkeypatch.setattr(manager, "generate_next_version", lambda: "v2.0.0")
    monkeypatch.setattr(manager, "save_model_artifact", lambda **kwargs: "candidate.joblib")
    monkeypatch.setattr(
        manager,
        "load_model_artifact",
        lambda path: {"model": "active", "preprocessor": "preprocessor"},
    )
    train_df = pd.DataFrame({"feature": [1, 2]})
    val_df = pd.DataFrame({"feature": [3, 4], "target": [0, 1]})
    monkeypatch.setattr(
        registry_module,
        "train_model",
        lambda **kwargs: (object(), object(), FakeReport(0.80), (train_df, val_df)),
    )
    monkeypatch.setattr(
        registry_module,
        "get_active_model",
        lambda **kwargs: {"model_version": "v1.0.0", "artifact_path": "active.joblib", "f1_score": 0.99},
    )
    seen = {}

    def evaluate_active(artifact, evaluation_df):
        seen["evaluation_df"] = evaluation_df
        if evaluation_error:
            raise evaluation_error
        return FakeReport(active_f1)

    monkeypatch.setattr(registry_module, "evaluate_existing_model", evaluate_active)
    monkeypatch.setattr(registry_module, "register_model_run", lambda **kwargs: None)
    monkeypatch.setattr(registry_module, "log_system_event", lambda **kwargs: None)
    return manager, val_df, seen


def test_candidate_compared_with_active_model_on_same_validation_rows(monkeypatch):
    manager, val_df, seen = _prepare_registry(monkeypatch, active_f1=0.90)

    promoted, message, metadata = manager.validate_and_register_candidate(
        pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    )

    assert seen["evaluation_df"] is val_df
    assert promoted is False
    assert metadata["status"] == "rejected"
    assert "same validation data" in metadata["rejection_reason"]


def test_active_model_evaluation_error_blocks_promotion(monkeypatch):
    manager, _, _ = _prepare_registry(
        monkeypatch,
        evaluation_error=RuntimeError("active artifact unavailable"),
    )

    promoted, message, metadata = manager.validate_and_register_candidate(
        pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    )

    assert promoted is False
    assert metadata["status"] == "rejected"
    assert "promotion blocked" in metadata["rejection_reason"]
    assert "active artifact unavailable" in message
