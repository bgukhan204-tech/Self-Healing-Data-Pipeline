"""Tests for Model Training, Evaluation Metrics, Candidate Validation, and Registry Promotion."""
import pytest
from src.database import get_active_model, init_db
from src.ingestion import generate_synthetic_dataset
from src.model_registry import ModelRegistryManager
from src.model_training import train_model


@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    """Setup isolated db and model artifacts dir for registry tests."""
    from config.settings import get_settings
    db_file = tmp_path / "test_reg.db"
    model_dir = tmp_path / "models"
    init_db(db_file)
    
    settings = get_settings()
    monkeypatch.setattr(settings, "database_path", db_file)
    monkeypatch.setattr(settings, "model_dir", model_dir)
    return db_file, model_dir


def test_train_model():
    """Verify standard model training returns accurate report and non-null metrics."""
    df = generate_synthetic_dataset(n_samples=300, random_seed=42)
    model, preproc, report, (train_df, val_df) = train_model(
        df=df,
        algorithm="RandomForest",
        random_seed=42,
    )
    
    assert report.accuracy > 0.60
    assert report.f1_score > 0.50
    assert report.roc_auc > 0.50
    assert len(report.confusion_matrix) == 2
    assert report.sample_count == len(val_df)


def test_candidate_promotion_and_rejection(setup_env):
    """Verify promotion of healthy candidate and rejection of degraded candidate."""
    db_file, model_dir = setup_env
    mgr = ModelRegistryManager()
    df = generate_synthetic_dataset(n_samples=400, random_seed=42)

    # 1. Train good initial candidate -> should be promoted
    promoted, msg, meta = mgr.validate_and_register_candidate(
        df=df,
        algorithm="RandomForest",
        degrade_intentionally=False,
    )
    assert promoted is True
    assert meta["status"] == "active"
    assert get_active_model(db_file)["model_version"] == meta["version"]

    # 2. Train degraded candidate -> should be rejected
    promoted_deg, msg_deg, meta_deg = mgr.validate_and_register_candidate(
        df=df,
        algorithm="RandomForest",
        degrade_intentionally=True,
    )
    assert promoted_deg is False
    assert meta_deg["status"] == "rejected"
    assert meta_deg["rejection_reason"] is not None
    
    # Active model must remain the original good version
    assert get_active_model(db_file)["model_version"] == meta["version"]
