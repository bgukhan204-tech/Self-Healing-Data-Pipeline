"""Application settings and environment configuration."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")


@dataclass
class Settings:
    """Centralized configuration for the Self-Healing Data Pipeline."""

    # Project Root
    base_dir: Path = BASE_DIR

    # Storage Paths
    database_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("DATABASE_PATH", "data/pipeline_history.db")
    )
    reference_data_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("REFERENCE_DATA_PATH", "data/sample_reference.csv")
    )
    model_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("MODEL_DIR", "artifacts/models")
    )
    quarantine_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("QUARANTINE_DIR", "data/quarantine")
    )
    uploads_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("UPLOADS_DIR", "data/uploads")
    )

    # Anomaly Detection Defaults
    anomaly_contamination: float = field(
        default_factory=lambda: float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))
    )
    random_seed: int = field(
        default_factory=lambda: int(os.getenv("RANDOM_SEED", "42"))
    )

    # Drift Detection Thresholds
    drift_pvalue_threshold: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_PVALUE_THRESHOLD", "0.05"))
    )
    drift_critical_pvalue: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_CRITICAL_PVALUE", "0.01"))
    )
    drift_wasserstein_threshold: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_WASSERSTEIN_THRESHOLD", "0.15"))
    )

    # Recovery Engine Parameters
    auto_recovery_mode: str = field(
        default_factory=lambda: os.getenv("AUTO_RECOVERY_MODE", "manual").lower()
    )
    max_retry_attempts: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    )
    retry_backoff_seconds: float = field(
        default_factory=lambda: float(os.getenv("RETRY_BACKOFF_SECONDS", "1.0"))
    )
    recovery_cooldown_seconds: int = field(
        default_factory=lambda: int(os.getenv("RECOVERY_COOLDOWN_SECONDS", "60"))
    )

    # Model Evaluation & Governance Criteria
    model_min_f1_score: float = field(
        default_factory=lambda: float(os.getenv("MODEL_MIN_F1_SCORE", "0.65"))
    )
    model_degradation_tolerance: float = field(
        default_factory=lambda: float(os.getenv("MODEL_DEGRADATION_TOLERANCE", "0.02"))
    )

    # Uploads & Safety
    max_upload_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    )

    def ensure_directories(self) -> None:
        """Create required directories safely if they don't exist."""
        for path in [
            self.database_path.parent,
            self.model_dir,
            self.quarantine_dir,
            self.uploads_dir,
            self.reference_data_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


_settings_instance = None


def get_settings() -> Settings:
    """Return a singleton instance of settings with ensured directory paths."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        _settings_instance.ensure_directories()
    return _settings_instance
