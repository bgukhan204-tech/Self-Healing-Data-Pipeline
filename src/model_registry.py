"""Model Registry, Governance, Validation Gate, and Rollback Management."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd
from config.settings import get_settings
from src.database import (
    get_active_model,
    get_connection,
    get_model_history,
    log_system_event,
    register_model_run,
    update_model_status,
)
from src.model_training import ModelEvaluationReport, evaluate_existing_model, train_model


class ModelRegistryManager:
    """Manages model lifecycle, artifact persistence, validation gates, and rollback."""

    def __init__(self):
        self.settings = get_settings()

    def generate_next_version(self) -> str:
        """Generate semantic version increment."""
        history = get_model_history(db_path=self.settings.database_path)
        if not history:
            return "v1.0.0"
        
        # Parse last version
        last_ver = history[0]["model_version"]
        try:
            parts = last_ver.lstrip("v").split(".")
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"v{major}.{minor + 1}.0"
        except Exception:
            return f"v1.{len(history)}.0"

    def save_model_artifact(
        self,
        model: Any,
        preprocessor: Any,
        version: str,
        report: ModelEvaluationReport,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist model bundle artifact securely."""
        self.settings.model_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.settings.model_dir / f"model_{version}.joblib"
        
        payload = {
            "version": version,
            "model": model,
            "preprocessor": preprocessor,
            "report": report.to_dict(),
            "metadata": metadata or {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(payload, artifact_path)
        return artifact_path

    def load_model_artifact(self, artifact_path: str) -> Dict[str, Any]:
        """Safely load and validate model artifact bundle."""
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")
        
        payload = joblib.load(path)
        required_keys = ["version", "model", "preprocessor", "report"]
        for k in required_keys:
            if k not in payload:
                raise ValueError(f"Corrupted model artifact: missing '{k}' key.")
        return payload

    def validate_and_register_candidate(
        self,
        df: pd.DataFrame,
        algorithm: str = "RandomForest",
        degrade_intentionally: bool = False,
        trigger_source: str = "MANUAL",
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Train a candidate model, evaluate against active model, and enforce promotion gates.
        
        Returns:
            (promoted: bool, message: str, candidate_metadata: Dict)
        """
        version = self.generate_next_version()
        
        # 1. Train candidate
        model, preproc, report, (train_df, val_df) = train_model(
            df=df,
            algorithm=algorithm,
            random_seed=self.settings.random_seed,
            degrade_intentionally=degrade_intentionally,
        )

        artifact_path = self.save_model_artifact(
            model=model,
            preprocessor=preproc,
            version=version,
            report=report,
            metadata={"trigger_source": trigger_source, "algorithm": algorithm},
        )

        # 2. Retrieve currently active model to compare
        active_model_info = get_active_model(db_path=self.settings.database_path)
        min_f1 = self.settings.model_min_f1_score
        tolerance = self.settings.model_degradation_tolerance

        is_promoted = False
        rejection_reason = None

        if active_model_info is None:
            # First model: only check minimum F1 threshold
            if report.f1_score >= min_f1:
                is_promoted = True
                status = "active"
                msg = f"Initial baseline model {version} promoted successfully (F1: {report.f1_score:.4f})."
            else:
                is_promoted = False
                status = "rejected"
                rejection_reason = f"Candidate F1 ({report.f1_score:.4f}) below minimum requirement ({min_f1:.4f})."
                msg = f"Candidate model {version} rejected: {rejection_reason}"
        else:
            # Active model exists: compare performance on validation set
            active_f1 = float(active_model_info.get("f1_score", 0.0))
            
            # Acceptance criteria: F1 >= min_f1 AND F1 >= active_f1 - tolerance
            if report.f1_score < min_f1:
                is_promoted = False
                status = "rejected"
                rejection_reason = f"F1-score ({report.f1_score:.4f}) below absolute minimum threshold ({min_f1:.4f})."
                msg = f"Candidate {version} rejected: {rejection_reason}"
            elif report.f1_score < (active_f1 - tolerance):
                is_promoted = False
                status = "rejected"
                rejection_reason = (
                    f"Candidate F1 ({report.f1_score:.4f}) degraded by more than allowed tolerance "
                    f"({tolerance * 100:.1f}%) compared to active model {active_model_info['model_version']} (F1: {active_f1:.4f})."
                )
                msg = f"Candidate {version} rejected: {rejection_reason}"
            else:
                is_promoted = True
                status = "active"
                msg = (
                    f"Candidate model {version} passed validation gates and was promoted to ACTIVE! "
                    f"(Candidate F1: {report.f1_score:.4f} vs Active F1: {active_f1:.4f})"
                )

        # 3. Register in SQLite
        register_model_run(
            model_version=version,
            algorithm=algorithm,
            accuracy=report.accuracy,
            precision_score=report.precision_score,
            recall_score=report.recall_score,
            f1_score=report.f1_score,
            status=status,
            artifact_path=str(artifact_path),
            rejection_reason=rejection_reason,
            metadata={
                "trigger_source": trigger_source,
                "roc_auc": report.roc_auc,
                "confusion_matrix": report.confusion_matrix,
            },
            db_path=self.settings.database_path,
        )

        # 4. Log immutable system event
        event_type = "MODEL_PROMOTED" if is_promoted else "MODEL_REJECTED"
        severity = "INFO" if is_promoted else "WARNING"
        log_system_event(
            event_type=event_type,
            severity=severity,
            component="MODEL_REGISTRY",
            description=msg,
            dataset_or_version=version,
            action_taken="PROMOTION_GATE_EVALUATION",
            outcome="PROMOTED" if is_promoted else "REJECTED",
            error_details=rejection_reason,
            db_path=self.settings.database_path,
        )

        result_meta = {
            "version": version,
            "status": status,
            "report": report.to_dict(),
            "rejection_reason": rejection_reason,
            "artifact_path": str(artifact_path),
        }

        return is_promoted, msg, result_meta

    def rollback_to_version(self, target_version: str) -> Tuple[bool, str]:
        """
        Roll back the active model to a previously validated historical version.
        """
        history = get_model_history(db_path=self.settings.database_path)
        target_info = next((m for m in history if m["model_version"] == target_version), None)
        
        if not target_info:
            return False, f"Target version '{target_version}' not found in model registry."

        # Verify target artifact exists and loads correctly
        try:
            self.load_model_artifact(target_info["artifact_path"])
        except Exception as e:
            return False, f"Cannot rollback: Target model artifact is missing or corrupted: {str(e)}"

        current_active = get_active_model(db_path=self.settings.database_path)
        if current_active and current_active["model_version"] == target_version:
            return True, f"Model {target_version} is already the currently active model."

        # Mark current active as rolled_back
        if current_active:
            update_model_status(
                model_version=current_active["model_version"],
                new_status="rolled_back",
                reason=f"Rolled back in favor of validated version {target_version}",
                db_path=self.settings.database_path,
            )

        # Restore target version to active
        update_model_status(
            model_version=target_version,
            new_status="active",
            reason="Restored via manual/automated rollback governance",
            db_path=self.settings.database_path,
        )

        msg = f"Successfully rolled back active production model to {target_version} (F1: {target_info['f1_score']:.4f})."
        log_system_event(
            event_type="MODEL_ROLLBACK",
            severity="WARNING",
            component="MODEL_REGISTRY",
            description=msg,
            dataset_or_version=target_version,
            action_taken="ROLLBACK_EXECUTED",
            outcome="ACTIVE_RESTORED",
            db_path=self.settings.database_path,
        )
        return True, msg
