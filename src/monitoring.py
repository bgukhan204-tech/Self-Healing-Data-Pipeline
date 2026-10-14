"""Pipeline Monitoring, End-to-End Orchestration, and System Health Reporting."""
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from config.settings import get_settings
from src.anomaly_detection import AnomalyDetectionResult, AnomalyDetector
from src.database import (
    get_active_model,
    init_db,
    log_anomaly_events,
    log_drift_events,
    log_pipeline_run,
    log_system_event,
)
from src.drift_detection import DatasetDriftReport, DriftMonitor
from src.ingestion import generate_synthetic_dataset, load_csv_safely
from src.model_registry import ModelRegistryManager
from src.recovery_engine import (
    RecoveryDecision,
    RecoveryEngine,
    RecoveryExecutionResult,
)
from src.validation import ValidationReport, validate_dataset


@dataclass
class PipelineRunResult:
    """Complete summary of an end-to-end pipeline run."""
    run_id: str
    dataset_name: str
    status: str  # 'Healthy', 'Warning', 'Recovering', 'Failed'
    records_total: int
    records_valid: int
    records_quarantined: int
    records_anomalous: int
    quality_score: float
    validation_report: ValidationReport
    anomaly_result: Optional[AnomalyDetectionResult]
    drift_report: Optional[DatasetDriftReport]
    recovery_decisions: List[RecoveryDecision]
    recovery_executions: List[RecoveryExecutionResult]
    execution_time_ms: float
    created_at: str


class PipelineOrchestrator:
    """Coordinates ingestion, validation, anomaly detection, drift, recovery, and persistence."""

    def __init__(self):
        self.settings = get_settings()
        init_db(self.settings.database_path)
        self.recovery_engine = RecoveryEngine()
        self.registry = ModelRegistryManager()
        self.anomaly_detector = AnomalyDetector(
            contamination=self.settings.anomaly_contamination,
            random_seed=self.settings.random_seed,
        )
        self.drift_monitor = DriftMonitor(
            p_val_threshold=self.settings.drift_pvalue_threshold,
            critical_p_val=self.settings.drift_critical_pvalue,
            wasserstein_threshold=self.settings.drift_wasserstein_threshold,
        )
        self.reference_data: Optional[pd.DataFrame] = None
        self._initialize_baseline_system()

    def _initialize_baseline_system(self) -> None:
        """Ensure reference dataset, anomaly detector, and baseline model exist."""
        ref_path = self.settings.reference_data_path
        
        # 1. Check or generate reference dataset
        if not ref_path.exists():
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_df = generate_synthetic_dataset(
                n_samples=1200,
                random_seed=self.settings.random_seed,
                drift_factor=0.0,
                outlier_ratio=0.0,
                missing_ratio=0.0,
            )
            ref_df.to_csv(ref_path, index=False)
            self.reference_data = ref_df
        else:
            loaded_df, err = load_csv_safely(ref_path)
            self.reference_data = loaded_df if loaded_df is not None else generate_synthetic_dataset()

        # 2. Fit anomaly detector & drift monitor on reference
        self.anomaly_detector.fit(self.reference_data)
        self.drift_monitor.set_reference_data(self.reference_data)

        # 3. Check or train baseline active model
        active_model = get_active_model()
        if active_model is None:
            self.registry.validate_and_register_candidate(
                df=self.reference_data,
                algorithm="RandomForest",
                degrade_intentionally=False,
                trigger_source="SYSTEM_INITIALIZATION",
            )

    def process_incoming_batch(
        self,
        df: pd.DataFrame,
        dataset_name: str = "incoming_batch.csv",
        auto_recover: Optional[bool] = None,
    ) -> PipelineRunResult:
        """
        Execute full self-healing pipeline workflow on an incoming dataset.
        """
        start_time = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now(timezone.utc).isoformat()
        
        is_auto_mode = (
            auto_recover
            if auto_recover is not None
            else (self.settings.auto_recovery_mode == "automatic")
        )

        recovery_decisions: List[RecoveryDecision] = []
        recovery_executions: List[RecoveryExecutionResult] = []

        log_system_event(
            event_type="PIPELINE_STARTED",
            severity="INFO",
            component="ORCHESTRATOR",
            description=f"Beginning pipeline execution for '{dataset_name}' (Run ID: {run_id}).",
            dataset_or_version=dataset_name,
            db_path=self.settings.database_path,
        )

        # 1. Validation & Quarantine
        val_report = validate_dataset(df, dataset_name=dataset_name)

        # Create the parent run row before anomaly/drift child events are persisted.
        # The final call below updates this row without deleting its child records.
        log_pipeline_run(
            run_id=run_id,
            dataset_name=dataset_name,
            records_total=val_report.total_records,
            records_valid=val_report.valid_records_count,
            records_quarantined=val_report.quarantined_records_count,
            records_anomalous=0,
            quality_score=val_report.quality_score,
            status="Processing",
            execution_time_ms=0.0,
            db_path=self.settings.database_path,
        )

        if val_report.quarantined_records_count > 0:
            quar_exec = self.recovery_engine.execute_quarantine(
                quarantined_df=val_report.quarantined_df,
                quarantine_file_path=val_report.quarantine_file_path,
                run_id=run_id,
            )
            recovery_executions.append(quar_exec)

        records_to_process = val_report.valid_df
        
        # Monitoring failures must affect the final run status.
        monitoring_failed = False

        # 2. Anomaly Detection (Isolation Forest)
        anomaly_res = None
        records_anomalous = 0
        if not records_to_process.empty:
            try:
                anomaly_res = self.anomaly_detector.predict(records_to_process)
                records_anomalous = anomaly_res.total_anomalies
                
                # Format anomalies for DB
                if records_anomalous > 0:
                    anom_payloads = []
                    for idx, row in anomaly_res.anomalous_records_df.iterrows():
                        anom_payloads.append({
                            "record_index": idx,
                            "anomaly_score": float(row["anomaly_score"]),
                            "detector_method": "isolation_forest",
                            "top_contributing_feature": str(row.get("top_contributing_indicator", "univariate_deviation")),
                            "record_payload": row.to_dict(),
                        })
                    log_anomaly_events(
                        run_id=run_id,
                        anomalies=anom_payloads,
                        db_path=self.settings.database_path,
                    )
            except Exception as e:
                monitoring_failed = True
                log_system_event(
                    event_type="ANOMALY_DETECTION_ERROR",
                    severity="ERROR",
                    component="ANOMALY_DETECTOR",
                    description=f"Anomaly detection encountered an error: {str(e)}",
                    error_details=str(e),
                    db_path=self.settings.database_path,
                )

        # 3. Drift Monitoring
        drift_report = None
        drift_status = "HEALTHY"
        if not records_to_process.empty and self.reference_data is not None:
            try:
                drift_report = self.drift_monitor.evaluate_drift(records_to_process)
                drift_status = drift_report.overall_drift_status
                
                # Batch log drift events
                drift_db_items = []
                for rep in drift_report.feature_reports:
                    drift_db_items.append({
                        "feature_name": rep.feature_name,
                        "test_method": rep.test_method,
                        "statistic": rep.statistic,
                        "p_value": rep.p_value,
                        "drift_detected": rep.drift_detected,
                        "severity": rep.severity,
                    })
                log_drift_events(
                    run_id=run_id,
                    drift_results=drift_db_items,
                    db_path=self.settings.database_path,
                )
            except Exception as e:
                monitoring_failed = True
                log_system_event(
                    event_type="DRIFT_DETECTION_ERROR",
                    severity="ERROR",
                    component="DRIFT_MONITOR",
                    description=f"Drift monitoring failed: {str(e)}",
                    error_details=str(e),
                )

        # 4. Decision Engine Evaluation
        decisions = self.recovery_engine.evaluate_pipeline_situation(
            has_processing_error=False,
            quarantined_count=val_report.quarantined_records_count,
            drift_status=drift_status,
        )
        recovery_decisions.extend(decisions)

        # 5. Autonomous Recovery Execution if in Automatic Mode
        if is_auto_mode:
            for d in decisions:
                # Retrain automatically only for critical drift. Warnings remain visible
                # but must not launch a potentially disruptive training cycle.
                if (
                    d.recommended_action == "TRIGGER_RETRAINING"
                    and d.can_auto_execute
                    and drift_status == "CRITICAL"
                ):
                    # Supervised retraining must use incoming labeled data, not the old
                    # reference baseline. RecoveryEngine validates suitability before training.
                    rec_res = self.recovery_engine.execute_drift_retraining(
                        training_df=records_to_process,
                        force=False,
                        run_id=run_id,
                    )
                    recovery_executions.append(rec_res)

        # 6. Overall Status Determination
        if val_report.valid_records_count == 0:
            status = "Failed"
        elif recovery_executions and any(e.status == "FAILED" for e in recovery_executions):
            status = "Warning"
        elif any(
            e.action_type in ["TRIGGER_RETRAINING", "ROLLBACK_MODEL"]
            and e.status == "SUCCESS"
            for e in recovery_executions
        ):
            status = "Recovering"
        elif monitoring_failed:
            status = "Warning"
        elif drift_status in ["WARNING", "CRITICAL"] or val_report.quarantined_records_count > 0:
            status = "Warning"
        else:
            status = "Healthy"

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 7. Record Pipeline Run Summary in SQLite
        log_pipeline_run(
            run_id=run_id,
            dataset_name=dataset_name,
            records_total=val_report.total_records,
            records_valid=val_report.valid_records_count,
            records_quarantined=val_report.quarantined_records_count,
            records_anomalous=records_anomalous,
            quality_score=val_report.quality_score,
            status=status,
            execution_time_ms=elapsed_ms,
            db_path=self.settings.database_path,
        )

        log_system_event(
            event_type="PIPELINE_COMPLETED",
            severity="INFO" if status == "Healthy" else "WARNING",
            component="ORCHESTRATOR",
            description=(
                f"Pipeline completed with status '{status}' in {elapsed_ms} ms. "
                f"Valid: {val_report.valid_records_count}, Quarantined: {val_report.quarantined_records_count}, "
                f"Anomalous: {records_anomalous}, Quality: {val_report.quality_score:.1f}%."
            ),
            dataset_or_version=dataset_name,
            action_taken="RUN_COMPLETED",
            outcome=status,
            db_path=self.settings.database_path,
        )

        return PipelineRunResult(
            run_id=run_id,
            dataset_name=dataset_name,
            status=status,
            records_total=val_report.total_records,
            records_valid=val_report.valid_records_count,
            records_quarantined=val_report.quarantined_records_count,
            records_anomalous=records_anomalous,
            quality_score=val_report.quality_score,
            validation_report=val_report,
            anomaly_result=anomaly_res,
            drift_report=drift_report,
            recovery_decisions=recovery_decisions,
            recovery_executions=recovery_executions,
            execution_time_ms=elapsed_ms,
            created_at=now_str,
        )
