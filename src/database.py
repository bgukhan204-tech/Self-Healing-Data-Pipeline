"""SQLite Database Management for Pipeline Events, Metrics, and Audit Trails."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from config.settings import get_settings


def get_connection(db_path: Optional[Path] = None, auto_init: bool = True) -> sqlite3.Connection:
    """Create and return a configured SQLite connection with row factory."""
    if db_path is None:
        db_path = get_settings().database_path
    elif isinstance(db_path, str):
        db_path = Path(db_path)
    
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency and foreign keys
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")

    # Auto-verify schema exists (avoid recursion if called from init_db)
    if auto_init:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='pipeline_runs';")
        has_table = cursor.fetchone()[0] > 0
        cursor.close()
        if not has_table:
            init_db(db_path)

    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize all SQLite tables and indexes."""
    with get_connection(db_path, auto_init=False) as conn:
        cursor = conn.cursor()
        
        # 1. Pipeline Runs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                records_total INTEGER NOT NULL,
                records_valid INTEGER NOT NULL,
                records_quarantined INTEGER NOT NULL,
                records_anomalous INTEGER NOT NULL,
                quality_score REAL NOT NULL,
                status TEXT NOT NULL,
                execution_time_ms REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Anomaly Events Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                record_index INTEGER NOT NULL,
                anomaly_score REAL NOT NULL,
                detector_method TEXT NOT NULL,
                top_contributing_feature TEXT,
                record_payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id) ON DELETE CASCADE
            );
        """)

        # 3. Drift Events Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                test_method TEXT NOT NULL,
                statistic REAL NOT NULL,
                p_value REAL NOT NULL,
                drift_detected INTEGER NOT NULL,
                severity TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id) ON DELETE CASCADE
            );
        """)

        # 4. Model Registry Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                model_version TEXT PRIMARY KEY,
                algorithm TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                status TEXT NOT NULL, -- 'active', 'candidate', 'rejected', 'rolled_back', 'archived'
                rejection_reason TEXT,
                artifact_path TEXT NOT NULL,
                metadata_json TEXT,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                promoted_at TIMESTAMP
            );
        """)

        # 5. Recovery Actions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recovery_actions (
                action_id TEXT PRIMARY KEY,
                run_id TEXT,
                trigger_event TEXT NOT NULL,
                action_type TEXT NOT NULL,
                status TEXT NOT NULL, -- 'SUCCESS', 'FAILED', 'IN_PROGRESS', 'SKIPPED'
                attempts INTEGER NOT NULL DEFAULT 1,
                outcome_summary TEXT NOT NULL,
                error_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 6. Immutable System Events / Audit Log Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                event_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL, -- 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                component TEXT NOT NULL,
                description TEXT NOT NULL,
                dataset_or_version TEXT,
                action_taken TEXT,
                outcome TEXT,
                error_details TEXT
            );
        """)

        # Indices for optimal query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_runs_created ON pipeline_runs(created_at DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_events_created ON system_events(timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_events_run ON anomaly_events(run_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drift_events_run ON drift_events(run_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recovery_actions_run ON recovery_actions(run_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status);")

        conn.commit()


# ==============================================================================
# LOGGING & WRITE HELPERS
# ==============================================================================

def log_system_event(
    event_type: str,
    severity: str,
    component: str,
    description: str,
    dataset_or_version: Optional[str] = None,
    action_taken: Optional[str] = None,
    outcome: Optional[str] = None,
    error_details: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Log an immutable audit trail event."""
    event_id = f"evt_{uuid.uuid4().hex[:12]}"
    now_utc = datetime.now(timezone.utc).isoformat()
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO system_events (
                event_id, timestamp, event_type, severity, component,
                description, dataset_or_version, action_taken, outcome, error_details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, now_utc, event_type, severity.upper(), component,
            description, dataset_or_version, action_taken, outcome, error_details
        ))
        conn.commit()
    return event_id


def log_pipeline_run(
    run_id: str,
    dataset_name: str,
    records_total: int,
    records_valid: int,
    records_quarantined: int,
    records_anomalous: int,
    quality_score: float,
    status: str,
    execution_time_ms: float,
    db_path: Optional[Path] = None,
) -> None:
    """Insert or update a pipeline run summary record."""
    now_utc = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pipeline_runs (
                run_id, dataset_name, records_total, records_valid, records_quarantined,
                records_anomalous, quality_score, status, execution_time_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                dataset_name = excluded.dataset_name,
                records_total = excluded.records_total,
                records_valid = excluded.records_valid,
                records_quarantined = excluded.records_quarantined,
                records_anomalous = excluded.records_anomalous,
                quality_score = excluded.quality_score,
                status = excluded.status,
                execution_time_ms = excluded.execution_time_ms,
                created_at = excluded.created_at
        """, (
            run_id, dataset_name, records_total, records_valid, records_quarantined,
            records_anomalous, quality_score, status, execution_time_ms, now_utc
        ))
        conn.commit()


def log_anomaly_events(
    run_id: str,
    anomalies: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> None:
    """Batch insert detected anomaly records."""
    if not anomalies:
        return
    now_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in anomalies:
        payload_str = json.dumps(item.get("record_payload", {}))
        rows.append((
            run_id,
            int(item.get("record_index", 0)),
            float(item.get("anomaly_score", 0.0)),
            str(item.get("detector_method", "isolation_forest")),
            str(item.get("top_contributing_feature", "unknown")),
            payload_str,
            now_utc
        ))
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO anomaly_events (
                run_id, record_index, anomaly_score, detector_method,
                top_contributing_feature, record_payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()


def log_drift_events(
    run_id: str,
    drift_results: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> None:
    """Batch insert feature drift evaluation results."""
    if not drift_results:
        return
    now_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in drift_results:
        rows.append((
            run_id,
            str(item.get("feature_name", "")),
            str(item.get("test_method", "")),
            float(item.get("statistic", 0.0)),
            float(item.get("p_value", 1.0)),
            1 if item.get("drift_detected", False) else 0,
            str(item.get("severity", "none")),
            now_utc
        ))
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO drift_events (
                run_id, feature_name, test_method, statistic,
                p_value, drift_detected, severity, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()


def log_recovery_action(
    trigger_event: str,
    action_type: str,
    status: str,
    outcome_summary: str,
    run_id: Optional[str] = None,
    attempts: int = 1,
    error_details: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Record an executed or attempted recovery action."""
    action_id = f"rec_{uuid.uuid4().hex[:12]}"
    now_utc = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO recovery_actions (
                action_id, run_id, trigger_event, action_type, status,
                attempts, outcome_summary, error_details, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            action_id, run_id, trigger_event, action_type, status,
            attempts, outcome_summary, error_details, now_utc
        ))
        conn.commit()
    return action_id


def register_model_run(
    model_version: str,
    algorithm: str,
    accuracy: float,
    precision_score: float,
    recall_score: float,
    f1_score: float,
    status: str,
    artifact_path: str,
    rejection_reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Register a trained model version in the SQLite model registry."""
    now_utc = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata or {})
    promoted_at = now_utc if status == "active" else None
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        # If new model is active, archive any currently active model
        if status == "active":
            cursor.execute("""
                UPDATE model_registry
                SET status = 'archived'
                WHERE status = 'active'
            """)

        cursor.execute("""
            INSERT OR REPLACE INTO model_registry (
                model_version, algorithm, accuracy, precision_score, recall_score,
                f1_score, status, rejection_reason, artifact_path, metadata_json,
                trained_at, promoted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_version, algorithm, accuracy, precision_score, recall_score,
            f1_score, status, rejection_reason, artifact_path, meta_json,
            now_utc, promoted_at
        ))
        conn.commit()


def update_model_status(
    model_version: str,
    new_status: str,
    reason: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Update model status (e.g., promote candidate, rollback, reject)."""
    now_utc = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if new_status == "active":
            # Demote currently active to archived
            cursor.execute("UPDATE model_registry SET status = 'archived' WHERE status = 'active'")
            cursor.execute("""
                UPDATE model_registry
                SET status = ?, promoted_at = ?
                WHERE model_version = ?
            """, (new_status, now_utc, model_version))
        elif new_status == "rolled_back":
            cursor.execute("""
                UPDATE model_registry
                SET status = ?, rejection_reason = ?
                WHERE model_version = ?
            """, (new_status, reason, model_version))
        else:
            cursor.execute("""
                UPDATE model_registry
                SET status = ?, rejection_reason = ?
                WHERE model_version = ?
            """, (new_status, reason, model_version))
        conn.commit()


# ==============================================================================
# READ & QUERY HELPERS
# ==============================================================================

def get_pipeline_runs(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve recent pipeline runs."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT run_id, dataset_name, records_total, records_valid,
                   records_quarantined, records_anomalous, quality_score,
                   status, execution_time_ms, created_at
            FROM pipeline_runs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_system_events(
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve filtered system events for the audit trail."""
    query = "SELECT * FROM system_events WHERE 1=1"
    params: List[Any] = []
    if severity and severity.upper() != "ALL":
        query += " AND severity = ?"
        params.append(severity.upper())
    if event_type and event_type.upper() != "ALL":
        query += " AND event_type = ?"
        params.append(event_type.upper())
    
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_recovery_actions(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve history of automated recovery actions."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM recovery_actions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_model_history(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve registered models ordered by training timestamp."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_version, algorithm, accuracy, precision_score,
                   recall_score, f1_score, status, rejection_reason,
                   artifact_path, metadata_json, trained_at, promoted_at
            FROM model_registry
            ORDER BY trained_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


def get_active_model(db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve the currently active production model if any."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM model_registry
            WHERE status = 'active'
            ORDER BY promoted_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        return dict(row) if row else None


def get_latest_metrics_summary(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Calculate aggregate summary stats for overview cards."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_runs,
                COALESCE(SUM(records_total), 0) as total_records,
                COALESCE(SUM(records_anomalous), 0) as total_anomalies,
                COALESCE(SUM(records_quarantined), 0) as total_quarantined,
                COALESCE(AVG(quality_score), 100.0) as avg_quality_score
            FROM pipeline_runs
        """)
        row = dict(cursor.fetchone() or {})

        cursor.execute("SELECT status FROM pipeline_runs ORDER BY created_at DESC LIMIT 1")
        last_status_row = cursor.fetchone()
        row["current_status"] = last_status_row["status"] if last_status_row else "Healthy"

        cursor.execute("""
            SELECT
                COUNT(*) as total_recoveries,
                COALESCE(SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END), 0) as successful_recoveries,
                COALESCE(SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END), 0) as failed_recoveries
            FROM recovery_actions
        """)
        rec_row = cursor.fetchone()
        total_rec = rec_row["total_recoveries"] if rec_row else 0
        succ_rec = rec_row["successful_recoveries"] if rec_row else 0
        failed_rec = rec_row["failed_recoveries"] if rec_row else 0
        # No recovery attempts is not a 100% success rate; None means not yet measurable.
        row["recovery_success_rate"] = (succ_rec / total_rec * 100.0) if total_rec > 0 else None
        row["total_recoveries"] = total_rec
        row["successful_recoveries"] = succ_rec
        row["failed_recoveries"] = failed_rec

        # Get latest drift count
        cursor.execute("""
            SELECT COUNT(*) as drifted_count 
            FROM drift_events 
            WHERE drift_detected = 1 
              AND created_at >= datetime('now', '-24 hours')
        """)
        drift_row = cursor.fetchone()
        row["drift_events_24h"] = drift_row["drifted_count"] if drift_row else 0

        return row
