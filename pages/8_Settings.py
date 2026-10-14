"""Page 8: System Configuration, Health Diagnostics, and Architecture Reference."""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Settings & Documentation", page_icon="⚙️", layout="wide")

from config.settings import get_settings
from src.database import get_connection
from src.ui_components import render_header, render_metric_card

settings = get_settings()

render_header(
    title="⚙️ Settings, Diagnostics & Technical Reference",
    subtitle="Inspect system health, review environment parameters, and read architectural specifications.",
    current_status="Healthy",
)

tab_health, tab_config, tab_docs = st.tabs(["🩺 System Diagnostics", "⚙️ Active Settings", "📚 Technical Documentation"])

# ------------------------------------------------------------------------------
# TAB 1: Diagnostics
# ------------------------------------------------------------------------------
with tab_health:
    st.markdown("#### Database Table Row Counts & Storage Health")
    
    with get_connection(settings.database_path) as conn:
        cursor = conn.cursor()
        tables = [
            "pipeline_runs", "anomaly_events", "drift_events", 
            "model_registry", "recovery_actions", "system_events"
        ]
        counts = {}
        for t in tables:
            cursor.execute(f"SELECT count(*) FROM {t}")
            counts[t] = cursor.fetchone()[0]

    d1, d2, d3 = st.columns(3)
    with d1:
        render_metric_card("Database Path", settings.database_path.name, subtitle=str(settings.database_path.parent))
    with d2:
        db_size_kb = round(settings.database_path.stat().st_size / 1024.0, 1) if settings.database_path.exists() else 0
        render_metric_card("DB File Size", f"{db_size_kb:,} KB", subtitle="SQLite WAL format")
    with d3:
        render_metric_card("Total Audit Events", f"{counts['system_events']:,} events", subtitle="Immutable compliance log")

    st.markdown("##### Detailed Table Row Counts:")
    st.dataframe(pd.DataFrame([{"Table": k, "Row Count": v} for k, v in counts.items()]), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: Active Settings
# ------------------------------------------------------------------------------
with tab_config:
    st.markdown("#### Pipeline Configuration Parameters")
    
    config_dict = {
        "DATABASE_PATH": str(settings.database_path),
        "REFERENCE_DATA_PATH": str(settings.reference_data_path),
        "MODEL_DIR": str(settings.model_dir),
        "QUARANTINE_DIR": str(settings.quarantine_dir),
        "UPLOADS_DIR": str(settings.uploads_dir),
        "ANOMALY_CONTAMINATION": settings.anomaly_contamination,
        "RANDOM_SEED": settings.random_seed,
        "DRIFT_PVALUE_THRESHOLD": settings.drift_pvalue_threshold,
        "DRIFT_CRITICAL_PVALUE": settings.drift_critical_pvalue,
        "DRIFT_WASSERSTEIN_THRESHOLD": settings.drift_wasserstein_threshold,
        "AUTO_RECOVERY_MODE": settings.auto_recovery_mode,
        "MAX_RETRY_ATTEMPTS": settings.max_retry_attempts,
        "RETRY_BACKOFF_SECONDS": settings.retry_backoff_seconds,
        "RECOVERY_COOLDOWN_SECONDS": settings.recovery_cooldown_seconds,
        "MODEL_MIN_F1_SCORE": settings.model_min_f1_score,
        "MODEL_DEGRADATION_TOLERANCE": settings.model_degradation_tolerance,
        "MAX_UPLOAD_SIZE_MB": settings.max_upload_size_mb,
    }

    st.dataframe(
        pd.DataFrame([{"Parameter": k, "Active Value": str(v)} for k, v in config_dict.items()]),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------------------------
# TAB 3: Beginner-Friendly Technical Reference
# ------------------------------------------------------------------------------
with tab_docs:
    st.markdown("#### Technical Documentation & MLOps Concepts Explained")

    st.markdown("""
### 1. Data Drift vs. Model Performance Degradation
- **Data Drift (Distribution Shift):** Measures whether incoming feature distributions differ statistically from the baseline dataset used during training. We detect this using **Kolmogorov-Smirnov (KS) two-sample tests** for numeric variables and **Chi-Square Goodness-of-Fit tests** for categorical categories.
- **Key Caveat:** Input data drift alone does *not* prove that a model has become inaccurate. A model may remain accurate if the decision boundary is unaffected. Only true outcome ground truth labels can measure actual accuracy drops.

### 2. Evaluation Metrics Explained
- **Accuracy:** The ratio of correct predictions to total observations. Can be misleading in imbalanced datasets.
- **Precision:** Of all records predicted as positive (e.g. loan defaults), what fraction actually defaulted? High precision prevents falsely declining good borrowers.
- **Recall:** Of all actual positive loan defaults, what fraction did the model successfully catch? High recall prevents missing financial credit risk.
- **F1-Score:** The harmonic mean of Precision and Recall ($2 \\times \\frac{P \\times R}{P + R}$). Balances both concerns and serves as our primary governance gate metric.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Measures the model's ranking ability across all classification thresholds.

### 3. Isolation Forest Anomaly Detection
- Builds an ensemble of random decision trees.
- Anomalies require fewer random recursive splits to isolate than normal cluster points, resulting in shorter average path lengths.
- Scores records on a normalized continuous scale and flags the top anomalous records based on the configured contamination rate.

### 4. Self-Healing Decision & Recovery Protocol
1. **Transient Processing Error:** Triggers bounded retry with exponential backoff (e.g. 1s, 2s, 4s). Fails cleanly after 3 attempts.
2. **Invalid Data:** Isolates corrupted rows into quarantine storage without modifying or deleting original files. Valid records continue downstream.
3. **Data Drift:** Triggers candidate model retraining on fresh labeled data, protected by a 60-second cooldown window to prevent retraining thrashing.
4. **Candidate Degradation:** Validates candidate against active model; candidate is rejected if F1 is lower than the active model by more than the degradation tolerance (2%).
5. **Deployment / Model Failure:** Executes automated rollback to the last validated healthy model version in the SQLite registry.
""")
