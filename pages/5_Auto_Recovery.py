"""Page 5: Autonomous Recovery Engine Control & Policy Auditing."""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Auto-Recovery", page_icon="🛡️", layout="wide")

from config.settings import get_settings
from src.database import get_recovery_actions
from src.monitoring import PipelineOrchestrator
from src.recovery_engine import RecoveryActionType, RecoveryEngine, RecoveryTrigger
from src.ui_components import render_header, render_metric_card

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()

orchestrator: PipelineOrchestrator = st.session_state.orchestrator
settings = get_settings()
engine = orchestrator.recovery_engine

render_header(
    title="🛡️ Autonomous Recovery Engine & Safeguards",
    subtitle="Configure automatic vs manual recovery policies, inspect cooldown status, and audit executed recovery decisions.",
    current_status="Healthy",
)

# Row 1: Policy & Engine Controls
col_mode, col_params = st.columns([0.45, 0.55])

with col_mode:
    st.markdown("#### 1. Recovery Mode Selector")
    current_mode = st.radio(
        "Select Operating Mode:",
        ["manual", "automatic"],
        index=0 if settings.auto_recovery_mode == "manual" else 1,
        format_func=lambda x: "🛡️ Manual Approval Mode (Recommendations Require User Confirmation)" if x == "manual" else "⚡ Automatic Recovery Mode (Safe Rules Execute Automatically)",
        help="In manual mode, recovery actions are presented as recommendations. In automatic mode, safe bounded recovery rules execute autonomously.",
    )
    if current_mode != settings.auto_recovery_mode:
        settings.auto_recovery_mode = current_mode
        st.success(f"Recovery mode switched to: **{current_mode.upper()}**")

with col_params:
    st.markdown("#### 2. Loop Protection & Cooldown Diagnostics")
    
    # Check cooldown for retraining
    cooling_retrain, rem_retrain = engine.check_cooldown(RecoveryActionType.TRIGGER_RETRAINING.value)
    cooling_rollback, rem_rollback = engine.check_cooldown(RecoveryActionType.ROLLBACK_MODEL.value)
    
    c1, c2 = st.columns(2)
    with c1:
        if cooling_retrain:
            st.warning(f"⏳ Retraining Cooldown Active: **{rem_retrain}s remaining**")
        else:
            st.success("✅ Retraining Ready (No cooldown)")
    with c2:
        if cooling_rollback:
            st.warning(f"⏳ Rollback Cooldown Active: **{rem_rollback}s remaining**")
        else:
            st.success("✅ Rollback Ready (No cooldown)")

    st.markdown(
        f"""
        <div style="font-size:0.8rem; color:#64748B; margin-top:10px;">
            ● <b>Configured Cooldown Window:</b> {settings.recovery_cooldown_seconds}s | 
            ● <b>Max Retries Allowed:</b> {settings.max_retry_attempts} attempts | 
            ● <b>Backoff Multiplier:</b> {settings.retry_backoff_seconds}s base
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:18px 0;'>", unsafe_allow_html=True)

# Row 2: Supported Recovery Policy Matrix
st.markdown("#### 3. Standardized Recovery Policies")

policies = [
    {
        "Trigger Event": "Temporary Processing Failure",
        "Action Type": "RETRY_WITH_BACKOFF",
        "Behavior": "Bounded retry with exponential backoff up to 3 attempts. Logs each attempt outcome.",
        "Safety Mechanism": "Max retry limit prevents infinite loops; fails cleanly if exhausted.",
    },
    {
        "Trigger Event": "Invalid Incoming Data",
        "Action Type": "QUARANTINE_RECORDS",
        "Behavior": "Isolates invalid rows into secure quarantine CSV storage with reason annotations; continues valid rows.",
        "Safety Mechanism": "Never silently drops or modifies records. Unaltered original and quarantine preserved.",
    },
    {
        "Trigger Event": "Data Drift Detected",
        "Action Type": "TRIGGER_RETRAINING",
        "Behavior": "Evaluates candidate model on fresh labeled data and verifies promotion criteria.",
        "Safety Mechanism": "60-second recovery cooldown prevents retraining thrashing; rejection gate blocks regressions.",
    },
    {
        "Trigger Event": "Candidate Model Degradation",
        "Action Type": "REJECT_CANDIDATE_MODEL",
        "Behavior": "Rejects degraded candidate model; preserves active production model.",
        "Safety Mechanism": "Candidate must meet minimum F1 score and not exceed 2% degradation tolerance.",
    },
    {
        "Trigger Event": "Deployment / Model Failure",
        "Action Type": "ROLLBACK_MODEL",
        "Behavior": "Reverts active pointer to the last known healthy validated model artifact.",
        "Safety Mechanism": "Verifies target artifact integrity before switching active state.",
    },
]
st.dataframe(pd.DataFrame(policies), use_container_width=True, hide_index=True)

# Row 3: Recovery Actions Audit History
st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:18px 0;'>", unsafe_allow_html=True)
st.markdown("#### 4. Audit Trail of Executed Recovery Actions")

rec_history = get_recovery_actions(limit=50, db_path=settings.database_path)
if rec_history:
    rec_table = []
    for r in rec_history:
        ts_short = r["created_at"][:19].replace("T", " ") if r.get("created_at") else "-"
        rec_table.append({
            "Action ID": r["action_id"],
            "Timestamp (UTC)": ts_short,
            "Trigger": r["trigger_event"],
            "Action": r["action_type"],
            "Status": r["status"],
            "Attempts": r["attempts"],
            "Outcome Summary": r["outcome_summary"],
        })
    st.dataframe(pd.DataFrame(rec_table), use_container_width=True, hide_index=True)
else:
    st.info("No recovery actions recorded in database yet.")
