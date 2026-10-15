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

# Row 2: Interactive Incident & Manual Recovery Center
st.markdown("#### 3. Incident & Recovery Action Center")

tab_incidents, tab_interventions = st.tabs(["⚡ Pending Incidents & Recommendations", "🛠️ Manual Operator Interventions"])

with tab_incidents:
    latest_run = st.session_state.get("latest_run_result", None)
    pending_decisions = []
    if latest_run and getattr(latest_run, "recovery_decisions", None):
        pending_decisions = latest_run.recovery_decisions

    if pending_decisions:
        st.write(f"**Found {len(pending_decisions)} recovery recommendations from recent run `{latest_run.run_id}`:**")
        for idx, dec in enumerate(pending_decisions):
            trigger_name = dec.trigger.value if hasattr(dec.trigger, "value") else str(dec.trigger)
            action_name = dec.recommended_action.value if hasattr(dec.recommended_action, "value") else str(dec.recommended_action)
            with st.container():
                st.markdown(
                    f"""
                    <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px 16px; margin-bottom:10px;">
                        <span style="font-weight:600; color:#F59E0B;">Trigger: {trigger_name}</span> &rarr; 
                        <span style="font-weight:600; color:#38BDF8;">Recommended Action: {action_name}</span>
                        <div style="color:#94A3B8; font-size:0.85rem; margin-top:4px;">{dec.explanation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                btn_col1, btn_col2 = st.columns([0.3, 0.7])
                with btn_col1:
                    if st.button(f"Approve & Execute: {action_name}", key=f"btn_approve_{idx}", type="primary"):
                        with st.spinner(f"Executing {action_name}..."):
                            rec_res = None
                            if action_name == RecoveryActionType.TRIGGER_RETRAINING.value:
                                rec_res = engine.execute_drift_retraining(
                                    training_df=orchestrator.reference_data,
                                    force=False,
                                    run_id=latest_run.run_id,
                                )
                            elif action_name == RecoveryActionType.ROLLBACK_MODEL.value:
                                rec_res = engine.execute_rollback(run_id=latest_run.run_id)
                            elif action_name == RecoveryActionType.ALERT_AND_HALT.value:
                                rec_res = engine.execute_alert_and_halt(
                                    reason=dec.explanation,
                                    run_id=latest_run.run_id,
                                )
                            else:
                                st.info(f"Action '{action_name}' does not require manual dispatch.")

                            if rec_res:
                                st.session_state.last_manual_recovery_res = rec_res
                                st.rerun()
    else:
        st.info("No unhandled incidents or pending recommendations currently requiring manual approval.")

    if "last_manual_recovery_res" in st.session_state:
        m_res = st.session_state.last_manual_recovery_res
        v_color = "#10B981" if m_res.verification_passed else "#EF4444"
        v_icon = "✅" if m_res.verification_passed else "❌"
        st.markdown(
            f"""
            <div style="background:rgba(16, 185, 129, 0.08); border:1px solid {v_color}; border-radius:8px; padding:12px 16px; margin-top:14px;">
                <div style="font-weight:600; color:{v_color};">{v_icon} Post-Recovery Verification: {'PASSED' if m_res.verification_passed else 'FAILED'}</div>
                <div style="font-size:0.88rem; color:#CBD5E1; margin-top:4px;"><b>Action:</b> {m_res.action_type} (Status: {m_res.status})</div>
                <div style="font-size:0.85rem; color:#94A3B8; margin-top:2px;"><b>Verification Details:</b> {m_res.verification_details or 'N/A'}</div>
                <div style="font-size:0.85rem; color:#94A3B8; margin-top:2px;"><b>Outcome:</b> {m_res.outcome_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_interventions:
    st.write("Execute ad-hoc emergency self-healing procedures under explicit operator authority:")
    int_col1, int_col2, int_col3 = st.columns(3)
    
    with int_col1:
        st.markdown("##### 🔄 Candidate Retraining")
        st.caption("Train a fresh candidate model on verified reference data.")
        force_retrain = st.checkbox("Force override cooldown", key="force_retrain_chk")
        if st.button("Trigger Retraining", key="btn_manual_retrain", type="primary"):
            with st.spinner("Training and verifying candidate model..."):
                m_res = engine.execute_drift_retraining(
                    training_df=orchestrator.reference_data,
                    force=force_retrain,
                    run_id="manual_operator_trigger",
                )
                st.session_state.last_manual_recovery_res = m_res
                st.rerun()

    with int_col2:
        st.markdown("##### ⏪ Model Rollback")
        st.caption("Revert active model to the previously validated version.")
        confirm_rb = st.checkbox("Confirm rollback authorization", key="confirm_rb_chk")
        if st.button("Execute Rollback", key="btn_manual_rollback", disabled=not confirm_rb):
            with st.spinner("Rolling back model and verifying artifact..."):
                m_res = engine.execute_rollback(run_id="manual_operator_trigger")
                st.session_state.last_manual_recovery_res = m_res
                st.rerun()

    with int_col3:
        st.markdown("##### 🛑 Safety Alert & Halt")
        st.caption("Trigger an immutable safety shutdown event in audit logs.")
        halt_reason = st.text_input("Incident Reason", value="Critical schema corruption or upstream anomaly", key="halt_reason_txt")
        if st.button("Halt & Log Critical Alert", key="btn_manual_halt"):
            with st.spinner("Logging critical alert..."):
                m_res = engine.execute_alert_and_halt(
                    reason=halt_reason,
                    run_id="manual_operator_trigger",
                )
                st.session_state.last_manual_recovery_res = m_res
                st.rerun()

st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:18px 0;'>", unsafe_allow_html=True)

# Row 3: Supported Recovery Policy Matrix
st.markdown("#### 4. Standardized Recovery Policies")

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
        "Safety Mechanism": "Verifies target artifact integrity and loads model sanity check before switching active state.",
    },
    {
        "Trigger Event": "Critical Pipeline Fault",
        "Action Type": "ALERT_AND_HALT",
        "Behavior": "Safely halts execution and logs immutable critical audit telemetry for human intervention.",
        "Safety Mechanism": "Prevents downstream data corruption and halts further automated state changes.",
    },
]
st.dataframe(pd.DataFrame(policies), use_container_width=True, hide_index=True)

# Row 4: Recovery Actions Audit History
st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:18px 0;'>", unsafe_allow_html=True)
st.markdown("#### 5. Audit Trail of Executed Recovery Actions")

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
