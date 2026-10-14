"""Page 6: Failure Simulation Lab - Safe, Reproducible Chaos Testing & Auto-Recovery Demo."""
import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Failure Simulation Lab", page_icon="🔬", layout="wide")

from config.settings import get_settings
from src.database import get_active_model
from src.ingestion import generate_synthetic_dataset
from src.monitoring import PipelineOrchestrator
from src.recovery_engine import RecoveryActionType, RecoveryTrigger
from src.ui_components import render_header, render_metric_card

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()

orchestrator: PipelineOrchestrator = st.session_state.orchestrator
settings = get_settings()

render_header(
    title="🔬 Failure Simulation & Chaos Recovery Lab",
    subtitle="Safely simulate pipeline anomalies, missing data, distribution drift, retry backoffs, and candidate model degradation.",
    current_status="Healthy",
)

st.markdown(
    """
    <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.25); border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
        <span style="font-weight:600; color:#38BDF8;">🧪 Simulation Environment Notice:</span>
        <span style="color:#94A3B8; font-size:0.88rem;">
            All experiments executed below are synthetic simulations designed for architectural demonstration, student presentation, 
            and technical interviews. They generate real SQLite audit records and execute actual application code.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

sim_tabs = st.tabs([
    "1. 📥 Missing Values & Quarantine",
    "2. ⚡ Outlier & Anomaly Injection",
    "3. 🌊 Distribution Drift & Retrain",
    "4. 🔄 Transient Processing Failure & Retry",
    "5. 🛑 Candidate Model Degradation & Rollback",
])

# ------------------------------------------------------------------------------
# TAB 1: Missing Values & Quarantine
# ------------------------------------------------------------------------------
with sim_tabs[0]:
    st.markdown("#### Scenario 1: Missing Field Injection & Isolated Quarantine")
    st.write(
        "Demonstrates automatic quarantine of corrupted records while allowing valid transactions to continue downstream."
    )
    
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        missing_rate = st.slider("Missing Value Ratio", 0.05, 0.40, 0.15, 0.05)
        seed1 = st.number_input("Random Seed", value=101, step=1, key="seed1")
        if st.button("Simulate Ingestion & Auto-Quarantine", key="btn_sim1", type="primary"):
            sim_df = generate_synthetic_dataset(
                n_samples=200,
                random_seed=seed1,
                missing_ratio=missing_rate,
            )
            with st.spinner("Processing corrupted batch through self-healing pipeline..."):
                res = orchestrator.process_incoming_batch(sim_df, dataset_name=f"sim_missing_seed{seed1}.csv")
                st.session_state.sim1_res = res
    with col2:
        if "sim1_res" in st.session_state:
            res = st.session_state.sim1_res
            st.success(f"Execution complete: Status = **{res.status}**")
            st.write(f"- **Total Ingested:** {res.records_total} records")
            st.write(f"- **Valid Processed:** {res.records_valid} records")
            st.write(f"- **Quarantined Records:** {res.records_quarantined} records (Isolated in `{res.validation_report.quarantine_file_path}`)")
            st.write(f"- **Data Quality Score:** {res.quality_score:.1f}%")
            if not res.validation_report.quarantined_df.empty:
                st.markdown("**Quarantined Rows Preview:**")
                st.dataframe(res.validation_report.quarantined_df.head(5), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: Outlier & Anomaly Injection
# ------------------------------------------------------------------------------
with sim_tabs[1]:
    st.markdown("#### Scenario 2: Severe Outlier Injection (Isolation Forest & Z-Score)")
    st.write("Inject extreme values (e.g. human age = 175, credit score = 150) and evaluate detector response.")
    
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        outlier_rate = st.slider("Outlier Injection Rate", 0.05, 0.35, 0.12, 0.01)
        seed2 = st.number_input("Random Seed", value=202, step=1, key="seed2")
        if st.button("Inject Outliers & Score Anomaly Space", key="btn_sim2", type="primary"):
            sim_df = generate_synthetic_dataset(
                n_samples=250,
                random_seed=seed2,
                outlier_ratio=outlier_rate,
            )
            with st.spinner("Scoring with Isolation Forest..."):
                res = orchestrator.process_incoming_batch(sim_df, dataset_name=f"sim_outliers_seed{seed2}.csv")
                st.session_state.sim2_res = res
    with col2:
        if "sim2_res" in st.session_state:
            res = st.session_state.sim2_res
            st.success(f"Analysis complete: Status = **{res.status}**")
            st.write(f"- **Flagged Anomalies:** {res.records_anomalous} of {res.records_valid} valid records")
            if res.anomaly_result and not res.anomaly_result.anomalous_records_df.empty:
                st.markdown("**Top Flagged Anomalies & Contributing Factors:**")
                st.dataframe(
                    res.anomaly_result.anomalous_records_df[
                        ["customer_id", "annual_income", "credit_score", "debt_to_income", "top_contributing_indicator"]
                    ].head(6),
                    use_container_width=True,
                )

# ------------------------------------------------------------------------------
# TAB 3: Distribution Drift & Retraining
# ------------------------------------------------------------------------------
with sim_tabs[2]:
    st.markdown("#### Scenario 3: Real Distribution Drift & Autonomous Retrain Evaluation")
    st.write("Simulates population demographic shift and triggers drift monitoring.")
    
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        drift_factor = st.slider("Distribution Shift Magnitude", 0.2, 1.2, 0.8, 0.1)
        seed3 = st.number_input("Random Seed", value=303, step=1, key="seed3")
        if st.button("Simulate Drift Shift", key="btn_sim3", type="primary"):
            sim_df = generate_synthetic_dataset(
                n_samples=300,
                random_seed=seed3,
                drift_factor=drift_factor,
            )
            with st.spinner("Testing distribution divergence with KS & Chi-Square tests..."):
                res = orchestrator.process_incoming_batch(
                    sim_df,
                    dataset_name=f"sim_drift_factor_{drift_factor}.csv",
                    auto_recover=True,
                )
                st.session_state.sim3_res = res
    with col2:
        if "sim3_res" in st.session_state:
            res = st.session_state.sim3_res
            st.warning(f"Drift evaluation completed! Overall Status: **{res.drift_report.overall_drift_status if res.drift_report else 'N/A'}**")
            if res.drift_report:
                st.write(f"- **Features Shifted:** {res.drift_report.drifted_features_count} / {res.drift_report.total_features_evaluated}")
                st.write(f"- **Recovery Actions Executed:** {len(res.recovery_executions)}")
                for r in res.recovery_executions:
                    st.write(f"  ● `{r.action_type}` &rarr; **{r.status}**: {r.outcome_summary}")
                st.dataframe(res.drift_report.summary_df.head(6), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: Transient Processing Failure & Retry
# ------------------------------------------------------------------------------
with sim_tabs[3]:
    st.markdown("#### Scenario 4: Temporary Processing Failure & Bounded Retry with Exponential Backoff")
    st.write("Demonstrates resilience against transient service drops using bounded retries.")

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        fail_mode = st.radio("Failure Severity:", ["Transient Failure (Recovers on Attempt 2)", "Permanent Failure (Retries Exhausted)"])
        if st.button("Simulate Pipeline Network Glitch", key="btn_sim4", type="primary"):
            tracker = {"attempts": 0}

            def test_operation():
                tracker["attempts"] += 1
                if fail_mode.startswith("Transient"):
                    if tracker["attempts"] < 2:
                        raise ConnectionResetError("Connection reset by peer: DB replica unavailable")
                    return "Batch ingested successfully after transient glitch."
                else:
                    raise TimeoutError("ETL target endpoint timed out after 30000ms.")

            with st.spinner("Executing bounded retry with exponential backoff..."):
                success, val, exec_res = orchestrator.recovery_engine.execute_bounded_retry(
                    operation=test_operation,
                    operation_name="SimulatedIngestionStream",
                    max_attempts=3,
                    backoff_sec=0.2,
                )
                st.session_state.sim4_res = exec_res
    with col2:
        if "sim4_res" in st.session_state:
            exec_res = st.session_state.sim4_res
            if exec_res.status == "SUCCESS":
                st.success(f"✅ Recovery Succeeded! Attempts: {exec_res.attempts}/3")
                st.write(f"**Outcome Summary:** {exec_res.outcome_summary}")
            else:
                st.error(f"❌ Retries Exhausted cleanly. Status: {exec_res.status}")
                st.write(f"**Failure Details:** {exec_res.error_details}")

# ------------------------------------------------------------------------------
# TAB 5: Candidate Degradation & Rollback
# ------------------------------------------------------------------------------
with sim_tabs[4]:
    st.markdown("#### Scenario 5: Candidate Model Degradation Gate & Automatic Rollback")
    st.write("Train an underperforming candidate model to prove that the validation gate rejects it and preserves the active model.")

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.write("Active Model before test:")
        curr_act = get_active_model(settings.database_path)
        st.info(f"Active Version: **{curr_act['model_version'] if curr_act else 'v1.0.0'}** (F1: {curr_act['f1_score']:.4f})")

        if st.button("Train Underperforming Candidate", key="btn_sim5", type="primary"):
            with st.spinner("Training constrained model and evaluating promotion criteria..."):
                promoted, msg, meta = orchestrator.registry.validate_and_register_candidate(
                    df=orchestrator.reference_data,
                    algorithm="RandomForest",
                    degrade_intentionally=True,
                    trigger_source="SIMULATION_LAB",
                )
                st.session_state.sim5_res = (promoted, msg, meta)
    with col2:
        if "sim5_res" in st.session_state:
            promoted, msg, meta = st.session_state.sim5_res
            if not promoted:
                st.error(f"🛑 Candidate {meta['version']} REJECTED by Validation Gate!")
                st.write(f"**Reason:** {meta['rejection_reason']}")
                st.write(f"**Candidate F1:** {meta['report']['f1_score']:.4f}")
                curr_act_after = get_active_model(settings.database_path)
                st.success(f"🛡️ Production Protected: Active model remains **{curr_act_after['model_version']}**.")
            else:
                st.success(f"Candidate passed: {msg}")
