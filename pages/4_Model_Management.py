"""Page 4: Model Governance, Training, Evaluation, and Version Rollback."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Model Management", page_icon="🧠", layout="wide")

from config.settings import get_settings
from src.database import get_active_model, get_model_history
from src.model_registry import ModelRegistryManager
from src.monitoring import PipelineOrchestrator
from src.ui_components import get_plotly_layout, render_header, render_metric_card

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()

orchestrator: PipelineOrchestrator = st.session_state.orchestrator
settings = get_settings()
registry = ModelRegistryManager()

render_header(
    title="🧠 Model Registry, Governance & Rollback",
    subtitle="Track model versions, enforce automated promotion validation gates, and execute instant rollbacks.",
    current_status="Healthy",
)

# Active Model Status & Metric Row
active_model = get_active_model(settings.database_path)
model_history = get_model_history(settings.database_path)

c1, c2, c3, c4 = st.columns(4)
with c1:
    act_ver = active_model["model_version"] if active_model else "None"
    render_metric_card("Active Production Model", act_ver, subtitle=active_model["algorithm"] if active_model else "")
with c2:
    f1_val = f"{active_model['f1_score']:.4f}" if active_model and active_model.get("f1_score") else "N/A"
    render_metric_card("Active F1-Score", f1_val, subtitle="Weighted harmonic mean", is_positive_delta=True)
with c3:
    acc_val = f"{active_model['accuracy'] * 100:.2f}%" if active_model and active_model.get("accuracy") else "N/A"
    render_metric_card("Active Accuracy", acc_val, subtitle="Validation split accuracy")
with c4:
    render_metric_card("Registered Versions", f"{len(model_history)} models", subtitle="In SQLite model registry")

st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:14px 0;'>", unsafe_allow_html=True)

# Main Section: 1. Manual Retrain & Candidate Evaluation, 2. Model History & Rollback Selector
col_left, col_right = st.columns([0.55, 0.45])

with col_left:
    st.markdown("#### 1. Retrain Candidate Model")
    with st.container():
        st.markdown(
            """
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:18px; margin-bottom:14px;">
                <p style="font-size:0.88rem; color:#94A3B8; margin-bottom:10px;">
                    Trigger candidate model training on reference or custom labeled data.
                    The candidate model must pass <b>acceptance criteria</b> (F1 &ge; """ + f"{settings.model_min_f1_score}" + """ and no degradation &gt; """ + f"{settings.model_degradation_tolerance*100:.0f}%" + """) before being promoted to ACTIVE.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        algo_choice = st.selectbox("Select Supervised Algorithm", ["RandomForest", "GradientBoosting"])
        simulate_weak = st.checkbox("🧪 Deliberately train degraded model (To test candidate rejection gate)")

        if st.button("🚀 Train & Evaluate Candidate Model", type="primary", use_container_width=True):
            with st.spinner("Splitting data, training model, and computing validation metrics..."):
                promoted, msg, meta = registry.validate_and_register_candidate(
                    df=orchestrator.reference_data,
                    algorithm=algo_choice,
                    degrade_intentionally=simulate_weak,
                    trigger_source="MANUAL_UI_TRIGGER",
                )
                if promoted:
                    st.success(f"🎉 {msg}")
                else:
                    st.error(f"🛑 {msg}")

                # Show candidate report details
                rep = meta["report"]
                st.markdown("##### Candidate Evaluation Report:")
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("F1-Score", f"{rep['f1_score']:.4f}")
                m_c2.metric("Accuracy", f"{rep['accuracy']:.4f}")
                m_c3.metric("Precision", f"{rep['precision_score']:.4f}")
                m_c4.metric("Recall", f"{rep['recall_score']:.4f}")

                # Confusion Matrix
                cm = rep["confusion_matrix"]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Predicted Good (0)", "Predicted Default (1)"],
                    y=["Actual Good (0)", "Actual Default (1)"],
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False,
                ))
                fig_cm.update_layout(get_plotly_layout(height=260))
                st.plotly_chart(fig_cm, use_container_width=True)

with col_right:
    st.markdown("#### 2. Rollback Active Production Model")
    with st.container():
        st.markdown(
            """
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:18px; margin-bottom:14px;">
                <p style="font-size:0.88rem; color:#94A3B8; margin-bottom:8px;">
                    Restore a previously validated model version instantly. 
                    The current active model will be marked as rolled back, and the selected version promoted.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        valid_rollback_targets = [
            m["model_version"] for m in model_history 
            if m["status"] in ["archived", "active"]
        ]
        
        if valid_rollback_targets:
            target_ver = st.selectbox("Select Target Version to Restore:", valid_rollback_targets)
            confirm_rollback = st.checkbox(f"⚠️ Confirm demoting current active model and restoring {target_ver}", key="chk_confirm_rollback")
            if st.button("⏪ Execute Version Rollback", disabled=not confirm_rollback, use_container_width=True, type="secondary"):
                with st.spinner(f"Validating artifact and rolling back to {target_ver}..."):
                    success, rollback_msg = registry.rollback_to_version(target_ver)
                    if success:
                        st.success(rollback_msg)
                        st.rerun()
                    else:
                        st.error(rollback_msg)
        else:
            st.info("No prior archived models available to roll back to.")

# Section 3: Model Version Registry Table
st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:18px 0;'>", unsafe_allow_html=True)
st.markdown("#### 3. Complete Model Version History & Governance Log")

if model_history:
    hist_table = []
    for m in model_history:
        trained_short = m["trained_at"][:19].replace("T", " ") if m.get("trained_at") else "-"
        hist_table.append({
            "Version": m["model_version"],
            "Status": m["status"].upper(),
            "Algorithm": m["algorithm"],
            "F1-Score": f"{m['f1_score']:.4f}" if m.get("f1_score") else "-",
            "Accuracy": f"{m['accuracy']:.4f}" if m.get("accuracy") else "-",
            "Precision": f"{m['precision_score']:.4f}" if m.get("precision_score") else "-",
            "Recall": f"{m['recall_score']:.4f}" if m.get("recall_score") else "-",
            "Trained At (UTC)": trained_short,
            "Rejection / Rollback Reason": m.get("rejection_reason") or "N/A (Accepted)",
        })
    st.dataframe(pd.DataFrame(hist_table), use_container_width=True, hide_index=True)
else:
    st.info("No model runs recorded in registry yet.")
