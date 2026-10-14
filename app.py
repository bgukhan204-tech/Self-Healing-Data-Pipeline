"""Self-Healing Data Pipeline - Overview Dashboard."""
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Self-Healing Data Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import get_settings
from src.database import (
    get_active_model,
    get_latest_metrics_summary,
    get_model_history,
    get_pipeline_runs,
    get_recent_drift_events,
    get_recent_failed_recoveries,
    get_recovery_actions,
    get_system_events,
)
from src.monitoring import PipelineOrchestrator
from src.ui_components import (
    apply_custom_css,
    get_plotly_layout,
    render_header,
    render_metric_card,
)

# Initialize settings and orchestrator in session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()

orchestrator: PipelineOrchestrator = st.session_state.orchestrator
settings = get_settings()

# Fetch latest database aggregates
summary_stats = get_latest_metrics_summary(settings.database_path)
current_status = summary_stats.get("current_status", "Healthy")
active_model = get_active_model(settings.database_path)

render_header(
    title="🛡️ Autonomous Self-Healing Data Pipeline",
    subtitle="Continuous ingestion monitoring, automated anomaly detection, drift mitigation, and governed auto-recovery.",
    current_status=current_status,
)

# Surface recent recovery failures instead of leaving them only in the audit history.
failed_recoveries_24h = summary_stats.get("failed_recoveries_24h", 0)
if failed_recoveries_24h:
    st.warning(
        f"Recovery attention required: {failed_recoveries_24h} failed recovery "
        f"action(s) recorded in the last 24 hours. Review the recovery outcomes "
        f"and audit trail below."
    )

# Show actionable context for failed recoveries, not just an aggregate count.
if summary_stats.get("failed_recoveries", 0):
    with st.expander("🔎 Inspect recent failed recovery actions", expanded=False):
        failed_actions = get_recent_failed_recoveries(
            limit=20, db_path=settings.database_path
        )
        if failed_actions:
            failed_df = pd.DataFrame(failed_actions)
            failed_df = failed_df.rename(columns={
                "action_id": "Action ID",
                "run_id": "Pipeline Run",
                "trigger_event": "Trigger",
                "action_type": "Recovery Action",
                "attempts": "Attempts",
                "outcome_summary": "Outcome",
                "error_details": "Error Details",
                "created_at": "Recorded At",
            })
            visible_columns = [
                "Recorded At", "Pipeline Run", "Trigger", "Recovery Action",
                "Attempts", "Outcome", "Error Details",
            ]
            st.dataframe(
                failed_df[visible_columns].fillna(""),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No failed recovery actions are currently recorded.")

# Surface recent feature-level drift detections for faster investigation.
drift_events_24h = summary_stats.get("drift_events_24h", 0)
if drift_events_24h:
    st.warning(
        f"Data drift detected: {drift_events_24h} feature-level drift event(s) "
        "recorded in the last 24 hours. Review the affected features and test results."
    )

if drift_events_24h:
    with st.expander("📉 Inspect recent detected drift", expanded=False):
        drift_events = get_recent_drift_events(
            limit=20, db_path=settings.database_path
        )
        if drift_events:
            drift_df = pd.DataFrame(drift_events).rename(columns={
                "run_id": "Pipeline Run",
                "feature_name": "Feature",
                "test_method": "Statistical Test",
                "statistic": "Test Statistic",
                "p_value": "P-value",
                "severity": "Severity",
                "created_at": "Recorded At",
            })
            st.dataframe(
                drift_df[[
                    "Recorded At", "Pipeline Run", "Feature", "Severity",
                    "Statistical Test", "Test Statistic", "P-value",
                ]].fillna(""),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No detected drift events are currently recorded.")

# Provide an operator-friendly run history view with quality and processing outcomes.
with st.expander("🧾 Inspect recent pipeline runs", expanded=False):
    run_filter_col1, run_filter_col2 = st.columns([0.42, 0.58])
    with run_filter_col1:
        run_status_filter = st.selectbox(
            "Filter by run status",
            options=["ALL", "Healthy", "Warning", "Recovering", "Error"],
            index=0,
            key="pipeline_run_status_filter",
        )
    with run_filter_col2:
        run_search = st.text_input(
            "Search run ID or dataset",
            placeholder="e.g. run_123 or customers.csv",
            key="pipeline_run_search",
        )
    recent_runs = get_pipeline_runs(
        limit=20,
        db_path=settings.database_path,
        status=None if run_status_filter == "ALL" else run_status_filter,
        search=run_search,
    )
    if recent_runs:
        runs_df = pd.DataFrame(recent_runs).rename(columns={
            "run_id": "Run ID",
            "dataset_name": "Dataset",
            "records_total": "Total Records",
            "records_valid": "Valid Records",
            "records_quarantined": "Quarantined",
            "records_anomalous": "Anomalies",
            "quality_score": "Quality Score (%)",
            "status": "Status",
            "execution_time_ms": "Duration (ms)",
            "created_at": "Recorded At",
        })
        st.dataframe(
            runs_df[[
                "Recorded At", "Run ID", "Dataset", "Status", "Total Records",
                "Valid Records", "Quarantined", "Anomalies", "Quality Score (%)",
                "Duration (ms)",
            ]].fillna(""),
            use_container_width=True,
            hide_index=True,
        )
        csv_data = runs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered runs (CSV)",
            data=csv_data,
            file_name="pipeline_run_history.csv",
            mime="text/csv",
            key="download_filtered_pipeline_runs",
        )
    else:
        st.caption("No pipeline runs are currently recorded.")

# Hero Description Banner
st.markdown(
    """
    <div class="hero-banner">
        <h4 style="color:#F8FAFC; margin-bottom:8px; font-weight:600;">Autonomous MLOps & Self-Healing Telemetry</h4>
        <p style="color:#94A3B8; font-size:0.9rem; line-height:1.5; margin-bottom:12px;">
            This platform monitors incoming production datasets, validates schema constraints, quarantines corrupted records, 
            detects multivariate anomalies via <b>Isolation Forest</b>, tracks statistical distribution drift via 
            <b>Kolmogorov-Smirnov & Chi-Square tests</b>, and autonomously executes bounded recovery protocols (retries, drift retraining, and candidate model rollback).
        </p>
        <div style="display:flex; gap:16px; flex-wrap:wrap; font-size:0.8rem; color:#64748B;">
            <span>● <b>Engine Mode:</b> <span style="color:#38BDF8;">""" + settings.auto_recovery_mode.upper() + """</span></span>
            <span>● <b>Max Retries:</b> <span style="color:#CBD5E1;">""" + str(settings.max_retry_attempts) + """</span></span>
            <span>● <b>Cooldown:</b> <span style="color:#CBD5E1;">""" + str(settings.recovery_cooldown_seconds) + """s</span></span>
            <span>● <b>Storage:</b> <span style="color:#CBD5E1;">SQLite (WAL Mode)</span></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Row 1: KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

total_records = summary_stats.get("total_records", 0)
total_anomalies = summary_stats.get("total_anomalies", 0)
anom_pct = (total_anomalies / total_records * 100.0) if total_records > 0 else 0.0

with col1:
    render_metric_card(
        title="Total Records Processed",
        value=f"{total_records:,}",
        subtitle=f"Across {summary_stats.get('total_runs', 0)} recorded pipeline runs",
    )

with col2:
    render_metric_card(
        title="Anomalies Detected",
        value=f"{total_anomalies:,}",
        subtitle=f"{anom_pct:.2f}% of processed traffic",
        delta=f"{anom_pct:.1f}% rate",
        is_positive_delta=(anom_pct < 10.0),
    )

with col3:
    avg_score = summary_stats.get("avg_quality_score", 100.0)
    render_metric_card(
        title="Data Quality Score",
        value=f"{avg_score:.1f}%",
        subtitle=f"{summary_stats.get('total_quarantined', 0):,} records quarantined",
        delta="+0.8%" if avg_score >= 90 else "-3.2%",
        is_positive_delta=(avg_score >= 85.0),
    )

with col4:
    active_ver = active_model["model_version"] if active_model else "None"
    active_f1 = f"{active_model['f1_score']:.3f}" if active_model and active_model.get("f1_score") else "N/A"
    rec_rate = summary_stats.get("recovery_success_rate")
    total_recoveries = summary_stats.get("total_recoveries", 0)
    failed_recoveries = summary_stats.get("failed_recoveries", 0)
    recovery_rate_label = f"{rec_rate:.1f}%" if rec_rate is not None else "N/A"
    recovery_delta = f"{rec_rate:.0f}% success" if rec_rate is not None else "No attempts"
    render_metric_card(
        title="Active Model & Recovery",
        value=f"{active_ver} ({active_f1})",
        subtitle=f"Recovery Success Rate: {recovery_rate_label} · {failed_recoveries} failed / {total_recoveries} attempts",
        delta=recovery_delta,
        is_positive_delta=(rec_rate >= 90.0) if rec_rate is not None else False,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Row 2: Interactive Historical Charts
chart_col1, chart_col2 = st.columns([0.60, 0.40])

runs_data = get_pipeline_runs(limit=25, db_path=settings.database_path)
runs_df = pd.DataFrame(runs_data)

with chart_col1:
    st.markdown("##### 📈 Ingestion Quality & Anomaly Volume (Recent Runs)")
    if not runs_df.empty:
        # Reverse to show chronological left-to-right
        chart_df = runs_df.iloc[::-1].copy()
        fig = go.Figure()

        # Quality score line (secondary axis)
        fig.add_trace(go.Scatter(
            x=chart_df["run_id"],
            y=chart_df["quality_score"],
            mode="lines+markers",
            name="Quality Score (%)",
            line=dict(color="#0EA5E9", width=2.5),
            marker=dict(size=6),
            yaxis="y2",
        ))

        # Anomalies detected bar
        fig.add_trace(go.Bar(
            x=chart_df["run_id"],
            y=chart_df["records_anomalous"],
            name="Anomalies Flagged",
            marker=dict(color="rgba(245, 158, 11, 0.75)", line=dict(color="#F59E0B", width=1)),
            yaxis="y",
        ))

        layout = get_plotly_layout(height=320)
        layout["yaxis2"] = dict(
            title="Quality Score (%)",
            overlaying="y",
            side="right",
            range=[0, 105],
            showgrid=False,
            font=dict(color="#0EA5E9"),
        )
        layout["yaxis"]["title"] = "Anomaly Count"
        layout["xaxis"]["title"] = "Pipeline Run ID"
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No pipeline runs recorded yet. Process an incoming dataset or run the Simulation Lab.")

with chart_col2:
    st.markdown("##### 🔄 Recovery Actions & Outcomes")
    rec_actions = get_recovery_actions(limit=50, db_path=settings.database_path)
    if rec_actions:
        rec_df = pd.DataFrame(rec_actions)
        status_counts = rec_df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        color_map = {
            "SUCCESS": "#10B981",
            "FAILED": "#EF4444",
            "SKIPPED": "#64748B",
            "COOLED_DOWN": "#F59E0B",
        }
        fig_pie = px.pie(
            status_counts,
            values="Count",
            names="Status",
            color="Status",
            color_discrete_map=color_map,
            hole=0.55,
        )
        layout_pie = get_plotly_layout(height=320)
        layout_pie["showlegend"] = True
        fig_pie.update_layout(layout_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No recovery actions triggered yet. The system is operating nominally.")

# Row 3: Recent Events and Quick Action Launchpad
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
event_col, launch_col = st.columns([0.68, 0.32])

with event_col:
    st.markdown("##### 📜 Recent Pipeline & Audit Trail Events")
    recent_events = get_system_events(limit=8, db_path=settings.database_path)
    if recent_events:
        evt_table = []
        for e in recent_events:
            ts_short = e["timestamp"].split("T")[-1][:8] if "T" in e["timestamp"] else e["timestamp"][:19]
            evt_table.append({
                "Time": ts_short,
                "Severity": e["severity"],
                "Component": e["component"],
                "Event": e["event_type"],
                "Description": e["description"],
            })
        st.dataframe(pd.DataFrame(evt_table), use_container_width=True, hide_index=True)
    else:
        st.caption("No events recorded yet.")

with launch_col:
    st.markdown("##### ⚡ Quick Launchpad")
    with st.container():
        st.markdown(
            """
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:16px;">
                <p style="font-size:0.85rem; color:#94A3B8; margin-bottom:12px;">
                    Test end-to-end functionality immediately using bundled reference and challenge datasets:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("🧪 Run Clean Production Batch (400 records)", use_container_width=True):
            with st.spinner("Processing clean batch through validation, anomaly, and drift monitors..."):
                sample_path = settings.base_dir / "data" / "sample_clean_batch.csv"
                if sample_path.exists():
                    df = pd.read_csv(sample_path)
                    res = orchestrator.process_incoming_batch(df, dataset_name="sample_clean_batch.csv")
                    st.success(f"Batch processed! Status: {res.status} | Quality: {res.quality_score:.1f}%")
                    st.rerun()

        if st.button("⚠️ Run Corrupted Ingestion Batch (Quarantine Test)", use_container_width=True):
            with st.spinner("Executing pipeline with corrupted payload..."):
                sample_path = settings.base_dir / "data" / "sample_corrupted_batch.csv"
                if sample_path.exists():
                    df = pd.read_csv(sample_path)
                    res = orchestrator.process_incoming_batch(df, dataset_name="sample_corrupted_batch.csv")
                    st.warning(f"Batch processed! Quarantined {res.records_quarantined} records into secure storage.")
                    st.rerun()

        if st.button("🔬 Open Failure Simulation Lab", type="primary", use_container_width=True):
            st.switch_page("pages/6_Simulation_Lab.py")
