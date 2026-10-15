"""Page 7: System Audit Trail, Event Telemetry, and Compliance Logs."""
import io
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Event Logs & Audit Trail", page_icon="📜", layout="wide")

from config.settings import get_settings
from src.database import get_pipeline_runs, get_recovery_actions, get_system_events
from src.ui_components import render_header, render_metric_card

settings = get_settings()

render_header(
    title="📜 Pipeline Run History & System Audit Trail",
    subtitle="Immutable SQLite-backed compliance log tracking pipeline executions, validation failures, and autonomous recoveries.",
    current_status="Healthy",
)

tab_runs, tab_events = st.tabs(["🚀 Pipeline Run History", "📋 System Audit Trail & Compliance"])

with tab_runs:
    st.markdown("#### End-to-End Pipeline Executions")
    runs = get_pipeline_runs(limit=100, db_path=settings.database_path)
    
    if runs:
        # Filter controls for runs
        r_col1, r_col2, r_col3 = st.columns([0.3, 0.3, 0.4])
        with r_col1:
            status_filter = st.selectbox("Filter Run Status", ["ALL", "Healthy", "Warning", "Recovering", "Failed"])
        with r_col2:
            dataset_filter = st.selectbox("Filter Dataset", ["ALL"] + sorted(list({r["dataset_name"] for r in runs})))
        with r_col3:
            run_search = st.text_input("Search Run ID:", placeholder="e.g. run_a1b2c3d4...")

        filtered_runs = runs
        if status_filter != "ALL":
            filtered_runs = [r for r in filtered_runs if r["status"] == status_filter]
        if dataset_filter != "ALL":
            filtered_runs = [r for r in filtered_runs if r["dataset_name"] == dataset_filter]
        if run_search.strip():
            filtered_runs = [r for r in filtered_runs if run_search.lower() in r["run_id"].lower()]

        # Metrics banner
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Runs Displayed", f"{len(filtered_runs)} / {len(runs)}")
        with m2:
            succ_count = sum(1 for r in filtered_runs if r["status"] == "Healthy")
            render_metric_card("Healthy Runs", f"{succ_count}")
        with m3:
            warn_count = sum(1 for r in filtered_runs if r["status"] in ["Warning", "Recovering"])
            render_metric_card("Warning / Recovering", f"{warn_count}")
        with m4:
            fail_count = sum(1 for r in filtered_runs if r["status"] == "Failed")
            render_metric_card("Failed Runs", f"{fail_count}")

        st.markdown("<br>", unsafe_allow_html=True)
        runs_table = []
        for r in filtered_runs:
            ts_short = r["created_at"][:19].replace("T", " ") if r.get("created_at") else "-"
            runs_table.append({
                "Run ID": r["run_id"],
                "Timestamp (UTC)": ts_short,
                "Dataset": r["dataset_name"],
                "Status": r["status"],
                "Duration (ms)": f"{r['execution_time_ms']:.1f}",
                "Quality Score": f"{r['quality_score']:.1f}%",
                "Total Rows": f"{r['records_total']:,}",
                "Valid Rows": f"{r['records_valid']:,}",
                "Quarantined": f"{r['records_quarantined']:,}",
                "Anomalies": f"{r['records_anomalous']:,}",
            })
        runs_df = pd.DataFrame(runs_table)
        st.dataframe(runs_df, use_container_width=True, hide_index=True)

        # Export Run History CSV
        run_csv_buf = io.StringIO()
        runs_df.to_csv(run_csv_buf, index=False)
        st.download_button(
            "📥 Export Run History (CSV)",
            data=run_csv_buf.getvalue(),
            file_name="pipeline_run_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No pipeline runs recorded in the database yet. Execute a batch from the Data Explorer or Simulation Lab.")

with tab_events:
    st.markdown("#### System Audit Trail & Event Telemetry")
    # Filters Row
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        severity_filter = st.selectbox("Severity Level", ["ALL", "INFO", "WARNING", "ERROR", "CRITICAL"])

    with f_col2:
        all_events = get_system_events(limit=500, db_path=settings.database_path)
        event_types = ["ALL"] + sorted(list({e["event_type"] for e in all_events}))
        event_type_filter = st.selectbox("Event Type", event_types)

    with f_col3:
        components = ["ALL"] + sorted(list({e["component"] for e in all_events}))
        component_filter = st.selectbox("Component", components)

    with f_col4:
        search_query = st.text_input("Search description / version:", placeholder="e.g. v1.0.0, retry...")

    # Filter records
    filtered = all_events
    if severity_filter != "ALL":
        filtered = [e for e in filtered if e["severity"] == severity_filter]
    if event_type_filter != "ALL":
        filtered = [e for e in filtered if e["event_type"] == event_type_filter]
    if component_filter != "ALL":
        filtered = [e for e in filtered if e["component"] == component_filter]
    if search_query.strip():
        q = search_query.lower()
        filtered = [
            e for e in filtered
            if q in e["description"].lower()
            or (e["dataset_or_version"] and q in e["dataset_or_version"].lower())
            or q in e["event_type"].lower()
        ]

    # Summary KPI Header
    st.markdown(f"**Found {len(filtered)} matching audit events:**")

    if filtered:
        events_df = pd.DataFrame(filtered)
        
        # Clean display columns
        display_df = events_df[[
            "timestamp", "severity", "component", "event_type", 
            "description", "dataset_or_version", "action_taken", "outcome"
        ]].copy()
        display_df.columns = [
            "Timestamp (UTC)", "Severity", "Component", "Event Type",
            "Description", "Target Artifact", "Action Taken", "Outcome"
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Detailed event inspector (JSON viewer)
        st.markdown("##### 🔍 Detailed Event Inspector")
        selected_event_id = st.selectbox("Inspect raw JSON payload for Event ID:", [e["event_id"] for e in filtered])
        selected_obj = next((e for e in filtered if e["event_id"] == selected_event_id), None)
        if selected_obj:
            st.json(selected_obj)

        # Export Buttons
        st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:14px 0;'>", unsafe_allow_html=True)
        exp_c1, exp_c2 = st.columns([0.2, 0.8])
        with exp_c1:
            csv_buf = io.StringIO()
            events_df.to_csv(csv_buf, index=False)
            st.download_button(
                "📥 Export Filtered Events (CSV)",
                data=csv_buf.getvalue(),
                file_name="audit_events_export.csv",
                mime="text/csv",
            )
    else:
        st.info("No events match the selected filter criteria.")
