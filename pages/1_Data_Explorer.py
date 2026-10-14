"""Page 1: Data Explorer & Quality Validation Engine."""
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Explorer & Validation", page_icon="🔍", layout="wide")

from config.settings import get_settings
from src.ingestion import load_csv_safely, save_uploaded_batch
from src.monitoring import PipelineOrchestrator
from src.ui_components import render_header, render_metric_card
from src.validation import validate_dataset

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()

orchestrator: PipelineOrchestrator = st.session_state.orchestrator
settings = get_settings()

render_header(
    title="🔍 Ingestion & Data Quality Validation Explorer",
    subtitle="Inspect dataset structures, enforce schema constraints, audit quality scores, and isolate quarantined records.",
    current_status="Healthy",
)

# Tabs: 1. Upload & Live Validate, 2. Bundled Samples Explorer, 3. Quarantine Storage Viewer
tab1, tab2, tab3 = st.tabs(["📤 Upload New Dataset", "📁 Bundled Datasets", "🛡️ Quarantined Records Archive"])

with tab1:
    st.markdown("#### Upload Incoming CSV Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file (Max 25 MB)",
        type=["csv"],
        help="Upload CSV files for schema verification, missingness audits, and constraint validation.",
    )

    df_to_analyze = None
    dataset_name = "uploaded_batch.csv"

    if uploaded_file is not None:
        raw_df, err = load_csv_safely(uploaded_file)
        if err:
            st.error(f"❌ Upload Error: {err}")
        else:
            df_to_analyze = raw_df
            dataset_name = uploaded_file.name
            st.success(f"File '{uploaded_file.name}' loaded successfully ({len(raw_df):,} rows, {raw_df.shape[1]} columns).")

    if df_to_analyze is not None:
        report = validate_dataset(df_to_analyze, dataset_name=dataset_name)

        # KPI Summary Cards
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            render_metric_card("Total Rows", f"{report.total_records:,}")
        with q2:
            render_metric_card(
                "Data Quality Score",
                f"{report.quality_score:.1f}%",
                is_positive_delta=(report.quality_score >= 80.0),
                delta="Passed" if report.is_valid else "Degraded",
            )
        with q3:
            render_metric_card("Valid Records", f"{report.valid_records_count:,}", subtitle="Passed constraints")
        with q4:
            render_metric_card(
                "Quarantined Records",
                f"{report.quarantined_records_count:,}",
                subtitle=f"{report.duplicate_count} duplicates flagged",
                delta=f"{report.quarantined_records_count} isolated",
                is_positive_delta=(report.quarantined_records_count == 0),
            )

        st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:14px 0;'>", unsafe_allow_html=True)

        col_left, col_right = st.columns([0.65, 0.35])
        with col_left:
            st.markdown("##### 📋 Dataset Preview (First 10 Rows)")
            st.dataframe(df_to_analyze.head(10), use_container_width=True)

        with col_right:
            st.markdown("##### 📊 Column Health & Types")
            st.dataframe(report.column_summary, use_container_width=True, hide_index=True)

        # Flagged issues breakdown
        if report.quarantined_records_count > 0:
            st.markdown("##### 🚨 Flagged Issues & Reasons")
            st.dataframe(report.quarantine_reasons_df, use_container_width=True, hide_index=True)

            # Download Quarantine Report
            csv_buf = io.StringIO()
            report.quarantined_df.to_csv(csv_buf, index=True)
            st.download_button(
                label="📥 Download Quarantined Records (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"quarantined_{dataset_name}",
                mime="text/csv",
                use_container_width=False,
            )

        # Action: Process through full pipeline
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Process Valid Subset Through Full Pipeline", type="primary", use_container_width=True):
            with st.spinner("Processing through anomaly detection, drift monitoring, and recovery engine..."):
                res = orchestrator.process_incoming_batch(df_to_analyze, dataset_name=dataset_name)
                st.success(f"Pipeline executed! Run ID: {res.run_id} | Status: {res.status}")
                st.info(f"Anomalies: {res.records_anomalous} | Quarantined: {res.records_quarantined} | Drift: {res.drift_report.overall_drift_status if res.drift_report else 'N/A'}")

with tab2:
    st.markdown("#### Explore Bundled Demonstration Datasets")
    sample_options = {
        "Clean Production Batch": "sample_clean_batch.csv",
        "Corrupted Ingestion Batch (Outliers + Missing)": "sample_corrupted_batch.csv",
        "Drifted Batch (Distribution Shift)": "sample_drifted_batch.csv",
        "Reference Baseline Dataset": "sample_reference.csv",
    }
    selected_sample = st.selectbox("Select a bundled dataset to inspect:", list(sample_options.keys()))
    file_rel = sample_options[selected_sample]
    file_path = settings.base_dir / "data" / file_rel

    if file_path.exists():
        sample_df = pd.read_csv(file_path)
        sample_report = validate_dataset(sample_df, dataset_name=file_rel)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Dataset Size", f"{len(sample_df):,} rows")
        with c2:
            render_metric_card("Quality Score", f"{sample_report.quality_score:.1f}%")
        with c3:
            render_metric_card("Valid Count", f"{sample_report.valid_records_count:,}")
        with c4:
            render_metric_card("Quarantined", f"{sample_report.quarantined_records_count:,}")

        st.dataframe(sample_df.head(15), use_container_width=True)

        if st.button(f"⚡ Ingest & Process '{selected_sample}'", key="ingest_bundled"):
            with st.spinner("Executing end-to-end pipeline run..."):
                res = orchestrator.process_incoming_batch(sample_df, dataset_name=file_rel)
                st.success(f"Processed successfully! Status: {res.status} | Run ID: {res.run_id}")
                st.rerun()

with tab3:
    st.markdown("#### Quarantined Records Archive")
    st.caption("All rows rejected by data validation rules are stored with isolation timestamps for compliance auditing.")
    
    quar_files = list(settings.quarantine_dir.glob("quarantined_*.csv"))
    if quar_files:
        chosen_q_file = st.selectbox("Select quarantined batch file:", [f.name for f in quar_files])
        target_q_path = settings.quarantine_dir / chosen_q_file
        q_view_df = pd.read_csv(target_q_path)
        
        st.markdown(f"**Found {len(q_view_df)} quarantined records in `{chosen_q_file}`:**")
        st.dataframe(q_view_df, use_container_width=True)

        # Download button
        with open(target_q_path, "r", encoding="utf-8") as f:
            st.download_button(
                "📥 Download This Quarantine File",
                data=f.read(),
                file_name=chosen_q_file,
                mime="text/csv",
            )
    else:
        st.info("No quarantined records recorded yet. Any invalid data uploaded will be quarantined here.")
