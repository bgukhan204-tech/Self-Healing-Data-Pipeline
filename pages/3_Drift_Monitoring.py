"""Page 3: Data Drift Monitoring with Kolmogorov-Smirnov, Wasserstein, and Chi-Square Tests."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Drift Monitoring", page_icon="🌊", layout="wide")

from config.settings import get_settings
from src.drift_detection import DRIFT_DISCLAIMER, DriftMonitor
from src.monitoring import PipelineOrchestrator
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.ui_components import get_plotly_layout, render_header, render_metric_card

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()

orchestrator: PipelineOrchestrator = st.session_state.orchestrator
settings = get_settings()

render_header(
    title="🌊 Continuous Data Drift Monitoring",
    subtitle="Evaluate population and distribution shifts using two-sample KS-tests, Wasserstein distance, and Chi-Square frequency tests.",
    current_status="Healthy",
)

# Academic / Enterprise MLOps Disclaimer
st.info(f"ℹ️ **Drift Principle**: {DRIFT_DISCLAIMER}")

# Sidebar threshold configuration
st.sidebar.markdown("### 🎚️ Statistical Drift Thresholds")
p_val_thresh = st.sidebar.slider(
    "Warning p-value (alpha)",
    min_value=0.01,
    max_value=0.10,
    value=float(settings.drift_pvalue_threshold),
    step=0.01,
    help="Null hypothesis: distributions are identical. p-value < threshold flags distribution shift.",
)

critical_p_val = st.sidebar.slider(
    "Critical p-value (alpha critical)",
    min_value=0.001,
    max_value=0.02,
    value=float(settings.drift_critical_pvalue),
    step=0.001,
    help="Threshold below which distribution divergence is marked CRITICAL.",
)

# Dataset selection
st.markdown("#### 1. Select Batch to Test Against Baseline Reference")
drift_batch_choice = st.selectbox(
    "Select incoming batch:",
    [
        "Bundled Drifted Batch (Severe Economic Distribution Shift)",
        "Bundled Clean Production Batch (Stable In-Distribution)",
        "Bundled Corrupted Ingestion Batch",
    ],
)

file_map = {
    "Bundled Drifted Batch (Severe Economic Distribution Shift)": "sample_drifted_batch.csv",
    "Bundled Clean Production Batch (Stable In-Distribution)": "sample_clean_batch.csv",
    "Bundled Corrupted Ingestion Batch": "sample_corrupted_batch.csv",
}

target_file = settings.base_dir / "data" / file_map[drift_batch_choice]
if target_file.exists():
    incoming_df = pd.read_csv(target_file)
    ref_df = orchestrator.reference_data

    monitor = DriftMonitor(
        p_val_threshold=p_val_thresh,
        critical_p_val=critical_p_val,
        wasserstein_threshold=settings.drift_wasserstein_threshold,
    )
    monitor.set_reference_data(ref_df)
    drift_report = monitor.evaluate_drift(incoming_df)

    # Metric Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Reference Sample", f"{len(ref_df):,} records", subtitle="Golden baseline distribution")
    with m2:
        render_metric_card("Current Batch Size", f"{len(incoming_df):,} records", subtitle="Incoming evaluation window")
    with m3:
        status_color = True if drift_report.overall_drift_status == "HEALTHY" else False
        render_metric_card(
            "Overall Drift Verdict",
            drift_report.overall_drift_status,
            subtitle=f"{drift_report.drifted_features_count} of {drift_report.total_features_evaluated} features shifted",
            is_positive_delta=status_color,
            delta="Nominal" if status_color else "Shift Detected",
        )
    with m4:
        render_metric_card(
            "Drifted Feature Ratio",
            f"{drift_report.drift_percentage:.1f}%",
            subtitle=f"Warning: p < {p_val_thresh} | Crit: p < {critical_p_val}",
            is_positive_delta=(drift_report.drift_percentage <= 20.0),
        )

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:14px 0;'>", unsafe_allow_html=True)

    # Row 2: Feature-Level Drift Table & Investigation Toolkit
    st.markdown("#### 2. Feature-Level Statistical Drift Investigation")
    
    # Filter and Sort controls
    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
    with filt_col1:
        sev_filter = st.selectbox("Filter Severity:", ["ALL", "CRITICAL", "WARNING", "NONE"], index=0)
    with filt_col2:
        test_filter = st.selectbox("Filter Test Method:", ["ALL", "ks_test", "chi_square"], index=0)
    with filt_col3:
        sort_by = st.selectbox("Sort Features By:", ["Distance Metric (High to Low)", "p-value (Low to High)", "Feature Name"])
    with filt_col4:
        only_drifted = st.checkbox("Show Only Drifted Features", value=False)

    # Apply filters
    display_summary = drift_report.summary_df.copy()
    if sev_filter != "ALL":
        display_summary = display_summary[display_summary["Severity"] == sev_filter]
    if test_filter != "ALL":
        display_summary = display_summary[display_summary["Test"] == test_filter]
    if only_drifted:
        display_summary = display_summary[display_summary["Drift Detected"] == "YES"]

    # Apply sorting
    if sort_by == "Distance Metric (High to Low)":
        display_summary = display_summary.sort_values("Distance Metric", ascending=False)
    elif sort_by == "p-value (Low to High)":
        display_summary = display_summary.sort_values("p-value", ascending=True)
    elif sort_by == "Feature Name":
        display_summary = display_summary.sort_values("Feature", ascending=True)

    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    # Download summary CSV
    exp_col1, exp_col2 = st.columns([0.3, 0.7])
    with exp_col1:
        import io
        drift_csv = io.StringIO()
        display_summary.to_csv(drift_csv, index=False)
        st.download_button(
            label="📥 Download Drift Investigation Report (CSV)",
            data=drift_csv.getvalue(),
            file_name=f"drift_investigation_{file_map[drift_batch_choice]}",
            mime="text/csv",
            use_container_width=True,
        )

    # Row 3: Interactive Distribution Overlay
    st.markdown("#### 3. Distribution Overlay (Reference vs Incoming Batch)")
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    feat_to_plot = st.selectbox("Select feature to visually compare distributions:", all_features)

    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        st.markdown(f"##### 📊 Distribution Histogram: `{feat_to_plot}`")
        fig_dist = go.Figure()

        if feat_to_plot in NUMERIC_FEATURES:
            fig_dist.add_trace(go.Histogram(
                x=ref_df[feat_to_plot],
                name="Reference Golden",
                opacity=0.6,
                marker=dict(color="#0EA5E9"),
                histnorm="probability density",
            ))
            fig_dist.add_trace(go.Histogram(
                x=incoming_df[feat_to_plot],
                name="Current Batch",
                opacity=0.6,
                marker=dict(color="#F59E0B"),
                histnorm="probability density",
            ))
            fig_dist.update_layout(barmode="overlay")
        else:
            ref_counts = ref_df[feat_to_plot].value_counts(normalize=True).reset_index()
            inc_counts = incoming_df[feat_to_plot].value_counts(normalize=True).reset_index()
            ref_counts.columns = ["Category", "Reference Frequency"]
            inc_counts.columns = ["Category", "Current Frequency"]
            cat_merged = pd.merge(ref_counts, inc_counts, on="Category", how="outer").fillna(0)

            fig_dist.add_trace(go.Bar(
                x=cat_merged["Category"],
                y=cat_merged["Reference Frequency"],
                name="Reference Golden",
                marker=dict(color="#0EA5E9"),
            ))
            fig_dist.add_trace(go.Bar(
                x=cat_merged["Category"],
                y=cat_merged["Current Frequency"],
                name="Current Batch",
                marker=dict(color="#F59E0B"),
            ))
            fig_dist.update_layout(barmode="group")

        layout = get_plotly_layout(height=340)
        layout["title"] = f"Empirical Distribution: {feat_to_plot}"
        fig_dist.update_layout(layout)
        st.plotly_chart(fig_dist, use_container_width=True)

    with dist_col2:
        st.markdown(f"##### 📦 Cumulative Distribution / Boxplot: `{feat_to_plot}`")
        if feat_to_plot in NUMERIC_FEATURES:
            comp_df = pd.concat([
                pd.DataFrame({"Dataset": "Reference", feat_to_plot: ref_df[feat_to_plot]}),
                pd.DataFrame({"Dataset": "Current Batch", feat_to_plot: incoming_df[feat_to_plot]}),
            ])
            fig_box = px.box(
                comp_df,
                x="Dataset",
                y=feat_to_plot,
                color="Dataset",
                color_discrete_map={"Reference": "#0EA5E9", "Current Batch": "#F59E0B"},
            )
            fig_box.update_layout(get_plotly_layout(height=340))
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.markdown(
                """
                <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:20px; margin-top:20px;">
                    <h5 style="color:#F8FAFC;">Chi-Square Frequency Shift Assessment</h5>
                    <p style="color:#94A3B8; font-size:0.88rem; line-height:1.5;">
                        Categorical drift assesses frequency divergence between expected baseline proportions 
                        and observed sample counts using the Chi-Square Goodness-of-Fit test alongside 
                        Total Variation Distance (TVD). High TVD indicates category shifts (e.g. surge in high-risk categories).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
