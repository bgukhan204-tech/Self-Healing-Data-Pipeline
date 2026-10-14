"""Page 2: Unsupervised Anomaly Detection & Statistical Baseline Comparison."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Anomaly Detection", page_icon="⚡", layout="wide")

from config.settings import get_settings
from src.anomaly_detection import AnomalyDetector, LIMITATIONS_DISCLAIMER
from src.monitoring import PipelineOrchestrator
from src.preprocessing import NUMERIC_FEATURES
from src.ui_components import get_plotly_layout, render_header, render_metric_card

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()

orchestrator: PipelineOrchestrator = st.session_state.orchestrator
settings = get_settings()

render_header(
    title="⚡ Multivariate Anomaly Detection Engine",
    subtitle="Identify statistical outliers using Scikit-Learn Isolation Forest and univariate Z-score baselines.",
    current_status="Healthy",
)

# Limitations Disclaimer Alert
st.warning(f"ℹ️ **Important ML Principle**: {LIMITATIONS_DISCLAIMER}")

# Sidebar controls
st.sidebar.markdown("### ⚙️ Detector Configuration")
contamination = st.sidebar.slider(
    "Expected Contamination Rate",
    min_value=0.01,
    max_value=0.20,
    value=float(settings.anomaly_contamination),
    step=0.01,
    help="Estimated proportion of outliers in the reference population.",
)

feature_selection = st.sidebar.multiselect(
    "Features Evaluated",
    options=NUMERIC_FEATURES,
    default=NUMERIC_FEATURES,
    help="Numeric features fed into the Isolation Forest multivariate tree ensemble.",
)

# Dataset selection
st.markdown("#### 1. Select Dataset to Score")
dataset_choice = st.selectbox(
    "Choose batch to evaluate:",
    [
        "Bundled Corrupted Ingestion Batch (High Anomaly Rate)",
        "Bundled Clean Production Batch (Normal Rate)",
        "Bundled Reference Baseline Dataset",
    ],
)

file_map = {
    "Bundled Corrupted Ingestion Batch (High Anomaly Rate)": "sample_corrupted_batch.csv",
    "Bundled Clean Production Batch (Normal Rate)": "sample_clean_batch.csv",
    "Bundled Reference Baseline Dataset": "sample_reference.csv",
}

target_path = settings.base_dir / "data" / file_map[dataset_choice]
if target_path.exists():
    eval_df = pd.read_csv(target_path)
    
    # Instantiate and fit detector with current contamination parameter
    detector = AnomalyDetector(
        features=feature_selection,
        contamination=contamination,
        random_seed=settings.random_seed,
    )
    detector.fit(orchestrator.reference_data)

    # Score incoming records
    res = detector.predict(eval_df)

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Evaluated Records", f"{res.total_records:,}")
    with c2:
        render_metric_card(
            "Anomalies Flagged",
            f"{res.total_anomalies:,}",
            subtitle=f"{res.anomaly_percentage:.1f}% anomaly rate",
            is_positive_delta=(res.total_anomalies <= res.total_records * 0.08),
        )
    with c3:
        stat_count = int(res.statistical_baseline_flags.sum())
        render_metric_card("Statistical Z-Score Flags", f"{stat_count:,}", subtitle="Univariate |z| > 3.0")
    with c4:
        both_count = int(np.logical_and(res.is_anomaly, res.statistical_baseline_flags).sum()) if 'np' in locals() else 0
        render_metric_card("Ensemble Agreement", f"{both_count} records", subtitle="Flagged by both models")

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:14px 0;'>", unsafe_allow_html=True)

    # Row 2: Visualizations
    st.markdown("#### 2. Visual Distribution & Outlier Space")
    vis_col1, vis_col2 = st.columns([0.62, 0.38])

    plot_df = eval_df.copy()
    plot_df["Anomaly Label"] = ["Anomaly" if a else "Normal" for a in res.is_anomaly]
    plot_df["Anomaly Score"] = res.anomaly_scores
    plot_df["Primary Deviation"] = res.top_contributing_features

    with vis_col1:
        st.markdown("##### 🌌 Scatter Projection (Income vs Credit Score)")
        x_col = st.selectbox("X-Axis Feature", feature_selection, index=min(1, len(feature_selection)-1))
        y_col = st.selectbox("Y-Axis Feature", feature_selection, index=min(2, len(feature_selection)-1))

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color="Anomaly Label",
            color_discrete_map={"Normal": "#0EA5E9", "Anomaly": "#EF4444"},
            hover_data=["customer_id", "Anomaly Score", "Primary Deviation"],
            size="Anomaly Score",
            size_max=12,
            opacity=0.8,
        )
        fig.update_layout(get_plotly_layout(height=360))
        st.plotly_chart(fig, use_container_width=True)

    with vis_col2:
        st.markdown("##### 📊 Anomaly Score Histogram")
        fig_hist = px.histogram(
            plot_df,
            x="Anomaly Score",
            color="Anomaly Label",
            nbins=30,
            color_discrete_map={"Normal": "#0EA5E9", "Anomaly": "#EF4444"},
            barmode="overlay",
            opacity=0.75,
        )
        fig_hist.update_layout(get_plotly_layout(height=360))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Row 3: Flagged Records Table with Reasons
    st.markdown("#### 3. Flagged Records & Root-Cause Contributing Features")
    if res.total_anomalies > 0:
        st.dataframe(res.anomalous_records_df, use_container_width=True)
    else:
        st.info("No records flagged as anomalies with the selected contamination threshold.")
