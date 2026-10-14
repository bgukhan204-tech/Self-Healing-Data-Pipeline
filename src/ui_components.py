"""Shared UI Components, Styling, and Plotly Chart Formatters for Streamlit."""
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CUSTOM_CSS = """
<style>
    /* Global font and background enhancements */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Enterprise metric card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 14px;
    }
    .metric-card:hover {
        border-color: rgba(14, 165, 233, 0.4);
        transform: translateY(-2px);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-subtitle {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 6px;
    }
    
    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-healthy {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-recovering {
        background-color: rgba(14, 165, 233, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(14, 165, 233, 0.35);
    }
    .badge-failed {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    
    /* Banner hero card */
    .hero-banner {
        background: radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 50%),
                    linear-gradient(180deg, #111827 0%, #0B0F19 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    
    /* Table styling tweaks */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
"""


def apply_custom_css():
    """Inject unified CSS styling into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, current_status: str = "Healthy"):
    """Render unified page header with real-time status pill."""
    apply_custom_css()
    
    status_lower = current_status.lower()
    badge_class = f"badge-{status_lower}" if status_lower in ["healthy", "warning", "recovering", "failed"] else "badge-healthy"
    
    col1, col2 = st.columns([0.78, 0.22])
    with col1:
        st.markdown(f"## {title}")
        st.markdown(f"<p style='color:#94A3B8; margin-top:-10px; font-size:0.95rem;'>{subtitle}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 8px;">
                <span style="color:#64748B; font-size:0.8rem; margin-right:8px;">PIPELINE STATUS:</span>
                <span class="badge {badge_class}">● {current_status.upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:10px 0 24px 0;'>", unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    is_positive_delta: bool = True,
):
    """Render a modern glassmorphic enterprise metric card."""
    delta_html = ""
    if delta:
        color = "#34D399" if is_positive_delta else "#F87171"
        arrow = "↑" if is_positive_delta else "↓"
        delta_html = f"<span style='color:{color}; font-weight:600; font-size:0.8rem;'>{arrow} {delta}</span>"
        
    sub_text = f"{delta_html} {subtitle or ''}" if delta else (subtitle or "")

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{sub_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_plotly_layout(height: int = 340) -> Dict[str, Any]:
    """Return dark enterprise theme layout dictionary for Plotly charts."""
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.5)",
        margin=dict(l=35, r=20, t=35, b=35),
        font=dict(family="Inter, sans-serif", color="#94A3B8", size=12),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            showline=True,
            linecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            showline=True,
            linecolor="rgba(255,255,255,0.1)",
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#CBD5E1", size=11),
        ),
    )
