"""UI components, glassmorphic styles, and layout elements for Streamlit app."""

import streamlit as st


def inject_custom_css():
    """Inject polished, modern cybersecurity-themed dark styling into Streamlit."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Gradient header banner */
    .hero-header {
        background: linear-gradient(135deg, rgba(14, 28, 54, 0.95) 0%, rgba(20, 45, 85, 0.95) 50%, rgba(10, 60, 90, 0.95) 100%);
        border: 1px solid rgba(0, 210, 255, 0.25);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    .hero-title {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #a0aec0;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Metric card */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 210, 255, 0.4);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #718096;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-delta-pos {
        font-size: 12px;
        font-weight: 600;
        color: #48bb78;
    }
    .metric-delta-neg {
        font-size: 12px;
        font-weight: 600;
        color: #f56565;
    }

    /* Threat badge */
    .badge-attack {
        display: inline-block;
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 13px;
        padding: 6px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 10px rgba(229, 62, 62, 0.4);
    }
    .badge-normal {
        display: inline-block;
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 13px;
        padding: 6px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 10px rgba(56, 161, 105, 0.4);
    }
    .badge-suspicious {
        display: inline-block;
        background: linear-gradient(135deg, #dd6b20 0%, #c05621 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 13px;
        padding: 6px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 10px rgba(221, 107, 32, 0.4);
    }

    /* Section card */
    .content-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    """Render the hero styled header."""
    html = f"""
    <div class="hero-header">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = "", is_positive: bool = True):
    """Render a modern glassmorphic card for headline KPIs."""
    delta_class = "metric-delta-pos" if is_positive else "metric-delta-neg"
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
