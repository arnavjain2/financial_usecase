import streamlit as st
from agents.supervisor_agent import supervisor_graph, RiskState
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# --- DARK/LIGHT MODE TOGGLE ---
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"

with st.sidebar:
    theme = st.selectbox(
        "Choose UI theme:",
        ("🌞 Light Mode", "🌚 Dark Mode"),
        index=0 if st.session_state.theme_mode == "light" else 1
    )
    st.session_state.theme_mode = "light" if theme.startswith("🌞") else "dark"

# --- Theme Colors ---
if st.session_state.theme_mode == "dark":
    BACKGROUND_COLOR = "#101014"
    CARD_COLOR = "#20202a"
    BODY_TEXT_COLOR = "#fff"
    METRIC_LABEL_COLOR = "#c8c8d0"
    PRIMARY_GRAD = "linear-gradient(135deg, #232526 0%, #414345 100%)"
else:
    BACKGROUND_COLOR = "#f8f8fb"
    CARD_COLOR = "#fff"
    BODY_TEXT_COLOR = "#101014"
    METRIC_LABEL_COLOR = "#64748b"
    PRIMARY_GRAD = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"

# --- Custom CSS (Dark/Light Mode Adaptation) ---
CUSTOM_CSS = rf"""
<style>
    body, .stApp {{ background: {BACKGROUND_COLOR} !important; color: {BODY_TEXT_COLOR} !important; }}

    .main {{ background: {BACKGROUND_COLOR}; }}
    .block-container {{ background: {CARD_COLOR}; color: {BODY_TEXT_COLOR}; }}

    h1, h2, h3, h4, h5, h6, label, p, span, div, input, textarea {{
        color: {BODY_TEXT_COLOR} !important;
        font-family: 'Inter', sans-serif;
    }}

    /* Buttons */
    .stButton > button {{
        background: {PRIMARY_GRAD};
        color: #fff !important;
        border-radius: 30px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        border: none;
        font-size: 1rem;
        box-shadow: 0 5px 14px {('rgba(30,34,40,0.2)' if st.session_state.theme_mode=="dark" else 'rgba(102,126,234,0.16)')};
        transition: transform 0.2s;
    }}

    .stButton > button:hover {{
        transform: scale(1.04);
        background: linear-gradient(135deg, #1e293b 10%, #6366f1 100%);
    }}

    /* Cards & Metrics */
    .risk-card, .info-card {{
        border-radius: 16px;
        background: {CARD_COLOR};
        color: {BODY_TEXT_COLOR};
        box-shadow: 0 6px 16px {"rgba(30, 34, 40, 0.16)" if st.session_state.theme_mode == "dark" else "rgba(102, 126, 234, 0.09)"};
        margin-bottom: 1.25rem;
        border: 1px solid {"#393951" if st.session_state.theme_mode == "dark" else "#e3e6ff"};
    }}

    .metric-value {{ font-size: 1.8rem; font-weight: 700; color: #6366f1; }}
    .metric-label {{ font-size: 1rem; color: {METRIC_LABEL_COLOR}; text-transform: uppercase; letter-spacing: 0.5px; }}

    .risk-low {{ background: linear-gradient(90deg,#bbf7d0 5%,#067666 100%); color:#105a5f; }}
    .risk-medium {{ background: linear-gradient(90deg,#fde68a 5%,#eab308 100%); color:#373308; }}
    .risk-high {{ background: linear-gradient(90deg,#fca5a5 5%,#ef4444 100%); color:#b91c1c; }}
    
    /* Chart background for dark mode */
    .stPlotlyChart {{
        background: {CARD_COLOR} !important;
    }}

    /* Expander head for better contrast */
    .stExpanderHeader {{
        background: {PRIMARY_GRAD};
        color: #fff !important;
        border-radius: 8px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown(f"""
<div style="text-align: center;">
    <h1>💼 Liquidity Risk Monitor</h1>
    <div style="font-size:1.15rem; color:{METRIC_LABEL_COLOR}; font-weight:400;">
        Next-gen Financial Risk Assessment & Monitoring Platform
    </div>
    <div style="color:#3b82f6; margin-top:.25rem; font-size:.96rem;">System Online • Last Updated: {datetime.now().strftime("%b %d, %Y %H:%M")}</div>
</div>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "result" not in st.session_state:
    st.session_state.result = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# --- MAIN ACTION ---
center_col = st.columns([1, 2, 1])[1]
with center_col:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("**🚀 Click below to analyze current liquidity risk:**\n\n"
        "*AI-driven analysis, scenario modeling, and cashflow forecasting—instantly delivered.*")
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("🔍 Run Analysis"):
        with st.spinner("Analyzing..."):
            initial_state: RiskState = {}
            result = supervisor_graph.invoke(initial_state)
            st.session_state.result = result
            st.session_state.analysis_done = True
            st.session_state.feedback_submitted = False

# --- RESULTS UI ---
if st.session_state.analysis_done and st.session_state.result is not None:
    result = st.session_state.result
    risk_level = result.get("risk", "unknown").lower()
    risk_colors = {
        "low": "risk-low",
        "medium": "risk-medium",
        "high": "risk-high"
    }
    risk_icons = {
        "low": "✅",
        "medium": "⚠️",
        "high": "🚨"
    }
    st.markdown(
        f"""<div class="risk-card {risk_colors.get(risk_level, 'risk-medium')}">
        <h2 style="font-size:2rem;">{risk_icons.get(risk_level)} Risk Level: {risk_level.upper()}</h2>
        <span style="font-size:1.05rem;opacity:0.85;">Analysis completed</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("## 📊 Analysis Report")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(result["report"])
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Chart Section Only for Medium/High risk ---
    if result["risk"] in ["medium", "high"]:
        st.markdown("## 📈 10-Day Cashflow Forecast")
        plt.style.use("dark_background" if st.session_state.theme_mode == "dark" else "default")
        fig, ax = plt.subplots(figsize=(12, 6))
        past_data = result["daily_cashflow"][-30:]
        primary_color = "#38bdf8" if st.session_state.theme_mode == "dark" else "#667eea"
        forecast_color = "#f43f5e" if st.session_state.theme_mode == "dark" else "#ef4444"
        ax.fill_between(past_data.index, past_data.values, color=primary_color, alpha=0.23)
        ax.plot(past_data.index, past_data.values, label="Past 30 Days", color=primary_color, linewidth=3)
        forecast_data = result["forecast_series"]
        ax.plot(forecast_data.index, forecast_data.values, "--o", c=forecast_color, lw=2.5, label="10-Day Forecast")
        ax.set_title("Liquidity Forecast Analysis", fontsize=18)
        ax.set_xlabel("Date")
        ax.set_ylabel("INR Amount")
        ax.legend()
        ax.tick_params(colors="#fff" if st.session_state.theme_mode=="dark" else "#333")
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig)

    # --- Expandable Details ---
    with st.expander("🔍 Show Detailed Scenarios & Features", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Features Extracted")
            st.json(result["features"])
        with c2:
            st.markdown("### Risk Scenarios")
            st.write(result["scenario_reasons"])

    # --- FEEDBACK SECTION ---
    st.markdown("---")
    st.markdown("## 💬 Feedback & Evaluation")
    if "user_satisfaction" not in st.session_state:
        st.session_state.user_satisfaction = "Yes"
    if "user_feedback" not in st.session_state:
        st.session_state.user_feedback = ""

    if not st.session_state.feedback_submitted:
        c1, c2 = st.columns(2)
        with c1:
            st.radio("Was the analysis/report satisfactory?", ("Yes", "No"), key="user_satisfaction")
        with c2:
            st.text_area("Any suggestions or comments (optional):", key="user_feedback", height=80)
        if st.button("✅ Submit Feedback & End Session"):
            import datetime, json, os
            from util.helpers import to_native
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feedback_log = {
                "timestamp": timestamp,
                "risk_analysis_result": result.get("report"),
                "satisfactory": st.session_state.user_satisfaction.lower(),
                "comments": st.session_state.user_feedback
            }
            os.makedirs("logs", exist_ok=True)
            with open(f"logs/user_feedback_{timestamp}.json", "w") as f:
                json.dump(to_native(feedback_log), f, indent=2)
            st.session_state.feedback_submitted = True
            st.success("🎉 Thanks for your feedback! It has been recorded.")
    else:
        st.info("✅ Feedback already submitted for this analysis session.")

else:
    # WELCOME SCREEN
    st.markdown(
        f"""<div class="info-card"><h3>🎯 Welcome to Liquidity Risk Monitor</h3>
        <p style="opacity:0.94;">A cutting-edge platform for comprehensive liquidity risk assessment,<br>
        cashflow forecasting, and actionable scenario modeling.</p>
        <ul>
            <li><span class="metric-label">AI-Assisted</span> risk & scenario analysis</li>
            <li><span class="metric-label">Forecasts</span> with advanced time series</li>
            <li><span class="metric-label">Detailed</span> features and insights</li>
            <li><span class="metric-label">Strategic</span> recommendations for treasury risk</li>
        </ul>
        <p><em>Click 'Run Analysis' to get started.</em></p>
        </div>""", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(
    f"""<div style="text-align:center;color:{METRIC_LABEL_COLOR};font-size:0.96rem;margin-top:1.9rem;">
    🔒 Secure · 🚀 Fast · 📊 Accurate | Liquidity Risk Monitor v2.0
    <br>[{'Light' if st.session_state.theme_mode=='light' else 'Dark'} mode]
    </div>""", unsafe_allow_html=True)
