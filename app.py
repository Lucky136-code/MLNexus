"""
MLNexus Streamlit Application.
"""

import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from charts import (plot_actual_vs_predicted, plot_confusion_matrix,
                    plot_correlation_heatmap, plot_cv_comparison,
                    plot_feature_importance, plot_model_comparison_clf,
                    plot_model_comparison_reg, plot_residuals,
                    plot_training_time)
from models import (ALGORITHM_INFO, CLASSIFICATION_RESULT_COLUMNS,
                    REGRESSION_INFO, REGRESSION_RESULT_COLUMNS,
                    get_feature_importance, train_all_models, tune_best_model)
from preprocessing import (coerce_object_columns_to_numeric,
                            fit_transform_train, load_dataset,
                            prepare_features_and_target, transform_test,
                            validate_dataset)

warnings.filterwarnings("ignore")

MAX_ROWS     = 10_000
RANDOM_STATE = 42

# Model family mapping
FAMILY_STYLE = {
    "Linear Model":           ("bdg-linear",  "Linear"),
    "Support Vector Machine": ("bdg-svm",     "SVM"),
    "Bayesian":               ("bdg-bayes",   "Bayesian"),
    "Instance-Based":         ("bdg-knn",     "k-NN"),
    "Tree / Ensemble":        ("bdg-tree",    "Tree"),
    "Neural Network":         ("bdg-neural",  "Neural"),
    "Discriminant Analysis":  ("bdg-disc",    "Disc."),
}

# SVG Icons
ICON_UPLOAD = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3"/><polyline points="6 9 12 3 18 9"/><line x1="4" y1="20" x2="20" y2="20"/></svg>"""

# 2. Preprocess sign: Configuration / Sliders
ICON_PREPROCESS = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>"""

# 3. Train sign: Brain connected to box hierarchy & gear (3rd image request)
ICON_TRAIN = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 4a3.5 3.5 0 0 0-3.5 3.5C4 8 3 9.5 3 11.5c0 2 1.5 3.5 3.5 3.5M9.5 20c-2.5 0-3.5-1.5-3.5-2.5"/><path d="M9.5 4c1.5 0 2.5 1 2.5 2.5V17.5c0 1.5-1 2.5-2.5 2.5"/><line x1="12" y1="12" x2="15" y2="12"/><rect x="15" y="3" width="6" height="5" rx="1"/><line x1="18" y1="8" x2="18" y2="10"/><rect x="15" y="10" width="6" height="5" rx="1"/><line x1="18" y1="15" x2="18" y2="17"/><circle cx="18" cy="19.5" r="2"/><path d="M18 16.5v.5M18 22v.5M15.5 19.5h.5M20 19.5h.5"/></svg>"""

# 4. Compare sign: 3 vertical bar columns on a baseline (4th image request)
ICON_COMPARE = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="2" y1="21" x2="22" y2="21"/><rect x="4" y="3" width="4" height="18"/><rect x="10" y="9" width="4" height="12"/><line x1="10" y1="13" x2="14" y2="13"/><rect x="16" y="15" width="4" height="6"/></svg>"""

# 5. Explain sign: Search / Magnifying glass
ICON_EXPLAIN = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>"""

# Checkmark icon for completed steps
ICON_CHECK = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"""

# Custom CSS
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & Base Font ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stMarkdown, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1f2937 !important;
}

/* ── Canvas Background ── */
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, header[data-testid="stHeader"], footer, .stDeployButton {
    visibility: hidden !important; height: 0 !important;
}

/* ── Block Container ── */
.main .block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1240px !important;
}

/* ══════════════════════════════════════════
   SIDEBAR & HIGH-CONTRAST CONTROLS
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.03) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

.sb-logo {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 1.25rem 0 1rem 0;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 0.5rem;
}
.sb-logo-text { font-size: 1.05rem; font-weight: 700; color: #0f172a; }

.sb-section {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #334155 !important; margin: 1.3rem 0 0.45rem 0;
}

/* Sidebar form control labels visibility */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}

/* Sidebar alert visibility */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #fef2f2 !important;
    border: 1px solid #fca5a5 !important;
    color: #991b1b !important;
    font-weight: 500 !important;
}

/* ── Primary Button (Red Theme) ── */
.stButton > button[kind="primary"] {
    background: #dc2626 !important; border: none !important;
    border-radius: 8px !important; color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 0.92rem !important;
    padding: 0.75rem 1.5rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 14px rgba(220,38,38,0.3) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    background: #b91c1c !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(220,38,38,0.4) !important;
}

/* ── Secondary Buttons ── */
.stButton > button:not([kind="primary"]) {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    background: #ffffff !important; color: #1f2937 !important;
    transition: all 0.15s ease !important;
    padding: 0.5rem 1.25rem !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #dc2626 !important; color: #dc2626 !important;
    background: #fff5f5 !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    background: #ffffff !important; color: #1f2937 !important;
}
.stDownloadButton > button:hover {
    border-color: #dc2626 !important; color: #dc2626 !important;
}

/* ── Metric Cards ── */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
}
div[data-testid="stMetric"] label {
    font-size: 0.72rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
    color: #475569 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important; font-weight: 600 !important;
    color: #0f172a !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] label {
    color: #0f172a !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploader"] section {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    padding: 1rem !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #dc2626 !important;
    background: #fff5f5 !important;
}
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploader"] button {
    background: #dc2626 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(220, 38, 38, 0.25) !important;
}
[data-testid="stFileUploader"] section button:hover,
[data-testid="stFileUploader"] button:hover {
    background: #b91c1c !important;
    color: #ffffff !important;
}
[data-testid="stFileUploader"] section button *,
[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
}
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section div[data-testid="stMarkdownContainer"] p {
    color: #475569 !important;
    font-weight: 500 !important;
}


/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 2px solid #e2e8f0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 0 !important;
    font-weight: 600 !important; color: #475569 !important;
    padding: 0.6rem 1.2rem !important; font-size: 0.86rem !important;
}
.stTabs [aria-selected="true"] {
    color: #dc2626 !important;
    border-bottom: 2px solid #dc2626 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .st-ae { background: #dc2626 !important; }

/* ── Progress Bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #dc2626, #ef4444) !important;
    border-radius: 9999px !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 8px !important; overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
}

/* ── Plotly Chart Card ── */
[data-testid="stPlotlyChart"] {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
    padding: 0.25rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important; font-size: 0.86rem !important;
    color: #1e293b !important;
}

/* ── Captions ── */
.stCaption p { font-size: 0.78rem !important; color: #475569 !important; font-weight: 500 !important; }

/* ══════════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════════ */

/* navbar */
.ag-nav {
    display: flex; align-items: center; justify-content: space-between;
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 2rem; height: 58px;
    margin: 0 -2rem 1.75rem -2rem;
    position: sticky; top: 0; z-index: 9999;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.ag-nav-left-empty { flex: 1; }
.ag-nav-center {
    flex: 1; display: flex; align-items: center; justify-content: center; gap: 0.5rem;
}
.ag-nav-right {
    flex: 1; display: flex; align-items: center; justify-content: flex-end;
    font-size: 0.8rem; color: #475569; font-weight: 500;
}

/* pipeline stepper - red process line theme */
.ag-pip {
    display: flex; align-items: flex-start;
    justify-content: center; padding: 1.25rem 2rem 1rem;
    background: #ffffff; border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    margin-bottom: 1.75rem; overflow-x: auto; gap: 0;
}
.ag-pip-step {
    display: flex; flex-direction: column;
    align-items: center; gap: 0.45rem; flex-shrink: 0;
}
.ag-pip-node {
    width: 44px; height: 44px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.3s ease;
}
.ag-pip-node.nd-pending {
    background: #f1f5f9 !important; color: #64748b !important;
    border: 1px solid #cbd5e1;
}
.ag-pip-node.nd-done {
    background: #dc2626 !important; color: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.15);
}
.ag-pip-node.nd-active {
    background: #dc2626 !important; color: #ffffff !important;
    animation: pip-pulse-red 1.6s ease-in-out infinite;
}
@keyframes pip-pulse-red {
    0%   { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.5); }
    70%  { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
    100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
}
.ag-pip-lbl {
    font-size: 0.73rem; font-weight: 600;
    color: #475569; white-space: nowrap;
}
.ag-pip-lbl.done   { color: #991b1b; font-weight: 700; }
.ag-pip-lbl.active { color: #dc2626; font-weight: 700; }
.ag-pip-conn {
    height: 3px; width: 56px; margin-top: 20px;
    flex-shrink: 0; border-radius: 9999px;
    transition: background 0.3s ease;
}
.ag-pip-conn.done    { background: #dc2626; }
.ag-pip-conn.active  { background: linear-gradient(90deg, #dc2626, #ef4444); }
.ag-pip-conn.pending { background: #e2e8f0; }

/* stat chip row */
.ag-stat-row { display:flex; flex-wrap:wrap; gap:0.6rem; margin-bottom:1.25rem; }
.ag-stat-chip {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:#ffffff; border:1px solid #e2e8f0;
    border-radius:8px; padding:0.4rem 0.95rem;
    font-size:0.82rem; font-weight:600; color:#334155;
    box-shadow:0 1px 3px rgba(0,0,0,0.03);
}
.ag-stat-chip .ag-stat-lbl { color:#475569; font-weight:600; }
.ag-stat-chip .v {
    font-family:'JetBrains Mono',monospace;
    font-weight:700; color:#0f172a;
}
.ag-stat-chip .ag-stat-sub { color:#64748b; font-weight:500; font-size:0.75rem; }

/* landing how-it-works cards */
.ag-how-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.5rem 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    height: 100%; display: flex; flex-direction: column; align-items: flex-start;
}
.ag-how-icon {
    width: 44px; height: 44px; border-radius: 10px;
    background: #fff5f5; color: #dc2626;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.85rem; border: 1px solid #fca5a5;
}
.ag-how-title {
    font-size: 0.95rem; font-weight: 700;
    color: #0f172a; margin-bottom: 0.4rem;
}
.ag-how-desc { font-size: 0.83rem; color: #475569; line-height: 1.6; }

/* leaderboard */
.ag-lb { display:flex; flex-direction:column; gap:0.45rem; margin:0.5rem 0 1rem 0; }
.ag-lb-row {
    display:flex; align-items:center; gap:0.85rem;
    background:#ffffff; border-radius:8px;
    border:1px solid #e2e8f0; padding:0.65rem 1rem;
    transition: box-shadow 0.18s, transform 0.18s;
}
.ag-lb-row:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    transform: translateY(-1px);
}
.ag-lb-row.winner {
    border-left: 4px solid #dc2626;
    background: #fff5f5;
    box-shadow: 0 2px 10px rgba(220,38,38,0.08);
}
.ag-lb-rank {
    font-family:'JetBrains Mono',monospace;
    font-size:0.82rem; font-weight:700;
    color:#64748b; width:28px; text-align:center; flex-shrink:0;
}
.ag-win-badge {
    background: #dc2626; color: #ffffff;
    font-size: 0.65rem; font-weight: 700;
    padding: 0.15rem 0.45rem; border-radius: 4px;
}
.ag-lb-name {
    font-size:0.88rem; font-weight:600; color:#0f172a;
    flex:1; min-width:120px;
}
.ag-lb-badge {
    font-size:0.68rem; font-weight:700;
    padding:0.2rem 0.6rem; border-radius:6px;
    white-space:nowrap; flex-shrink:0;
}
/* family badge colours */
.bdg-linear { background:#dbeafe; color:#1e40af; }
.bdg-svm    { background:#fce7f3; color:#9d174d; }
.bdg-bayes  { background:#ffedd5; color:#9a3412; }
.bdg-knn    { background:#fef9c3; color:#854d0e; }
.bdg-tree   { background:#dcfce7; color:#166534; }
.bdg-neural { background:#ede9fe; color:#5b21b6; }
.bdg-disc   { background:#e0e7ff; color:#3730a3; }

.ag-lb-bar-wrap { flex:2; display:flex; align-items:center; gap:0.55rem; min-width:80px; }
.ag-lb-bar-track {
    flex:1; height:6px; background:#e2e8f0;
    border-radius:9999px; overflow:hidden;
}
.ag-lb-bar-fill {
    height:100%; border-radius:9999px;
    background:#475569;
    transition: width 0.6s cubic-bezier(0.23,1,0.32,1);
}
.ag-lb-bar-fill.w { background: linear-gradient(90deg, #dc2626, #ef4444); }
.ag-lb-score {
    font-family:'JetBrains Mono',monospace;
    font-size:0.82rem; font-weight:700; color:#1e293b;
    width:50px; text-align:right; flex-shrink:0;
}
.ag-lb-time {
    font-size:0.75rem; color:#64748b; font-weight:500;
    width:48px; text-align:right; flex-shrink:0;
}

/* leaderboard column header */
.ag-lb-hdr {
    display:flex; align-items:center; gap:0.85rem;
    padding:0 1rem 0.4rem 1rem;
    font-size:0.7rem; font-weight:700;
    letter-spacing:0.07em; text-transform:uppercase; color:#475569;
}
.ag-lb-hdr-rank  { width:28px; flex-shrink:0; text-align:center; }
.ag-lb-hdr-name  { flex:1; min-width:120px; }
.ag-lb-hdr-badge { width:68px; flex-shrink:0; }
.ag-lb-hdr-bar   { flex:2; min-width:80px; }
.ag-lb-hdr-time  { width:48px; text-align:right; flex-shrink:0; }

/* explanation card */
.ag-explain {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #dc2626;
    border-radius: 10px;
    padding: 1.3rem 1.5rem; margin-bottom: 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.ag-explain h3 {
    font-size:1.02rem; font-weight:700; color:#0f172a;
    margin:0 0 0.55rem 0;
}
.ag-explain p {
    font-size:0.88rem; color:#334155; line-height:1.65;
    margin:0 0 1rem 0;
}
.ag-met-trio { display:flex; gap:0.6rem; flex-wrap:wrap; }
.ag-met-chip {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:8px; padding:0.6rem 1rem;
    text-align:center; min-width:90px;
}
.ag-met-chip .mv {
    font-family:'JetBrains Mono',monospace;
    font-size:1.1rem; font-weight:700; color:#0f172a; display:block;
}
.ag-met-chip .ml {
    font-size:0.68rem; font-weight:700; color:#475569;
    text-transform:uppercase; letter-spacing:0.05em;
}

/* chart header */
.ag-ch-hdr { margin:1.5rem 0 0.4rem 0; }
.ag-ch-title { font-size:0.92rem; font-weight:700; color:#0f172a; margin:0; }
.ag-ch-sub   { font-size:0.78rem; color:#475569; margin:0.15rem 0 0 0; font-weight:500; }

/* section label */
.ag-sec {
    font-size:0.72rem; font-weight:700;
    letter-spacing:0.09em; text-transform:uppercase;
    color:#334155; margin:2rem 0 0.8rem 0;
    display:flex; align-items:center; gap:0.5rem;
}
.ag-sec::after {
    content:''; flex:1; height:1px; background:#e2e8f0;
}

/* landing CTA */
.ag-cta {
    text-align:center; padding:2rem 1rem 1.75rem;
}
.ag-cta h2 {
    font-size:1.6rem; font-weight:700; color:#0f172a;
    margin-bottom:0.45rem;
}
.ag-cta p {
    color:#475569; font-size:0.95rem; font-weight:500;
    max-width:520px; margin:0 auto; line-height:1.6;
}
</style>
"""

# UI Components
def navbar_html(task_type: str = "") -> str:
    return f"""
    <div class="ag-nav">
        <div class="ag-nav-left-empty"></div>
        <div class="ag-nav-center">
            <span style="color:#dc2626; display:flex; align-items:center;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            </span>
            <span style="font-weight:800; color:#0f172a; font-size:1.25rem; letter-spacing:-0.02em;">ML</span><span style="font-weight:800; color:#dc2626; font-size:1.25rem; letter-spacing:-0.02em;">Nexus</span>
        </div>
        <div class="ag-nav-right"></div>
    </div>"""


def pipeline_html(step: int) -> str:
    """step: 0=nothing, 1=uploaded, 2=training, 3=comparing, 4=explaining, 5=done"""
    labels = ["Upload", "Preprocess", "Train", "Compare", "Explain"]
    icons  = [ICON_UPLOAD, ICON_PREPROCESS, ICON_TRAIN, ICON_COMPARE, ICON_EXPLAIN]

    html = '<div class="ag-pip">'
    for i in range(5):
        if step > i:
            node_cls = "nd-done"; lbl_cls = "done"
            icon = ICON_CHECK
        elif step == i:
            node_cls = "nd-active"; lbl_cls = "active"
            icon = icons[i]
        else:
            node_cls = "nd-pending"; lbl_cls = "pending"
            icon = icons[i]

        html += f"""
        <div class="ag-pip-step">
            <div class="ag-pip-node {node_cls}">{icon}</div>
            <div class="ag-pip-lbl {lbl_cls}">{labels[i]}</div>
        </div>"""

        if i < 4:
            conn_cls = ("done" if step > i else ("active" if step == i else "pending"))
            html += f'<div class="ag-pip-conn {conn_cls}"></div>'

    return html + "</div>"


def stat_row_html(df: pd.DataFrame) -> str:
    items = [
        ("Rows", f"{df.shape[0]:,}", "records"),
        ("Columns", str(df.shape[1]), "features"),
        ("Missing", f"{int(df.isna().sum().sum()):,}", "nulls"),
        ("Duplicates", str(int(df.duplicated().sum())), "copies"),
    ]
    inner = "".join(
        f'<span class="ag-stat-chip">'
        f'<span class="ag-stat-lbl">{lbl_name}:</span><span class="v">{val}</span>'
        f'<span class="ag-stat-sub">{sub_lbl}</span></span>'
        for lbl_name, val, sub_lbl in items
    )
    return f'<div class="ag-stat-row">{inner}</div>'


def family_badge(family: str) -> str:
    css, short = FAMILY_STYLE.get(family, ("bdg-disc", family[:5]))
    return f'<span class="ag-lb-badge {css}">{short}</span>'


def leaderboard_html(results_df: pd.DataFrame,
                     task_type: str, best_name: str) -> str:
    primary = "F1-Score" if task_type == "classification" else "R²"
    max_s   = max(results_df[primary].max(), 1e-9)

    header = """
    <div class="ag-lb-hdr">
        <span class="ag-lb-hdr-rank">#</span>
        <span class="ag-lb-hdr-name">Model</span>
        <span class="ag-lb-hdr-badge">Family</span>
        <span class="ag-lb-hdr-bar">Score</span>
        <span class="ag-lb-hdr-time">Time</span>
    </div>"""

    rows = ""
    for _, row in results_df.iterrows():
        rank   = int(row["Rank"])
        name   = str(row["Model"])
        fam    = str(row["Family"])
        score  = float(row[primary])
        t      = float(row["Training Time (s)"])
        win    = (name == best_name)
        pct    = max(0.0, min(100.0, score / max_s * 100))

        rank_html = (
            '<div class="ag-lb-rank w"><span class="ag-win-badge">#1</span></div>' if win
            else f'<div class="ag-lb-rank">{rank}</div>'
        )
        bar_cls  = "ag-lb-bar-fill w" if win else "ag-lb-bar-fill"
        row_cls  = "ag-lb-row winner" if win else "ag-lb-row"

        rows += f"""
        <div class="{row_cls}">
            {rank_html}
            <div class="ag-lb-name">{name}</div>
            {family_badge(fam)}
            <div class="ag-lb-bar-wrap">
                <div class="ag-lb-bar-track">
                    <div class="{bar_cls}" style="width:{pct:.1f}%"></div>
                </div>
                <div class="ag-lb-score">{score:.4f}</div>
            </div>
            <div class="ag-lb-time">{t:.2f}s</div>
        </div>"""

    return f'<div class="ag-lb">{header}{rows}</div>'


def explain_card_html(name: str, p_label: str,
                      results: dict, algo_dict: dict) -> str:
    desc    = algo_dict.get(name, "")
    primary = results["best_primary"]
    t       = results["best_time"]
    rdf     = results["results_df"]

    # Extra metric chips depending on task
    # NOTE: built as single flush-left lines — Streamlit's markdown parser
    # turns indented HTML after a blank line into a code block otherwise.
    rt = results["task_type"]
    try:
        row = rdf[rdf["Model"] == name].iloc[0]
        if rt == "classification":
            extras = (
                f'<div class="ag-met-chip"><span class="mv">{row["Accuracy"]:.4f}</span><span class="ml">Accuracy</span></div>'
                f'<div class="ag-met-chip"><span class="mv">{row["Precision"]:.4f}</span><span class="ml">Precision</span></div>'
                f'<div class="ag-met-chip"><span class="mv">{row["Recall"]:.4f}</span><span class="ml">Recall</span></div>'
            )
        else:
            extras = (
                f'<div class="ag-met-chip"><span class="mv">{row["MAE"]:.4f}</span><span class="ml">MAE</span></div>'
                f'<div class="ag-met-chip"><span class="mv">{row["RMSE"]:.4f}</span><span class="ml">RMSE</span></div>'
            )
    except Exception:
        extras = ""

    return (
        '<div class="ag-explain">'
        f'<h3>Why {name} was selected</h3>'
        f'<p>{desc}</p>'
        '<div class="ag-met-trio">'
        f'<div class="ag-met-chip"><span class="mv">{primary:.4f}</span><span class="ml">{p_label}</span></div>'
        f'{extras}'
        f'<div class="ag-met-chip"><span class="mv">{t}s</span><span class="ml">Train time</span></div>'
        '</div></div>'
    )


def chart_header(title: str, sub: str = "") -> str:
    sub_html = f'<p class="ag-ch-sub">{sub}</p>' if sub else ""
    return (f'<div class="ag-ch-hdr">'
            f'<p class="ag-ch-title">{title}</p>{sub_html}</div>')


def section(label: str) -> str:
    return f'<div class="ag-sec">{label}</div>'


# Streamlit Config
st.set_page_config(
    page_title="MLNexus",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


# Sidebar Controls
with st.sidebar:
    st.markdown(
        '<div class="sb-logo">'
        '<span style="color:#dc2626; display:flex; align-items:center;">'
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>'
        '</span>'
        '<span class="sb-logo-text"><span style="font-weight:800; color:#0f172a;">ML</span><span style="font-weight:800; color:#dc2626;">Nexus</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section">Task type</div>', unsafe_allow_html=True)
    task_type = st.radio(
        "Task type", ["Classification", "Regression"],
        horizontal=True, label_visibility="collapsed",
    ).lower()

    st.markdown('<div class="sb-section">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV / TSV", type=["csv", "tsv"],
        label_visibility="collapsed",
        help="Header row required · ≥ 10 rows · ≥ 2 columns",
    )

    test_size_pct = 20
    run_cv = False
    run_tuning = False

    # ── Load data ────────────────────────────────────────────────────────────
    df, uploaded_name = None, None

    if st.session_state.get("use_sample"):
        try:
            df = pd.read_csv("sample_data.csv")
            uploaded_name = "sample_data.csv"
        except FileNotFoundError:
            st.session_state["use_sample"] = False

    if uploaded_file is not None:
        st.session_state["use_sample"] = False
        try:
            df = load_dataset(uploaded_file)
            uploaded_name = uploaded_file.name
        except ValueError as exc:
            st.error(str(exc))

    if df is not None:
        probs = validate_dataset(df)
        if probs:
            for p in probs:
                st.error(p)
            df = None

    # Cache-key: new file or changed task type clears previous results
    if df is not None:
        cache_key = f"{uploaded_name}_{task_type}"
        if st.session_state.get("loaded_key") != cache_key:
            st.session_state["loaded_key"]     = cache_key
            st.session_state["automl_results"] = None

        st.success(f"**{uploaded_name}**  \n{df.shape[0]:,} rows × {df.shape[1]} cols")

        common_targets = ("target","class","label","y","species","diagnosis",
                          "outcome","survived","purchased","price","salary",
                          "value","score","result")
        default_idx = next(
            (i for i, c in enumerate(df.columns)
             if c.strip().lower() in common_targets),
            df.shape[1] - 1,
        )
        st.markdown('<div class="sb-section">Target column</div>', unsafe_allow_html=True)
        target_col = st.selectbox(
            "Target column", df.columns,
            index=default_idx, label_visibility="collapsed",
        )

        if st.session_state.get("last_target") != target_col:
            st.session_state["last_target"]    = target_col
            st.session_state["automl_results"] = None

        st.markdown("")
        start_training = st.button(
            "Start AutoML", type="primary", width="stretch",
        )
    else:
        st.info("Upload a CSV/TSV file to begin.")
        start_training = False
        target_col     = None


# ============================================================================
# NAVBAR
# ============================================================================
st.markdown(navbar_html(task_type), unsafe_allow_html=True)


# ============================================================================
# PIPELINE DIAGRAM  (step determined from session state)
# ============================================================================
_results_pre = st.session_state.get("automl_results")
if _results_pre is not None:
    _pip_step = 5
elif df is not None:
    _pip_step = 1
else:
    _pip_step = 0

st.markdown(pipeline_html(_pip_step), unsafe_allow_html=True)


# ============================================================================
# LANDING SCREEN
# ============================================================================
if df is None:
    st.markdown(
        '<div class="ag-cta">'
        '<h2>Upload your dataset · pick a task · get results</h2>'
        '<p>Up to 16 algorithms trained, ranked, and explained automatically.<br>'
        'Optional cross-validation and hyperparameter tuning in one click.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    cards = [
        (ICON_UPLOAD, "1. Upload", "CSV or TSV with a header row. Instant stats, column types, and correlation heatmap appear automatically."),
        (ICON_PREPROCESS, "2. Configure", "Choose Classification or Regression, pick the target column, set the test split, and enable CV / tuning."),
        (ICON_COMPARE, "3. Analyze", "Leaderboard with coloured family badges, score bars, feature importance, confusion matrix, and explainability."),
    ]
    for col, (icon_svg, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f'<div class="ag-how-card">'
                f'<div class="ag-how-icon">{icon_svg}</div>'
                f'<div class="ag-how-title">{title}</div>'
                f'<div class="ag-how-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )




# Dataset overview
elif _results_pre is None:
    st.markdown(section("Dataset overview"), unsafe_allow_html=True)
    st.markdown(stat_row_html(df), unsafe_allow_html=True)

    tab_prev, tab_info, tab_corr = st.tabs([
        "Preview (first 10 rows)",
        "Column info",
        "Correlation",
    ])
    with tab_prev:
        st.dataframe(df.head(10), width="stretch")
    with tab_info:
        typed = coerce_object_columns_to_numeric(df.copy())
        ci = pd.DataFrame({
            "Column":   df.columns,
            "Type":     ["Numeric" if pd.api.types.is_numeric_dtype(typed[c])
                         else "Categorical" for c in df.columns],
            "Missing":  [int(df[c].isna().sum()) for c in df.columns],
            "Missing %":[round(100 * df[c].isna().mean(), 1) for c in df.columns],
            "Unique":   [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(ci, width="stretch", hide_index=True)
    with tab_corr:
        try:
            fig_c, _ = plot_correlation_heatmap(df)
            st.plotly_chart(fig_c, width="stretch")
        except Exception:
            st.info("Not enough numeric columns for a correlation heatmap.")

    st.markdown(
        '<p style="font-size:0.82rem;color:#475569;text-align:center;font-weight:500;'
        'margin-top:0.75rem">Configure in the sidebar, then click '
        '<strong style="color:#dc2626">Start AutoML</strong></p>',
        unsafe_allow_html=True,
    )


# Pipeline execution
def run_automl(df, target_col, task_type, test_frac, run_cv, run_tuning):
    progress = st.progress(0.0, text="Starting pipeline…")
    try:
        # Step 1 — prepare
        progress.progress(0.08, text="Step 1/5 — Preparing features & target…")
        X, y, target_encoder, col_info = prepare_features_and_target(
            df, target_col, task_type
        )

        if task_type == "classification" and col_info["n_classes"] > 20:
            st.warning(
                f"Target has **{col_info['n_classes']} distinct values**. "
                "Consider switching to **Regression** mode for continuous targets."
            )

        subsampled = False
        if len(X) > MAX_ROWS:
            idx = X.sample(MAX_ROWS, random_state=RANDOM_STATE).index
            X, y = X.loc[idx], y.loc[idx]
            subsampled = True

        # Step 2 — split
        progress.progress(0.18, text="Step 2/5 — Train / test split…")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_frac, random_state=RANDOM_STATE,
                stratify=(y if task_type == "classification" else None),
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_frac, random_state=RANDOM_STATE,
            )

        # Step 3 — leakage-free transform
        progress.progress(0.28, text="Step 3/5 — Leakage-free preprocessing…")
        X_train, pipeline = fit_transform_train(X_train, col_info)
        X_test             = transform_test(X_test, col_info, pipeline)

        # Step 4 — train
        trained, failed = train_all_models(
            X_train, X_test, y_train, y_test,
            task_type, progress, run_cv=run_cv,
        )
        if not trained:
            st.error("All models failed. Please check your dataset.")
            return None

        # Step 5 — evaluate
        progress.progress(0.92, text="Step 5/5 — Ranking & evaluating…")
        trained.sort(key=lambda r: r["_sort_key"], reverse=True)
        best      = trained[0]
        best_name = best["Model"]

        # Optional tuning
        tuning_info = None
        if run_tuning:
            tuned_model, tuning_info = tune_best_model(
                best["_model"], best_name,
                X_train, y_train, task_type, progress,
            )
            if tuning_info:
                tp = tuned_model.predict(X_test)
                best["_model"] = tuned_model; best["_predictions"] = tp
                if task_type == "classification":
                    from sklearn.metrics import f1_score as _f1
                    best["F1-Score"]  = _f1(y_test, tp, average="weighted", zero_division=0)
                    best["_sort_key"] = best["F1-Score"]
                else:
                    from sklearn.metrics import r2_score as _r2
                    best["R²"] = _r2(y_test, tp); best["_sort_key"] = best["R²"]

        # Build leaderboard dataframe
        keep = ([c for c in CLASSIFICATION_RESULT_COLUMNS if c != "Rank"]
                if task_type == "classification"
                else [c for c in REGRESSION_RESULT_COLUMNS if c != "Rank"])
        results_df = pd.DataFrame(trained)[keep]
        results_df.insert(0, "Rank", range(1, len(results_df) + 1))

        cm = None
        if task_type == "classification":
            cm = confusion_matrix(
                y_test, best["_predictions"],
                labels=np.arange(col_info["n_classes"]),
            )

        try:
            imp, imp_method = get_feature_importance(
                best["_model"], X_test, y_test, list(X.columns), task_type,
            )
        except Exception:
            imp, imp_method = None, None

        progress.progress(1.0, text="AutoML complete!")

        return {
            "task_type":         task_type,
            "target":            target_col,
            "results_df":        results_df,
            "best_name":         best_name,
            "best_primary":      best["_sort_key"],
            "best_time":         best["Training Time (s)"],
            "best_predictions":  best["_predictions"],
            "importance":        imp,
            "importance_method": imp_method,
            "cm":                cm,
            "col_info":          col_info,
            "y_test":            y_test,
            "n_train":           len(X_train),
            "n_test":            len(X_test),
            "n_features":        X.shape[1],
            "missing_imputed":   col_info["missing_before"],
            "n_numeric":         len(col_info["numeric_cols"]),
            "n_categorical":     len(col_info["categorical_cols"]),
            "dropped_target":    col_info["dropped_target_rows"],
            "subsampled":        subsampled,
            "failed":            failed,
            "tuning_info":       tuning_info,
            "run_cv":            run_cv,
        }

    except ValueError as exc:
        st.error(f"{exc}")
        return None
    except Exception as exc:
        st.error(f"Pipeline failed unexpectedly: {exc}")
        return None


if start_training and df is not None:
    st.session_state["automl_results"] = run_automl(
        df, target_col, task_type,
        test_size_pct / 100, run_cv, run_tuning,
    )


# ============================================================================
# RESULTS PANEL
# ============================================================================
results = st.session_state.get("automl_results")

if results is not None:
    rt      = results["task_type"]
    rname   = results["best_name"]
    rdf     = results["results_df"]
    p_label = "F1-Score" if rt == "classification" else "R²"
    algo_d  = ALGORITHM_INFO if rt == "classification" else REGRESSION_INFO

    # Preprocessing caption
    st.markdown(
        f'<p style="font-size:0.78rem;color:#475569;margin-bottom:1.25rem;font-weight:500">'
        f'Leakage-free preprocessing · Imputed '
        f'<strong>{results["missing_imputed"]:,}</strong> missing values '
        f'(fitted on train split only) · '
        f'<strong>{results["n_categorical"]}</strong> categorical columns encoded · '
        f'<strong>{results["n_numeric"]}</strong> numeric columns scaled · '
        f'<strong>{results["n_train"]:,}</strong> train / '
        f'<strong>{results["n_test"]:,}</strong> test · '
        f'<strong>{results["n_features"]}</strong> features'
        + (f' · subsampled to {MAX_ROWS:,} rows' if results["subsampled"] else "")
        + '</p>',
        unsafe_allow_html=True,
    )

    # ── 1. Leaderboard ───────────────────────────────────────────────────────
    st.markdown(section("Model Leaderboard"), unsafe_allow_html=True)
    st.markdown(leaderboard_html(rdf, rt, rname), unsafe_allow_html=True)

    left_dl, _, right_cv = st.columns([1, 2, 1])
    with left_dl:
        st.download_button(
            "Export CSV",
            rdf.assign(Best=["Yes" if m == rname else "" for m in rdf["Model"]])
               .to_csv(index=False).encode("utf-8"),
            file_name=f"automl_{results['target']}.csv",
            mime="text/csv",
        )
    with right_cv:
        if results["run_cv"]:
            st.caption("CV column = 3-fold mean ± std on ≤ 5,000 training rows")

    # ── 2. Explanation card ──────────────────────────────────────────────────
    st.markdown(section("Why this model?"), unsafe_allow_html=True)
    st.markdown(explain_card_html(rname, p_label, results, algo_d),
                unsafe_allow_html=True)

    why_col, tip_col = st.columns(2)
    with why_col:
        if rt == "classification":
            st.info("**Why F1?** Harmonic mean of Precision & Recall — robust to class imbalance, unlike raw accuracy.")
        else:
            st.info("**Why R²?** Fraction of variance explained. MAE/RMSE give error magnitude in target units.")
    with tip_col:
        if results.get("tuning_info"):
            ti = results["tuning_info"]
            st.success(
                f"Tuning improved **{rname}**  \n"
                f"Best params: `{ti['best_params']}`  \n"
                f"CV {ti['scoring']}: **{ti['cv_score']}**"
            )

    # ── 3. Performance comparison ────────────────────────────────────────────
    st.markdown(section("Performance comparison"), unsafe_allow_html=True)
    st.markdown(chart_header("All Models · Test Set",
                             "Higher is better for all metrics"), unsafe_allow_html=True)
    if rt == "classification":
        st.plotly_chart(plot_model_comparison_clf(rdf), width="stretch")
    else:
        st.plotly_chart(plot_model_comparison_reg(rdf), width="stretch")

    if results["run_cv"]:
        cv_fig = plot_cv_comparison(rdf, rt)
        if cv_fig:
            st.markdown(chart_header("Test Score vs CV Score",
                                     "Large gap may indicate overfitting"),
                        unsafe_allow_html=True)
            st.plotly_chart(cv_fig, width="stretch")

    # ── 4. Detailed analysis grid ────────────────────────────────────────────
    st.markdown(section("Detailed analysis"), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    if rt == "classification":
        with col_a:
            st.markdown(chart_header("Confusion Matrix",
                                     f"Best model: {rname}"), unsafe_allow_html=True)
            st.plotly_chart(
                plot_confusion_matrix(
                    results["cm"], results["col_info"]["class_names"], rname),
                width="stretch",
            )
        with col_b:
            if results["importance"] is not None:
                st.markdown(chart_header("Top 5 Features"), unsafe_allow_html=True)
                top5 = results["importance"].head(5).copy()
                top5.insert(0, "#", range(1, len(top5) + 1))
                top5["Importance"] = top5["Importance"].round(4)
                st.dataframe(top5, hide_index=True, width="stretch")
                st.caption(results["importance_method"])
    else:
        y_arr = (results["y_test"].values
                 if hasattr(results["y_test"], "values")
                 else np.asarray(results["y_test"]))
        with col_a:
            st.markdown(chart_header("Actual vs Predicted",
                                     "Points on the diagonal = perfect prediction"),
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_actual_vs_predicted(y_arr, results["best_predictions"], rname),
                width="stretch",
            )
        with col_b:
            st.markdown(chart_header("Residual Plot",
                                     "Ideal: random scatter centred on y = 0"),
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_residuals(y_arr, results["best_predictions"], rname),
                width="stretch",
            )

    # Feature importance full chart
    if results["importance"] is not None:
        st.markdown(chart_header("Feature Importance",
                                 results["importance_method"]), unsafe_allow_html=True)
        st.plotly_chart(
            plot_feature_importance(results["importance"],
                                    results["importance_method"]),
            width="stretch",
        )

    # Training time
    st.markdown(chart_header("Training Time",
                             "Log scale — lower is faster"), unsafe_allow_html=True)
    st.plotly_chart(plot_training_time(rdf), width="stretch")

    # ── 5. Algorithm reference ───────────────────────────────────────────────
    st.markdown(section("Algorithm reference"), unsafe_allow_html=True)
    with st.expander("How each trained algorithm works"):
        for name, desc in algo_d.items():
            trophy = " (selected)" if name == rname else ""
            st.markdown(f"**{name}{trophy}** — {desc}")

    if results["failed"]:
        st.warning(
            "Some models were skipped:\n\n"
            + "\n\n".join(f"• {f}" for f in results["failed"])
        )

    # Dataset peek (available after training)
    if df is not None:
        with st.expander("View dataset used for training"):
            st.dataframe(df.head(10), width="stretch")



