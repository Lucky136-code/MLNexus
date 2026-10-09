"""
MLNexus Streamlit Application — Modern Automated Machine Learning Platform.
"""

import warnings
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from preprocessing import (
    load_dataset, validate_dataset, auto_detect_task_type,
    analyze_target_and_features, prepare_features_and_target,
    prepare_unsupervised_features, fit_transform_train,
    transform_test, coerce_object_columns_to_numeric
)
from models import (
    train_all_models, train_unsupervised_models, tune_best_model,
    get_feature_importance, export_model_artifact,
    ALGORITHM_INFO, REGRESSION_INFO, UNSUPERVISED_INFO,
    CLASSIFICATION_RESULT_COLUMNS, REGRESSION_RESULT_COLUMNS,
    UNSUPERVISED_RESULT_COLUMNS
)
from charts import (
    plot_model_comparison_clf, plot_model_comparison_reg,
    plot_model_comparison_unsupervised, plot_confusion_matrix,
    plot_actual_vs_predicted, plot_residuals, plot_feature_importance,
    plot_training_time, plot_correlation_heatmap, plot_cv_comparison,
    plot_cluster_pca_scatter
)

warnings.filterwarnings("ignore")

# Streamlit Page Setup (Must be very first Streamlit call)
st.set_page_config(
    page_title="MLNexus — Automated ML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_ROWS = 10_000
RANDOM_STATE = 42
TASK_OPTIONS = ["Classification", "Regression", "Unsupervised"]
NONE_UNSUPERVISED = "None (Unsupervised Learning)"

FAMILY_STYLE = {
    "Linear Model":           ("bdg-linear",  "Linear"),
    "Support Vector Machine": ("bdg-svm",     "SVM"),
    "Bayesian":               ("bdg-bayes",   "Bayesian"),
    "Instance-Based":         ("bdg-knn",     "k-NN"),
    "Tree / Ensemble":        ("bdg-tree",    "Tree"),
    "Neural Network":         ("bdg-neural",  "Neural"),
    "Discriminant Analysis":  ("bdg-disc",    "Disc."),
    "Clustering":             ("bdg-cluster", "Cluster"),
    "Dimensionality Reduction": ("bdg-dim",   "DimRed"),
    "Anomaly Detection":      ("bdg-anomaly", "Anomaly"),
}

# SVG Icons
ICON_UPLOAD = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3"/><polyline points="6 9 12 3 18 9"/><line x1="4" y1="20" x2="20" y2="20"/></svg>"""
ICON_INSPECT = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>"""
ICON_TRAIN = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 4a3.5 3.5 0 0 0-3.5 3.5C4 8 3 9.5 3 11.5c0 2 1.5 3.5 3.5 3.5M9.5 20c-2.5 0-3.5-1.5-3.5-2.5"/><path d="M9.5 4c1.5 0 2.5 1 2.5 2.5V17.5c0 1.5-1 2.5-2.5 2.5"/><line x1="12" y1="12" x2="15" y2="12"/><rect x="15" y="3" width="6" height="5" rx="1"/><line x1="18" y1="8" x2="18" y2="10"/><rect x="15" y="10" width="6" height="5" rx="1"/><line x1="18" y1="15" x2="18" y2="17"/><circle cx="18" cy="19.5" r="2"/><path d="M18 16.5v.5M18 22v.5M15.5 19.5h.5M20 19.5h.5"/></svg>"""
ICON_COMPARE = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="2" y1="21" x2="22" y2="21"/><rect x="4" y="3" width="4" height="18"/><rect x="10" y="9" width="4" height="12"/><line x1="10" y1="13" x2="14" y2="13"/><rect x="16" y="15" width="4" height="6"/></svg>"""
ICON_EXPLAIN = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>"""
ICON_CHECK = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"""

# Custom CSS
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stMarkdown, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0f172a !important;
}

[data-testid="stAppViewContainer"] {
    background: #f8fafc !important;
}

#MainMenu, footer, .stDeployButton {
    visibility: hidden !important; height: 0 !important; display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    overflow: visible !important;
    z-index: 100000 !important;
}

[data-testid="stCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="stHeaderNavButton"],
button[data-testid="stHeaderCollapseButton"] {
    position: fixed !important;
    top: 12px !important;
    right: 18px !important;
    left: auto !important;
    z-index: 100005 !important;
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 42px !important;
    height: 42px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
[data-testid="stCollapsedControl"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
    border-color: #6366f1 !important;
    background: #f5f3ff !important;
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.2) !important;
}
[data-testid="stCollapsedControl"] *,
[data-testid="stSidebarCollapseButton"] * {
    color: #4f46e5 !important;
    fill: #4f46e5 !important;
    stroke: #4f46e5 !important;
}

.main .block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1280px !important;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 4px 0 24px rgba(15, 23, 42, 0.03) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

.sb-logo {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 1.25rem 0 1rem 0;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 0.75rem;
}
.sb-logo-text { font-size: 1.15rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; }

.sb-section {
    font-size: 0.75rem; font-weight: 800;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #64748b !important; margin: 1.4rem 0 0.5rem 0;
}

[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1.5rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 6px 20px -2px rgba(79, 70, 229, 0.35) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px -2px rgba(79, 70, 229, 0.5) !important;
}

.stButton > button:not([kind="primary"]) {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 0.88rem !important; font-weight: 600 !important;
    background: #ffffff !important; color: #1e293b !important;
    transition: all 0.2s ease !important;
    padding: 0.55rem 1.25rem !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #6366f1 !important; color: #4f46e5 !important;
    background: #f5f3ff !important;
    transform: translateY(-1px) !important;
}

.stDownloadButton > button {
    background: #ffffff !important;
    border: 1.5px solid #6366f1 !important;
    color: #4f46e5 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.25rem !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: #4f46e5 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
}

[data-testid="stFileUploader"] section {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    padding: 1.25rem !important;
}

[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] section button,
button[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
    background-color: #f5f3ff !important;
    background: #f5f3ff !important;
    color: #4f46e5 !important;
    border: 1.5px solid #6366f1 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.12) !important;
}

[data-testid="stFileUploader"] button *,
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploader"] section button *,
button[data-testid="stBaseButton-secondary"] *,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] * {
    color: #4f46e5 !important;
    fill: #4f46e5 !important;
    stroke: #4f46e5 !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #4f46e5 !important;
    background: #4f46e5 !important;
    color: #ffffff !important;
}

[data-testid="stFileUploader"] button:hover *,
[data-testid="stFileUploaderDropzone"] button:hover * {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section p {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Radio Button Option Labels Fix */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p,
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] *,
div[data-baseweb="radio"] *,
div[data-baseweb="radio"] label span {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    opacity: 1 !important;
    visibility: visible !important;
}

.ag-nav {
    display: flex; align-items: center; justify-content: center;
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 1.5rem; height: 60px;
    margin: 0 -2rem 1.75rem -2rem;
    position: sticky; top: 0; z-index: 9999;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}

.ag-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.75rem;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    margin-bottom: 1.75rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.ag-card-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.25rem; padding-bottom: 0.85rem;
    border-bottom: 1px solid #f1f5f9;
}
.ag-card-title {
    font-size: 1.1rem; font-weight: 800; color: #0f172a;
    display: flex; align-items: center; gap: 0.6rem;
}

.ag-pip {
    display: flex; align-items: center;
    justify-content: space-between; padding: 1.25rem 2rem;
    background: #ffffff; border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
    margin-bottom: 2rem; overflow-x: auto;
}
.ag-pip-step {
    display: flex; flex-direction: column;
    align-items: center; gap: 0.4rem; flex-shrink: 0;
}
.ag-pip-node {
    width: 44px; height: 44px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    font-weight: 700;
}
.ag-pip-node.nd-pending { background: #f1f5f9 !important; color: #94a3b8 !important; border: 1.5px solid #cbd5e1; }
.ag-pip-node.nd-done { background: #10b981 !important; color: #ffffff !important; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3); }
.ag-pip-node.nd-active { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important; color: #ffffff !important; box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.25); }
.ag-pip-lbl { font-size: 0.78rem; font-weight: 700; color: #64748b; white-space: nowrap; }
.ag-pip-lbl.done { color: #059669; }
.ag-pip-lbl.active { color: #4f46e5; }
.ag-pip-conn { height: 3px; flex: 1; max-width: 80px; margin: 0 0.5rem; border-radius: 9999px; }
.ag-pip-conn.done { background: #10b981; }
.ag-pip-conn.active { background: linear-gradient(90deg, #10b981, #6366f1); }
.ag-pip-conn.pending { background: #e2e8f0; }

.ag-stat-row { display:flex; flex-wrap:wrap; gap:0.75rem; margin-bottom:1.5rem; }
.ag-stat-chip {
    display:inline-flex; align-items:center; gap:0.5rem;
    background:#ffffff; border:1px solid #e2e8f0;
    border-radius:10px; padding:0.5rem 1.15rem;
    font-size:0.85rem; font-weight:700; color:#334155;
    box-shadow:0 2px 6px rgba(15,23,42,0.03);
}
.ag-stat-chip .ag-stat-lbl { color:#64748b; font-weight:600; }
.ag-stat-chip .v { font-family:'JetBrains Mono',monospace; font-weight:800; color:#0f172a; }
.ag-stat-chip .ag-stat-sub { color:#94a3b8; font-weight:600; font-size:0.78rem; }

.ag-hero { text-align: center; padding: 2.5rem 1rem 2rem; }
.ag-hero-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: #f5f3ff; border: 1px solid #c7d2fe;
    border-radius: 9999px; padding: 0.4rem 1rem;
    color: #4f46e5; font-size: 0.82rem; font-weight: 700; margin-bottom: 1.25rem;
}
.ag-hero h1 { font-size: 2.35rem; font-weight: 800; color: #0f172a; letter-spacing: -0.03em; margin-bottom: 0.75rem; }
.ag-hero-gradient-text { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.ag-hero p { font-size: 1.05rem; color: #475569; max-width: 720px; margin: 0 auto 2rem auto; line-height: 1.6; font-weight: 500; }

.ag-lb { display:flex; flex-direction:column; gap:0.5rem; margin:0.75rem 0 1.25rem 0; }
.ag-lb-row { display:flex; align-items:center; gap:1rem; background:#ffffff; border-radius:10px; border:1px solid #e2e8f0; padding:0.75rem 1.15rem; }
.ag-lb-row.winner { border-left: 4px solid #4f46e5; background: #faf5ff; box-shadow: 0 4px 16px rgba(99, 102, 241, 0.12); }
.ag-lb-rank { font-family:'JetBrains Mono',monospace; font-size:0.88rem; font-weight:800; color:#64748b; width:95px; min-width:95px; text-align:center; flex-shrink:0; display:flex; align-items:center; justify-content:center; }
.ag-win-badge { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #ffffff !important; font-size: 0.72rem; font-weight: 800; padding: 0.25rem 0.65rem; border-radius: 6px; white-space: nowrap !important; }
.ag-lb-name { font-size:0.92rem; font-weight:700; color:#0f172a; flex:1; min-width:130px; }
.ag-lb-badge { font-size:0.7rem; font-weight:800; padding:0.25rem 0.65rem; border-radius:6px; white-space:nowrap; flex-shrink:0; }

.bdg-linear  { background:#dbeafe; color:#1e40af; }
.bdg-svm     { background:#fce7f3; color:#9d174d; }
.bdg-bayes   { background:#ffedd5; color:#9a3412; }
.bdg-knn     { background:#fef9c3; color:#854d0e; }
.bdg-tree    { background:#dcfce7; color:#166534; }
.bdg-neural  { background:#ede9fe; color:#5b21b6; }
.bdg-disc    { background:#e0e7ff; color:#3730a3; }
.bdg-cluster { background:#e0f2fe; color:#0369a1; }
.bdg-dim     { background:#f3e8ff; color:#6b21a8; }
.bdg-anomaly { background:#ffe4e6; color:#be123c; }

.ag-lb-bar-wrap { flex:2; display:flex; align-items:center; gap:0.65rem; min-width:90px; }
.ag-lb-bar-track { flex:1; height:8px; background:#f1f5f9; border-radius:9999px; overflow:hidden; }
.ag-lb-bar-fill { height:100%; border-radius:9999px; background:#64748b; }
.ag-lb-bar-fill.w { background: linear-gradient(90deg, #4f46e5, #7c3aed); }
.ag-lb-score { font-family:'JetBrains Mono',monospace; font-size:0.88rem; font-weight:800; color:#0f172a; width:56px; text-align:right; flex-shrink:0; }
.ag-lb-time { font-size:0.78rem; color:#64748b; font-weight:600; width:52px; text-align:right; flex-shrink:0; }
.ag-lb-hdr { display:flex; align-items:center; gap:1rem; padding:0 1.15rem 0.4rem 1.15rem; font-size:0.72rem; font-weight:800; letter-spacing:0.07em; text-transform:uppercase; color:#64748b; }
.ag-lb-hdr-rank { width:95px; min-width:95px; text-align:center; flex-shrink:0; }

.ag-explain { background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #4f46e5; border-radius: 12px; padding: 1.5rem 1.75rem; margin-bottom: 1.5rem; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04); }
.ag-explain h3 { font-size:1.1rem; font-weight:800; color:#0f172a; margin:0 0 0.6rem 0; }
.ag-explain p { font-size:0.92rem; color:#334155; line-height:1.65; margin:0 0 1.1rem 0; font-weight:500; }
.ag-met-trio { display:flex; gap:0.75rem; flex-wrap:wrap; }
.ag-met-chip { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.65rem 1.1rem; text-align:center; min-width:95px; }
.ag-met-chip .mv { font-family:'JetBrains Mono',monospace; font-size:1.15rem; font-weight:800; color:#0f172a; display:block; }
.ag-met-chip .ml { font-size:0.7rem; font-weight:800; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; }

.ag-sec { font-size:0.75rem; font-weight:800; letter-spacing:0.09em; text-transform:uppercase; color:#4f46e5; margin:2.25rem 0 0.9rem 0; display:flex; align-items:center; gap:0.6rem; }
.ag-sec::after { content:''; flex:1; height:1px; background:#e2e8f0; }

@media (max-width: 768px) {
    .main .block-container { padding: 0.5rem 0.75rem 2rem 0.75rem !important; }
    .ag-hero h1 { font-size: 1.75rem !important; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── SESSION STATE MANAGEMENT ────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state["step"] = 1
if "df" not in st.session_state:
    st.session_state["df"] = None
if "uploaded_name" not in st.session_state:
    st.session_state["uploaded_name"] = None
if "task_type" not in st.session_state:
    st.session_state["task_type"] = "classification"
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None
if "automl_results" not in st.session_state:
    st.session_state["automl_results"] = None
if "test_size_pct" not in st.session_state:
    st.session_state["test_size_pct"] = 20
if "run_cv" not in st.session_state:
    st.session_state["run_cv"] = False
if "run_tuning" not in st.session_state:
    st.session_state["run_tuning"] = False
if "exclude_cols" not in st.session_state:
    st.session_state["exclude_cols"] = []

# Helper UI functions
def navbar_html() -> str:
    return f"""
    <div class="ag-nav">
        <div class="ag-nav-logo-center">
            <span style="color:#4f46e5; display:flex; align-items:center;">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            </span>
            <span style="font-weight:800; color:#0f172a; font-size:1.35rem; letter-spacing:-0.03em; margin-left:6px;">ML</span><span style="font-weight:800; color:#4f46e5; font-size:1.35rem; letter-spacing:-0.03em;">Nexus</span>
            <span style="background:#f5f3ff; color:#4f46e5; border:1px solid #c7d2fe; font-size:0.68rem; font-weight:800; padding:0.15rem 0.55rem; border-radius:9999px; margin-left:8px;">AutoML Engine v3.0</span>
        </div>
    </div>"""

def stepper_bar_html(active_step: int) -> str:
    labels = ["1. Upload Dataset", "2. Inspect & Configure", "3. Train Models", "4. Compare Results", "5. Explain & Export"]
    icons  = [ICON_UPLOAD, ICON_INSPECT, ICON_TRAIN, ICON_COMPARE, ICON_EXPLAIN]

    html = '<div class="ag-pip">'
    for i in range(5):
        s_num = i + 1
        if active_step > s_num:
            node_cls = "nd-done"; lbl_cls = "done"; icon = ICON_CHECK
        elif active_step == s_num:
            node_cls = "nd-active"; lbl_cls = "active"; icon = icons[i]
        else:
            node_cls = "nd-pending"; lbl_cls = "pending"; icon = icons[i]

        html += f"""
        <div class="ag-pip-step">
            <div class="ag-pip-node {node_cls}">{icon}</div>
            <div class="ag-pip-lbl {lbl_cls}">{labels[i]}</div>
        </div>"""

        if i < 4:
            conn_cls = "done" if active_step > s_num else ("active" if active_step == s_num else "pending")
            html += f'<div class="ag-pip-conn {conn_cls}"></div>'

    return html + "</div>"

def stat_row_html(df: pd.DataFrame) -> str:
    items = [
        ("Total Rows", f"{df.shape[0]:,}", "records"),
        ("Features", str(df.shape[1]), "columns"),
        ("Missing Values", f"{int(df.isna().sum().sum()):,}", "null cells"),
        ("Duplicates", str(int(df.duplicated().sum())), "identical rows"),
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

def leaderboard_html(results_df: pd.DataFrame, task_type: str, best_name: str) -> str:
    primary = ("F1-Score" if task_type == "classification"
               else ("R²" if task_type == "regression"
                     else "Silhouette Score"))
    max_s = max(results_df[primary].max(), 1e-9)

    header = """
    <div class="ag-lb-hdr">
        <span class="ag-lb-hdr-rank">#</span>
        <span class="ag-lb-hdr-name">Model Algorithm</span>
        <span class="ag-lb-hdr-badge">Family</span>
        <span class="ag-lb-hdr-bar">Primary Score</span>
        <span class="ag-lb-hdr-time">Time</span>
    </div>"""

    rows = ""
    for _, row in results_df.iterrows():
        rank  = int(row["Rank"])
        name  = str(row["Model"])
        fam   = str(row["Family"])
        score = float(row[primary])
        t     = float(row["Training Time (s)"])
        win   = (name == best_name)
        pct   = max(0.0, min(100.0, score / max_s * 100))

        rank_html = (
            '<div class="ag-lb-rank w"><span class="ag-win-badge">#1 WINNER</span></div>' if win
            else f'<div class="ag-lb-rank">#{rank}</div>'
        )
        bar_cls = "ag-lb-bar-fill w" if win else "ag-lb-bar-fill"
        row_cls = "ag-lb-row winner" if win else "ag-lb-row"

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

def section(label: str) -> str:
    return f'<div class="ag-sec">{label}</div>'

# ── AUTOML ENGINE PIPELINE EXECUTION ─────────────────────────────────────────
def run_automl_engine(df, target_col, task_type, test_frac, run_cv, run_tuning, exclude_cols, status_container):
    progress = st.progress(0.0, text="Initializing AutoML Pipeline…")
    try:
        # Unsupervised Flow
        if task_type == "unsupervised":
            status_container.write("📌 **Step 1/3** — Fitting zero-leakage median imputers & feature scaling...")
            progress.progress(0.15, text="Step 1/3 — Scaling & Imputing Features…")
            X_scaled, col_info = prepare_unsupervised_features(df, exclude_cols=exclude_cols)

            status_container.write("🤖 **Step 2/3** — Training 10 Unsupervised algorithms (Clustering, PCA, Anomaly Detection)...")
            progress.progress(0.40, text="Step 2/3 — Training Unsupervised Models…")
            trained, failed = train_unsupervised_models(X_scaled, progress)

            if not trained:
                st.error("All unsupervised models failed. Please check your dataset.")
                return None

            status_container.write("📊 **Step 3/3** — Evaluating Silhouette Scores & PCA cluster projections...")
            progress.progress(0.90, text="Step 3/3 — Evaluating Clusters & Anomalies…")
            trained.sort(key=lambda r: r["_sort_key"], reverse=True)
            best = trained[0]
            best_name = best["Model"]

            results_df = pd.DataFrame(trained)[[c for c in UNSUPERVISED_RESULT_COLUMNS if c != "Rank"]]
            results_df.insert(0, "Rank", range(1, len(results_df) + 1))

            model_pickle = export_model_artifact(best["_model"], None, None, col_info, task_type)

            progress.progress(1.0, text="AutoML Pipeline Complete!")

            return {
                "task_type": task_type,
                "target": "N/A (Unsupervised)",
                "results_df": results_df,
                "best_name": best_name,
                "best_primary": best["_sort_key"],
                "best_time": best["Training Time (s)"],
                "best_predictions": best["_predictions"],
                "best_model": best["_model"],
                "importance": None,
                "importance_method": None,
                "cm": None,
                "col_info": col_info,
                "y_test": None,
                "X_scaled": X_scaled,
                "n_train": len(X_scaled),
                "n_test": len(X_scaled),
                "n_features": X_scaled.shape[1],
                "subsampled": False,
                "failed": failed,
                "tuning_info": None,
                "run_cv": False,
                "model_pickle": model_pickle,
            }

        # Supervised Flow
        status_container.write("📌 **Step 1/5** — Validating target variable & handling date features...")
        progress.progress(0.08, text="Step 1/5 — Preparing Features & Target…")
        X, y, target_encoder, col_info = prepare_features_and_target(
            df, target_col, task_type, exclude_cols=exclude_cols
        )

        subsampled = False
        if len(X) > MAX_ROWS:
            idx = X.sample(MAX_ROWS, random_state=RANDOM_STATE).index
            X, y = X.loc[idx], y.loc[idx]
            subsampled = True

        status_container.write(f"✂️ **Step 2/5** — Performing train/test split ({int((1-test_frac)*100)}% train / {int(test_frac*100)}% test)...")
        progress.progress(0.18, text="Step 2/5 — Train / Test Split…")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_frac, random_state=RANDOM_STATE,
                stratify=(y if task_type == "classification" else None),
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_frac, random_state=RANDOM_STATE,
            )

        status_container.write("🛡️ **Step 3/5** — Fitting split-isolated imputers & scalers (Zero Data Leakage)...")
        progress.progress(0.28, text="Step 3/5 — Leakage-free Preprocessing…")
        X_train, pipeline = fit_transform_train(X_train, col_info)
        X_test = transform_test(X_test, col_info, pipeline)

        status_container.write(f"🤖 **Step 4/5** — Fitting up to 16 Machine Learning algorithms ({task_type.capitalize()})...")
        trained, failed = train_all_models(
            X_train, X_test, y_train, y_test,
            task_type, progress, run_cv=run_cv, col_info=col_info
        )
        if not trained:
            st.error("All models failed. Please inspect your dataset.")
            return None

        status_container.write("📊 **Step 5/5** — Ranking leaderboard, calculating feature importances & confusion matrix...")
        progress.progress(0.92, text="Step 5/5 — Ranking & Evaluating Models…")
        trained.sort(key=lambda r: r["_sort_key"], reverse=True)
        best = trained[0]
        best_name = best["Model"]

        tuning_info = None
        if run_tuning:
            status_container.write(f"⚙️ **Hyperparameter Tuning** — Tuning best model ({best_name}) via GridSearch...")
            tuned_model, tuning_info = tune_best_model(
                best["_model"], best_name, X_train, y_train, task_type, progress
            )
            if tuning_info:
                tp = tuned_model.predict(X_test)
                best["_model"] = tuned_model
                best["_predictions"] = tp
                if task_type == "classification":
                    from sklearn.metrics import f1_score as _f1
                    is_bin = (col_info.get("n_classes") == 2)
                    best["F1-Score"] = _f1(y_test, tp, average="binary" if is_bin else "weighted", pos_label=1, zero_division=0)
                    best["_sort_key"] = best["F1-Score"]
                else:
                    from sklearn.metrics import r2_score as _r2
                    best["R²"] = _r2(y_test, tp)
                    best["_sort_key"] = best["R²"]

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

        model_pickle = export_model_artifact(
            best["_model"], pipeline, target_encoder, col_info, task_type
        )

        progress.progress(1.0, text="AutoML Pipeline Complete!")

        return {
            "task_type": task_type,
            "target": target_col,
            "results_df": results_df,
            "best_name": best_name,
            "best_primary": best["_sort_key"],
            "best_time": best["Training Time (s)"],
            "best_predictions": best["_predictions"],
            "best_model": best["_model"],
            "importance": imp,
            "importance_method": imp_method,
            "cm": cm,
            "col_info": col_info,
            "y_test": y_test,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X.shape[1],
            "subsampled": subsampled,
            "failed": failed,
            "tuning_info": tuning_info,
            "run_cv": run_cv,
            "model_pickle": model_pickle,
        }

    except Exception as exc:
        st.error(f"❌ **Pipeline Execution Error**: {exc}")
        return None

# ── SIDEBAR CONTROLS ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sb-logo">'
        '<span style="color:#4f46e5; display:flex; align-items:center;">'
        '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>'
        '</span>'
        '<span class="sb-logo-text"><span style="color:#0f172a;">ML</span><span style="color:#4f46e5;">Nexus</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section">Navigation / Workflow</div>', unsafe_allow_html=True)
    if st.session_state["df"] is not None:
        if st.button("🔄 Upload New Dataset", key="sb_new_df_btn", use_container_width=True):
            st.session_state["df"] = None
            st.session_state["uploaded_name"] = None
            st.session_state["automl_results"] = None
            st.session_state["target_col"] = None
            st.session_state["step"] = 1
            st.rerun()

        if st.session_state["step"] > 1:
            if st.button("⚙️ Dataset Configuration (Stage 2)", key="sb_goto_stage2", use_container_width=True):
                st.session_state["step"] = 2
                st.rerun()

        if st.session_state["automl_results"] is not None:
            if st.button("🏆 View Leaderboard (Stage 4)", key="sb_goto_stage4", use_container_width=True):
                st.session_state["step"] = 4
                st.rerun()
            if st.button("💡 View Diagnostics (Stage 5)", key="sb_goto_stage5", use_container_width=True):
                st.session_state["step"] = 5
                st.rerun()

    st.markdown('<div class="sb-section">Project Info</div>', unsafe_allow_html=True)
    st.caption("MLNexus v3.0 — Multi-Stage Leakage-Free AutoML Engine.")

# ── NAVBAR & STEPPER HEADER ──────────────────────────────────────────────────
st.markdown(navbar_html(), unsafe_allow_html=True)
st.markdown(stepper_bar_html(st.session_state["step"]), unsafe_allow_html=True)

# ── WIZARD STAGE ROUTING ─────────────────────────────────────────────────────

# -----------------------------------------------------------------------------
# STAGE 1: UPLOAD DATASET
# -----------------------------------------------------------------------------
if st.session_state["step"] == 1 or st.session_state["df"] is None:
    st.markdown(
        '<div class="ag-hero">'
        '<div class="ag-hero-badge">⚡ Automated Machine Learning Platform</div>'
        '<h1>Train & Rank ML Models in <span class="ag-hero-gradient-text">1-Click</span></h1>'
        '<p>Upload your dataset. MLNexus performs zero-leakage data preparation, fits Classification, Regression, and Unsupervised algorithms, ranks leaderboard performance, and generates explainable AI metrics.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ag-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ag-card-header">'
        '<div class="ag-card-title">'
        '<span style="color:#4f46e5;">' + ICON_UPLOAD + '</span>'
        '<span>Dataset Upload & Task Selection</span>'
        '</div>'
        '<span style="background:#f5f3ff; color:#4f46e5; border:1px solid #c7d2fe; font-size:0.75rem; font-weight:800; padding:0.25rem 0.6rem; border-radius:6px;">Stage 1 of 5</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_u1, col_u2 = st.columns([1, 1.2])

    with col_u1:
        st.markdown('<p style="font-weight:800; color:#0f172a; font-size:0.92rem; margin-bottom:0.4rem;">1. Select Initial Task Preference</p>', unsafe_allow_html=True)
        cur_idx = TASK_OPTIONS.index(st.session_state["task_type"].capitalize()) if st.session_state["task_type"].capitalize() in TASK_OPTIONS else 0
        sel_task = st.radio(
            "Task Preference", TASK_OPTIONS,
            index=cur_idx, horizontal=True, key="landing_task_radio"
        ).lower()
        st.session_state["task_type"] = sel_task

        st.markdown('<p style="font-weight:700; color:#64748b; font-size:0.82rem; margin-top:1.1rem; margin-bottom:0.4rem;">Or Instant Demo Datasets:</p>', unsafe_allow_html=True)
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            if st.button("Load Sample CSV", key="btn_sample_csv", use_container_width=True):
                try:
                    loaded_df = pd.read_csv("sample_data.csv")
                    st.session_state["df"] = loaded_df
                    st.session_state["uploaded_name"] = "sample_data.csv (Demo Dataset)"
                    st.session_state["target_col"] = "purchased"
                    st.session_state["task_type"] = "classification"
                    st.session_state["step"] = 2
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load sample CSV: {exc}")
        with s_c2:
            if st.button("Load Sample TSV", key="btn_sample_tsv", use_container_width=True):
                try:
                    loaded_df = pd.read_csv("sample_data.tsv", sep="\t")
                    st.session_state["df"] = loaded_df
                    st.session_state["uploaded_name"] = "sample_data.tsv (Demo TSV)"
                    st.session_state["target_col"] = "purchased"
                    st.session_state["task_type"] = "classification"
                    st.session_state["step"] = 2
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load sample TSV: {exc}")

    with col_u2:
        st.markdown('<p style="font-weight:800; color:#0f172a; font-size:0.92rem; margin-bottom:0.4rem;">2. Upload CSV or TSV Dataset File</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV or TSV dataset",
            type=["csv", "tsv"],
            key="landing_file_uploader",
            help="Supports CSV and TSV formats with header rows",
        )
        if uploaded_file is not None:
            try:
                loaded_df = load_dataset(uploaded_file)
                probs = validate_dataset(loaded_df)
                if probs:
                    for p in probs:
                        st.error(p)
                else:
                    st.session_state["df"] = loaded_df
                    st.session_state["uploaded_name"] = uploaded_file.name
                    st.session_state["step"] = 2
                    st.rerun()
            except ValueError as exc:
                st.error(f"❌ **Upload Error**: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STAGE 2: INSPECT & CONFIGURE
# -----------------------------------------------------------------------------
elif st.session_state["step"] == 2:
    df = st.session_state["df"]
    name = st.session_state["uploaded_name"]

    st.markdown(section(f"Dataset Inspection — {name}"), unsafe_allow_html=True)
    st.markdown(stat_row_html(df), unsafe_allow_html=True)

    tab_prev, tab_info, tab_diag, tab_corr = st.tabs([
        "👁️ Data Preview (First 10 rows)",
        "📋 Column Health & Data Types",
        "⚠️ Target & Feature Diagnostics",
        "📈 Feature Correlation Heatmap",
    ])

    with tab_prev:
        st.dataframe(df.head(10), use_container_width=True)

    with tab_info:
        typed = coerce_object_columns_to_numeric(df.copy())
        ci = pd.DataFrame({
            "Column": df.columns,
            "Type": ["Numeric" if pd.api.types.is_numeric_dtype(typed[c]) else "Categorical" for c in df.columns],
            "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
            "Missing %": [round(100 * df[c].isna().mean(), 1) for c in df.columns],
            "Unique Values": [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(ci, use_container_width=True, hide_index=True)

    # Initialize Target Column selection
    target_options = [NONE_UNSUPERVISED] + list(df.columns)
    if st.session_state["target_col"] not in target_options:
        common_targets = ("target","class","label","y","species","diagnosis","outcome","survived","purchased","price","salary","value","score")
        found = next((c for c in df.columns if c.strip().lower() in common_targets), df.columns[-1])
        st.session_state["target_col"] = found

    cur_target = st.session_state["target_col"]
    diag = analyze_target_and_features(df, cur_target, st.session_state["task_type"])

    with tab_diag:
        if diag["warnings"]:
            for w in diag["warnings"]:
                st.warning(f"⚠️ {w}")
        else:
            st.success("✅ Dataset passes initial health diagnostics with no major target or leakage warnings.")

        if diag["class_distribution"]:
            st.markdown("**Class Distribution for Target Variable:**")
            dist_df = pd.DataFrame(list(diag["class_distribution"].items()), columns=["Class Label", "Sample Count"])
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

        if diag["id_cols"]:
            st.info(f"💡 Detected likely ID / Key columns: `{diag['id_cols']}`. Consider excluding them below.")

    # PROMINENT AUTOML LAUNCH & CONFIGURATION CARD
    st.markdown('<div class="ag-card" style="margin-top:1.75rem; border-top: 4px solid #4f46e5;">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ag-card-header">'
        '<div class="ag-card-title">'
        '<span style="color:#4f46e5;">⚡</span>'
        '<span>Configure Target Variable & AutoML Pipeline</span>'
        '</div>'
        f'<span style="background:#f5f3ff; color:#4f46e5; border:1px solid #c7d2fe; font-size:0.75rem; font-weight:800; padding:0.25rem 0.6rem; border-radius:6px;">Stage 2 of 5</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_t1, col_t2 = st.columns([1.5, 1.8])

    with col_t1:
        st.markdown('<p style="font-weight:800; color:#0f172a; font-size:0.9rem; margin-bottom:0.3rem;">Target Variable (Column to Predict)</p>', unsafe_allow_html=True)
        sel_t_idx = target_options.index(cur_target) if cur_target in target_options else 0
        new_target = st.selectbox(
            "Target Column Selector", target_options,
            index=sel_t_idx, label_visibility="collapsed", key="stage2_target_select"
        )
        if new_target != st.session_state["target_col"]:
            st.session_state["target_col"] = new_target
            if new_target == NONE_UNSUPERVISED:
                st.session_state["task_type"] = "unsupervised"
            else:
                st.session_state["task_type"] = auto_detect_task_type(df, new_target)
            st.rerun()

    with col_t2:
        st.markdown('<p style="font-weight:800; color:#0f172a; font-size:0.9rem; margin-bottom:0.3rem;">Select ML Task Mode</p>', unsafe_allow_html=True)
        auto_rec = auto_detect_task_type(df, st.session_state["target_col"])
        st.caption(f"✨ Auto-Detection Recommendation: **{auto_rec.capitalize()}**")

        cur_mode_idx = TASK_OPTIONS.index(st.session_state["task_type"].capitalize()) if st.session_state["task_type"].capitalize() in TASK_OPTIONS else 0
        new_mode = st.radio(
            "Task Mode Radio", TASK_OPTIONS,
            index=cur_mode_idx, horizontal=True, label_visibility="collapsed", key="stage2_task_radio"
        ).lower()
        if new_mode != st.session_state["task_type"]:
            st.session_state["task_type"] = new_mode
            st.rerun()

    with st.expander("⚙️ Advanced Pipeline Controls (Test split %, Cross-Validation, Tuning, Column Exclusion)"):
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            st.session_state["test_size_pct"] = st.slider(
                "Test Split %", min_value=10, max_value=40, value=st.session_state["test_size_pct"], step=5,
                help="Percentage of data held out for final test evaluation"
            )
            st.session_state["run_cv"] = st.checkbox(
                "Enable 3-Fold Cross-Validation", value=st.session_state["run_cv"],
                help="Evaluate 3-fold cross-validation on training rows for robust ranking"
            )
            st.session_state["run_tuning"] = st.checkbox(
                "Tune Winning Model (GridSearch)", value=st.session_state["run_tuning"],
                help="Perform hyperparameter optimization on winning algorithm"
            )
        with c_adv2:
            st.markdown("**Exclude Identifier Columns from Training:**")
            ex_candidates = diag["id_cols"] + [c for c in df.columns if c != st.session_state["target_col"] and c not in diag["id_cols"]]
            sel_ex = st.multiselect(
                "Select columns to drop from feature set:",
                options=[c for c in df.columns if c != st.session_state["target_col"]],
                default=diag["id_cols"],
                key="stage2_exclude_cols"
            )
            st.session_state["exclude_cols"] = sel_ex

    st.markdown("")
    col_act1, col_act2 = st.columns([2, 1])
    with col_act1:
        if st.button("⚡ Launch AutoML Pipeline", type="primary", key="stage2_launch_btn", use_container_width=True):
            st.session_state["step"] = 3
            st.rerun()
    with col_act2:
        if st.button("🔄 Upload Different File", key="stage2_reset_btn", use_container_width=True):
            st.session_state["df"] = None
            st.session_state["uploaded_name"] = None
            st.session_state["step"] = 1
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STAGE 3: TRAIN MODELS (DEDICATED EXECUTION VIEW)
# -----------------------------------------------------------------------------
elif st.session_state["step"] == 3:
    df = st.session_state["df"]
    target_col = st.session_state["target_col"]
    task_type = st.session_state["task_type"]

    st.markdown('<div class="ag-card" style="border-top: 4px solid #4f46e5;">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ag-card-header">'
        '<div class="ag-card-title"><span style="color:#4f46e5;">⚡</span><span>AutoML Engine Training Pipeline Active</span></div>'
        '<span style="background:#f5f3ff; color:#4f46e5; border:1px solid #c7d2fe; font-size:0.75rem; font-weight:800; padding:0.25rem 0.6rem; border-radius:6px;">Stage 3 of 5 — Training...</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.status("⚡ Running MLNexus AutoML Training Engine...", expanded=True) as status:
        res = run_automl_engine(
            df, target_col, task_type,
            st.session_state["test_size_pct"] / 100.0,
            st.session_state["run_cv"],
            st.session_state["run_tuning"],
            st.session_state["exclude_cols"],
            status_container=status
        )

        if res is not None:
            st.session_state["automl_results"] = res
            status.update(label="✅ AutoML Pipeline Complete!", state="complete", expanded=False)
            st.session_state["step"] = 4
            st.rerun()
        else:
            status.update(label="❌ AutoML Pipeline Failed", state="error", expanded=True)
            if st.button("⚙️ Return to Configuration", key="stage3_fail_btn"):
                st.session_state["step"] = 2
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STAGE 4: COMPARE RESULTS (LEADERBOARD & METRICS)
# -----------------------------------------------------------------------------
elif st.session_state["step"] == 4:
    results = st.session_state["automl_results"]
    if results is None:
        st.warning("No completed pipeline results found. Please launch training from Stage 2.")
        if st.button("Go to Configuration"):
            st.session_state["step"] = 2
            st.rerun()
    else:
        rt    = results["task_type"]
        rname = results["best_name"]
        rdf   = results["results_df"]
        p_lbl = ("F1-Score" if rt == "classification" else ("R²" if rt == "regression" else "Silhouette Score"))

        # WINNER HERO CARD
        st.markdown(
            f'<div class="ag-card" style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #ffffff; padding: 1.5rem 1.75rem; border: none;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">'
            f'<div>'
            f'<span style="background:rgba(255,255,255,0.2); color:#ffffff; font-size:0.75rem; font-weight:800; padding:0.25rem 0.75rem; border-radius:9999px; text-transform:uppercase; letter-spacing:0.06em;">🏆 WINNING ALGORITHM</span>'
            f'<h2 style="font-size:1.8rem; font-weight:800; color:#ffffff; margin:0.4rem 0 0.2rem 0; letter-spacing:-0.02em;">{rname}</h2>'
            f'<p style="margin:0; color:rgba(255,255,255,0.85); font-size:0.9rem; font-weight:500;">Outperformed {len(rdf)-1} other algorithms on target dataset</p>'
            f'</div>'
            f'<div style="display:flex; gap:1rem; align-items:center;">'
            f'<div style="background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25); border-radius:12px; padding:0.75rem 1.25rem; text-align:center;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:1.6rem; font-weight:800; color:#ffffff; display:block;">{results["best_primary"]:.4f}</span>'
            f'<span style="font-size:0.7rem; font-weight:700; color:rgba(255,255,255,0.8); text-transform:uppercase;">Top {p_lbl}</span>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25); border-radius:12px; padding:0.75rem 1.25rem; text-align:center;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:1.6rem; font-weight:800; color:#ffffff; display:block;">{results["best_time"]}s</span>'
            f'<span style="font-size:0.7rem; font-weight:700; color:rgba(255,255,255,0.8); text-transform:uppercase;">Train Time</span>'
            f'</div>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(section("Model Leaderboard"), unsafe_allow_html=True)
        st.markdown(leaderboard_html(rdf, rt, rname), unsafe_allow_html=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Export Leaderboard CSV",
                rdf.to_csv(index=False).encode("utf-8"),
                file_name=f"automl_leaderboard_{results['target']}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "📦 Export Fitted Model Pipeline (.pkl)",
                results["model_pickle"],
                file_name=f"trained_model_pipeline_{rname.lower().replace(' ', '_')}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
            )

        st.markdown(section("Performance Comparison Charts"), unsafe_allow_html=True)
        if rt == "classification":
            st.plotly_chart(plot_model_comparison_clf(rdf), use_container_width=True)
        elif rt == "regression":
            st.plotly_chart(plot_model_comparison_reg(rdf), use_container_width=True)
        else:
            st.plotly_chart(plot_model_comparison_unsupervised(rdf), use_container_width=True)

        if results["run_cv"]:
            cv_fig = plot_cv_comparison(rdf, rt)
            if cv_fig:
                st.plotly_chart(cv_fig, use_container_width=True)

        st.markdown("")
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        with col_nav1:
            if st.button("💡 View Diagnostics & Explainability (Stage 5)", type="primary", use_container_width=True):
                st.session_state["step"] = 5
                st.rerun()
        with col_nav2:
            if st.button("⚙️ Back to Configuration (Stage 2)", use_container_width=True):
                st.session_state["step"] = 2
                st.rerun()
        with col_nav3:
            if st.button("🔄 Upload New Dataset", use_container_width=True):
                st.session_state["df"] = None
                st.session_state["uploaded_name"] = None
                st.session_state["step"] = 1
                st.rerun()

# -----------------------------------------------------------------------------
# STAGE 5: EXPLAIN & EXPORT (EXPLAINABLE AI & DIAGNOSTICS)
# -----------------------------------------------------------------------------
elif st.session_state["step"] == 5:
    results = st.session_state["automl_results"]
    if results is None:
        st.warning("No pipeline results available.")
        if st.button("Go to Configuration"):
            st.session_state["step"] = 2
            st.rerun()
    else:
        rt    = results["task_type"]
        rname = results["best_name"]
        rdf   = results["results_df"]
        p_lbl = ("F1-Score" if rt == "classification" else ("R²" if rt == "regression" else "Silhouette Score"))
        algo_d = (ALGORITHM_INFO if rt == "classification" else (REGRESSION_INFO if rt == "regression" else UNSUPERVISED_INFO))

        st.markdown(section("Explainable AI Insights"), unsafe_allow_html=True)
        desc = algo_d.get(rname, "")

        st.markdown(
            '<div class="ag-explain">'
            f'<h3>Why {rname} Outperformed All Models</h3>'
            f'<p>{desc}</p>'
            '<div class="ag-met-trio">'
            f'<div class="ag-met-chip"><span class="mv">{results["best_primary"]:.4f}</span><span class="ml">{p_lbl}</span></div>'
            f'<div class="ag-met-chip"><span class="mv">{results["best_time"]}s</span><span class="ml">Train time</span></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(section("Detailed Model Diagnostics"), unsafe_allow_html=True)

        if rt == "classification":
            c_a, c_b = st.columns(2)
            with c_a:
                st.plotly_chart(
                    plot_confusion_matrix(results["cm"], results["col_info"]["class_names"], rname),
                    use_container_width=True
                )
            with c_b:
                if results["importance"] is not None:
                    top5 = results["importance"].head(5).copy()
                    top5.insert(0, "#", range(1, len(top5) + 1))
                    top5["Importance"] = top5["Importance"].round(4)
                    st.dataframe(top5, hide_index=True, use_container_width=True)
                    st.caption(results["importance_method"])

        elif rt == "regression":
            y_arr = results["y_test"].values if hasattr(results["y_test"], "values") else np.asarray(results["y_test"])
            c_a, c_b = st.columns(2)
            with c_a:
                st.plotly_chart(plot_actual_vs_predicted(y_arr, results["best_predictions"], rname), use_container_width=True)
            with c_b:
                st.plotly_chart(plot_residuals(y_arr, results["best_predictions"], rname), use_container_width=True)

        else:
            c_a, c_b = st.columns(2)
            with c_a:
                st.plotly_chart(plot_cluster_pca_scatter(results["X_scaled"], results["best_predictions"], rname), use_container_width=True)
            with c_b:
                if results["best_predictions"] is not None:
                    gc = pd.Series(results["best_predictions"]).value_counts().reset_index()
                    gc.columns = ["Cluster / Group ID", "Sample Count"]
                    st.dataframe(gc, use_container_width=True, hide_index=True)

        if results["importance"] is not None:
            st.markdown(section("Feature Importance Ranking"), unsafe_allow_html=True)
            st.plotly_chart(plot_feature_importance(results["importance"], results["importance_method"]), use_container_width=True)

        st.markdown(section("Algorithm Reference Documentation"), unsafe_allow_html=True)
        with st.expander(f"📚 How each of the {len(rdf)} trained algorithms works"):
            for name, d_text in algo_d.items():
                trophy = " 🏆 (Winner)" if name == rname else ""
                st.markdown(f"**{name}{trophy}** — {d_text}")

        st.markdown("")
        c_nav1, c_nav2, c_nav3 = st.columns(3)
        with c_nav1:
            if st.button("🏆 Back to Leaderboard (Stage 4)", type="primary", use_container_width=True):
                st.session_state["step"] = 4
                st.rerun()
        with c_nav2:
            if st.button("⚙️ Back to Configuration (Stage 2)", use_container_width=True):
                st.session_state["step"] = 2
                st.rerun()
        with c_nav3:
            if st.button("🔄 Upload New Dataset", use_container_width=True):
                st.session_state["df"] = None
                st.session_state["uploaded_name"] = None
                st.session_state["step"] = 1
                st.rerun()
