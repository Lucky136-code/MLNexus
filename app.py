"""
MLNexus Streamlit Application — Modern AutoML Platform.
"""

import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from charts import (plot_actual_vs_predicted, plot_cluster_pca_scatter,
                    plot_confusion_matrix, plot_correlation_heatmap,
                    plot_cv_comparison, plot_feature_importance,
                    plot_model_comparison_clf, plot_model_comparison_reg,
                    plot_model_comparison_unsupervised, plot_residuals,
                    plot_training_time)
from models import (ALGORITHM_INFO, CLASSIFICATION_RESULT_COLUMNS,
                    REGRESSION_INFO, REGRESSION_RESULT_COLUMNS,
                    UNSUPERVISED_INFO, UNSUPERVISED_RESULT_COLUMNS,
                    get_feature_importance, train_all_models,
                    train_unsupervised_models, tune_best_model)
from preprocessing import (auto_detect_task_type, coerce_object_columns_to_numeric,
                            fit_transform_train, load_dataset,
                            prepare_features_and_target, prepare_unsupervised_features,
                            transform_test, validate_dataset)

warnings.filterwarnings("ignore")

MAX_ROWS     = 10_000
RANDOM_STATE = 42
TASK_OPTIONS = ["Classification", "Regression", "Unsupervised"]

# Model family styling
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
ICON_PREPROCESS = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>"""
ICON_TRAIN = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 4a3.5 3.5 0 0 0-3.5 3.5C4 8 3 9.5 3 11.5c0 2 1.5 3.5 3.5 3.5M9.5 20c-2.5 0-3.5-1.5-3.5-2.5"/><path d="M9.5 4c1.5 0 2.5 1 2.5 2.5V17.5c0 1.5-1 2.5-2.5 2.5"/><line x1="12" y1="12" x2="15" y2="12"/><rect x="15" y="3" width="6" height="5" rx="1"/><line x1="18" y1="8" x2="18" y2="10"/><rect x="15" y="10" width="6" height="5" rx="1"/><line x1="18" y1="15" x2="18" y2="17"/><circle cx="18" cy="19.5" r="2"/><path d="M18 16.5v.5M18 22v.5M15.5 19.5h.5M20 19.5h.5"/></svg>"""
ICON_COMPARE = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="2" y1="21" x2="22" y2="21"/><rect x="4" y="3" width="4" height="18"/><rect x="10" y="9" width="4" height="12"/><line x1="10" y1="13" x2="14" y2="13"/><rect x="16" y="15" width="4" height="6"/></svg>"""
ICON_EXPLAIN = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>"""
ICON_CHECK = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"""

# Modern Custom CSS
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Reset & Base Fonts */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stMarkdown, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0f172a !important;
}

[data-testid="stAppViewContainer"] {
    background: #f8fafc !important;
}

/* Hide Streamlit Chrome & Position Styled Hamburger Button */
#MainMenu, footer, .stDeployButton {
    visibility: hidden !important; height: 0 !important; display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    overflow: visible !important;
    z-index: 100000 !important;
}

/* ☰ Sidebar Toggle Button */
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

/* Container Spacing */
.main .block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1280px !important;
}

/* Sidebar Styling */
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

/* Sidebar form labels */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Primary Action Buttons (Electric Gradient) */
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

/* Secondary Buttons */
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

/* Download Button */
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

/* Radio Control Labels & Options */
[data-testid="stRadio"] > label {
    font-weight: 700 !important;
    color: #0f172a !important;
    font-size: 0.9rem !important;
}
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span,
div[data-baseweb="radio"] label span {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* File Uploader Container & Dropzone */
[data-testid="stFileUploader"] section {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    padding: 1.25rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #6366f1 !important;
    background: #f5f3ff !important;
}

/* File Uploader Browse Button */
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploader"] button,
button[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #4f46e5 !important;
    border: 1.5px solid #6366f1 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.12) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] section button:hover,
[data-testid="stFileUploader"] button:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    background: #4f46e5 !important;
    color: #ffffff !important;
    border-color: #4f46e5 !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
}

/* File Uploader Button text & icons */
[data-testid="stFileUploader"] section button *,
[data-testid="stFileUploader"] button *,
button[data-testid="stBaseButton-secondary"] * {
    color: inherit !important;
    fill: currentColor !important;
    stroke: currentColor !important;
    font-weight: 700 !important;
}

/* File Uploader Helper & Instructions Text */
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section div[data-testid="stMarkdownContainer"] p,
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.35rem !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03) !important;
}
div[data-testid="stMetric"] label {
    font-size: 0.75rem !important; font-weight: 800 !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
    color: #64748b !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important; font-weight: 700 !important;
    color: #0f172a !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.5rem !important;
    border-bottom: 2px solid #e2e8f0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 8px 8px 0 0 !important;
    font-weight: 700 !important; color: #64748b !important;
    padding: 0.7rem 1.4rem !important; font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    color: #4f46e5 !important;
    border-bottom: 3px solid #4f46e5 !important;
    background: #ffffff !important;
}

/* Navbar */
.ag-nav {
    display: flex; align-items: center; justify-content: center;
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 1.5rem; height: 60px;
    margin: 0 -2rem 1.75rem -2rem;
    position: sticky; top: 0; z-index: 9999;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}
.ag-nav-logo-center {
    display: flex; align-items: center; justify-content: center; gap: 0.2rem;
}

/* Custom Cards */
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

/* Stepper Bar */
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
.ag-pip-node.nd-pending {
    background: #f1f5f9 !important; color: #94a3b8 !important;
    border: 1.5px solid #cbd5e1;
}
.ag-pip-node.nd-done {
    background: #10b981 !important; color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}
.ag-pip-node.nd-active {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important; color: #ffffff !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.25);
    animation: pip-pulse 2s ease-in-out infinite;
}
@keyframes pip-pulse {
    0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
    100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
}
.ag-pip-lbl {
    font-size: 0.78rem; font-weight: 700;
    color: #64748b; white-space: nowrap;
}
.ag-pip-lbl.done { color: #059669; }
.ag-pip-lbl.active { color: #4f46e5; }
.ag-pip-conn {
    height: 3px; flex: 1; max-width: 80px; margin: 0 0.5rem;
    border-radius: 9999px; transition: background 0.3s ease;
}
.ag-pip-conn.done { background: #10b981; }
.ag-pip-conn.active { background: linear-gradient(90deg, #10b981, #6366f1); }
.ag-pip-conn.pending { background: #e2e8f0; }

/* Stat Row Chips */
.ag-stat-row { display:flex; flex-wrap:wrap; gap:0.75rem; margin-bottom:1.5rem; }
.ag-stat-chip {
    display:inline-flex; align-items:center; gap:0.5rem;
    background:#ffffff; border:1px solid #e2e8f0;
    border-radius:10px; padding:0.5rem 1.15rem;
    font-size:0.85rem; font-weight:700; color:#334155;
    box-shadow:0 2px 6px rgba(15,23,42,0.03);
}
.ag-stat-chip .ag-stat-lbl { color:#64748b; font-weight:600; }
.ag-stat-chip .v {
    font-family:'JetBrains Mono',monospace;
    font-weight:800; color:#0f172a;
}
.ag-stat-chip .ag-stat-sub { color:#94a3b8; font-weight:600; font-size:0.78rem; }

/* Landing Hero CTA */
.ag-hero {
    text-align: center; padding: 2.5rem 1rem 2rem;
}
.ag-hero-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: #f5f3ff; border: 1px solid #c7d2fe;
    border-radius: 9999px; padding: 0.4rem 1rem;
    color: #4f46e5; font-size: 0.82rem; font-weight: 700;
    margin-bottom: 1.25rem;
}
.ag-hero h1 {
    font-size: 2.35rem; font-weight: 800; color: #0f172a;
    letter-spacing: -0.03em; margin-bottom: 0.75rem;
    line-height: 1.2;
}
.ag-hero-gradient-text {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.ag-hero p {
    font-size: 1.05rem; color: #475569; max-width: 720px;
    margin: 0 auto 2rem auto; line-height: 1.6; font-weight: 500;
}

/* Feature Cards */
.ag-feat-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 1.6rem 1.4rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    height: 100%; transition: all 0.2s ease;
}
.ag-feat-card:hover {
    border-color: #c7d2fe;
    transform: translateY(-3px);
    box-shadow: 0 12px 28px -4px rgba(99, 102, 241, 0.12);
}
.ag-feat-icon {
    width: 48px; height: 48px; border-radius: 12px;
    background: #f5f3ff; color: #4f46e5;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 1rem; border: 1px solid #ddd6fe;
}
.ag-feat-title {
    font-size: 1.02rem; font-weight: 800;
    color: #0f172a; margin-bottom: 0.45rem;
}
.ag-feat-desc { font-size: 0.86rem; color: #64748b; line-height: 1.6; }

/* Leaderboard */
.ag-lb { display:flex; flex-direction:column; gap:0.5rem; margin:0.75rem 0 1.25rem 0; }
.ag-lb-row {
    display:flex; align-items:center; gap:1rem;
    background:#ffffff; border-radius:10px;
    border:1px solid #e2e8f0; padding:0.75rem 1.15rem;
    transition: all 0.2s ease;
}
.ag-lb-row:hover {
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    transform: translateY(-1px);
    border-color: #c7d2fe;
}
.ag-lb-row.winner {
    border-left: 4px solid #4f46e5;
    background: #faf5ff;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.12);
}
.ag-lb-rank {
    font-family:'JetBrains Mono',monospace;
    font-size:0.88rem; font-weight:800;
    color:#64748b; width:32px; text-align:center; flex-shrink:0;
}
.ag-win-badge {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #ffffff;
    font-size: 0.7rem; font-weight: 800;
    padding: 0.2rem 0.55rem; border-radius: 6px;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
}
.ag-lb-name {
    font-size:0.92rem; font-weight:700; color:#0f172a;
    flex:1; min-width:130px;
}
.ag-lb-badge {
    font-size:0.7rem; font-weight:800;
    padding:0.25rem 0.65rem; border-radius:6px;
    white-space:nowrap; flex-shrink:0;
}
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
.ag-lb-bar-track {
    flex:1; height:8px; background:#f1f5f9;
    border-radius:9999px; overflow:hidden;
}
.ag-lb-bar-fill {
    height:100%; border-radius:9999px;
    background:#64748b;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
.ag-lb-bar-fill.w { background: linear-gradient(90deg, #4f46e5, #7c3aed); }
.ag-lb-score {
    font-family:'JetBrains Mono',monospace;
    font-size:0.88rem; font-weight:800; color:#0f172a;
    width:56px; text-align:right; flex-shrink:0;
}
.ag-lb-time {
    font-size:0.78rem; color:#64748b; font-weight:600;
    width:52px; text-align:right; flex-shrink:0;
}

/* Leaderboard column header */
.ag-lb-hdr {
    display:flex; align-items:center; gap:1rem;
    padding:0 1.15rem 0.4rem 1.15rem;
    font-size:0.72rem; font-weight:800;
    letter-spacing:0.07em; text-transform:uppercase; color:#64748b;
}
.ag-lb-hdr-rank  { width:32px; flex-shrink:0; text-align:center; }
.ag-lb-hdr-name  { flex:1; min-width:130px; }
.ag-lb-hdr-badge { width:68px; flex-shrink:0; }
.ag-lb-hdr-bar   { flex:2; min-width:90px; }
.ag-lb-hdr-time  { width:52px; text-align:right; flex-shrink:0; }

/* Explanation Card */
.ag-explain {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #4f46e5;
    border-radius: 12px;
    padding: 1.5rem 1.75rem; margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
.ag-explain h3 {
    font-size:1.1rem; font-weight:800; color:#0f172a;
    margin:0 0 0.6rem 0;
}
.ag-explain p {
    font-size:0.92rem; color:#334155; line-height:1.65;
    margin:0 0 1.1rem 0; font-weight:500;
}
.ag-met-trio { display:flex; gap:0.75rem; flex-wrap:wrap; }
.ag-met-chip {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:10px; padding:0.65rem 1.1rem;
    text-align:center; min-width:95px;
}
.ag-met-chip .mv {
    font-family:'JetBrains Mono',monospace;
    font-size:1.15rem; font-weight:800; color:#0f172a; display:block;
}
.ag-met-chip .ml {
    font-size:0.7rem; font-weight:800; color:#64748b;
    text-transform:uppercase; letter-spacing:0.05em;
}

.ag-ch-hdr { margin:1.5rem 0 0.5rem 0; }
.ag-ch-title { font-size:0.95rem; font-weight:800; color:#0f172a; margin:0; }
.ag-ch-sub   { font-size:0.8rem; color:#64748b; margin:0.15rem 0 0 0; font-weight:500; }

.ag-sec {
    font-size:0.75rem; font-weight:800;
    letter-spacing:0.09em; text-transform:uppercase;
    color:#4f46e5; margin:2.25rem 0 0.9rem 0;
    display:flex; align-items:center; gap:0.6rem;
}
.ag-sec::after {
    content:''; flex:1; height:1px; background:#e2e8f0;
}

@media (max-width: 768px) {
    .main .block-container { padding: 0.5rem 0.75rem 2rem 0.75rem !important; }
    .ag-hero h1 { font-size: 1.75rem !important; }
}
</style>
"""

# UI Component Helpers
def navbar_html(task_type: str = "") -> str:
    return f"""
    <div class="ag-nav">
        <div class="ag-nav-logo-center">
            <span style="color:#4f46e5; display:flex; align-items:center;">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            </span>
            <span style="font-weight:800; color:#0f172a; font-size:1.35rem; letter-spacing:-0.03em; margin-left:6px;">ML</span><span style="font-weight:800; color:#4f46e5; font-size:1.35rem; letter-spacing:-0.03em;">Nexus</span>
            <span style="background:#f5f3ff; color:#4f46e5; border:1px solid #c7d2fe; font-size:0.68rem; font-weight:800; padding:0.15rem 0.55rem; border-radius:9999px; margin-left:8px;">AutoML Engine v2.4</span>
        </div>
    </div>"""


def pipeline_html(step: int) -> str:
    """step: 0=nothing, 1=uploaded, 2=training, 3=comparing, 4=explaining, 5=done"""
    labels = ["Upload Data", "Preprocess Prep", "Train 16 Models", "Compare Leaderboard", "Explain AI"]
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


def leaderboard_html(results_df: pd.DataFrame,
                     task_type: str, best_name: str) -> str:
    primary = ("F1-Score" if task_type == "classification"
               else ("R²" if task_type == "regression"
                     else "Silhouette Score"))
    max_s   = max(results_df[primary].max(), 1e-9)

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
        rank   = int(row["Rank"])
        name   = str(row["Model"])
        fam    = str(row["Family"])
        score  = float(row[primary])
        t      = float(row["Training Time (s)"])
        win    = (name == best_name)
        pct    = max(0.0, min(100.0, score / max_s * 100))

        rank_html = (
            '<div class="ag-lb-rank w"><span class="ag-win-badge">#1 WINNER</span></div>' if win
            else f'<div class="ag-lb-rank">#{rank}</div>'
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

    rt = results["task_type"]
    try:
        row = rdf[rdf["Model"] == name].iloc[0]
        if rt == "classification":
            extras = (
                f'<div class="ag-met-chip"><span class="mv">{row["Accuracy"]:.4f}</span><span class="ml">Accuracy</span></div>'
                f'<div class="ag-met-chip"><span class="mv">{row["Precision"]:.4f}</span><span class="ml">Precision</span></div>'
                f'<div class="ag-met-chip"><span class="mv">{row["Recall"]:.4f}</span><span class="ml">Recall</span></div>'
            )
        elif rt == "regression":
            extras = (
                f'<div class="ag-met-chip"><span class="mv">{row["MAE"]:.4f}</span><span class="ml">MAE</span></div>'
                f'<div class="ag-met-chip"><span class="mv">{row["RMSE"]:.4f}</span><span class="ml">RMSE</span></div>'
            )
        else:
            ch_val = row["Calinski-Harabasz"]
            db_val = row["Davies-Bouldin"]
            ch_str = f"{ch_val:.1f}" if pd.notna(ch_val) else "N/A"
            db_str = f"{db_val:.4f}" if pd.notna(db_val) else "N/A"
            extras = (
                f'<div class="ag-met-chip"><span class="mv">{ch_str}</span><span class="ml">Calinski-Harabasz</span></div>'
                f'<div class="ag-met-chip"><span class="mv">{db_str}</span><span class="ml">Davies-Bouldin</span></div>'
            )
    except Exception:
        extras = ""

    return (
        '<div class="ag-explain">'
        f'<h3>Why {name} Outperformed All Models</h3>'
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


# Pipeline execution definition
def run_automl(df, target_col, task_type, test_frac, run_cv, run_tuning, status_container=None):
    progress = st.progress(0.0, text="Starting pipeline…")
    try:
        if task_type == "unsupervised":
            if status_container:
                status_container.write("📌 **Step 1/3** — Imputing missing values & feature scaling...")
            progress.progress(0.15, text="Step 1/3 — Preparing features & scaling…")
            X_scaled, col_info = prepare_unsupervised_features(df)

            if status_container:
                status_container.write("🤖 **Step 2/3** — Fitting 10 Unsupervised Learning algorithms (Clustering, PCA, Outlier Detection)...")
            progress.progress(0.40, text="Step 2/3 — Training unsupervised models…")
            trained, failed = train_unsupervised_models(X_scaled, progress)

            if not trained:
                st.error("All unsupervised models failed. Please check your dataset.")
                return None

            if status_container:
                status_container.write("📊 **Step 3/3** — Evaluating Silhouette Scores & PCA cluster projections...")
            progress.progress(0.90, text="Step 3/3 — Evaluating clusters & anomalies…")
            trained.sort(key=lambda r: r["_sort_key"], reverse=True)
            best = trained[0]
            best_name = best["Model"]

            results_df = pd.DataFrame(trained)[[c for c in UNSUPERVISED_RESULT_COLUMNS if c != "Rank"]]
            results_df.insert(0, "Rank", range(1, len(results_df) + 1))

            progress.progress(1.0, text="AutoML complete!")

            return {
                "task_type":         task_type,
                "target":            "N/A (Unsupervised)",
                "results_df":        results_df,
                "best_name":         best_name,
                "best_primary":      best["_sort_key"],
                "best_time":         best["Training Time (s)"],
                "best_predictions":  best["_predictions"],
                "best_category":     best.get("_category", "clustering"),
                "importance":        None,
                "importance_method": None,
                "cm":                None,
                "col_info":          col_info,
                "y_test":            None,
                "X_scaled":          X_scaled,
                "n_train":           len(X_scaled),
                "n_test":            len(X_scaled),
                "n_features":        X_scaled.shape[1],
                "missing_imputed":   col_info["missing_before"],
                "n_numeric":         len(col_info["numeric_cols"]),
                "n_categorical":     len(col_info["categorical_cols"]),
                "dropped_target":    0,
                "subsampled":        False,
                "failed":            failed,
                "tuning_info":       None,
                "run_cv":            False,
            }

        # Step 1 — prepare
        if status_container:
            status_container.write("📌 **Step 1/5** — Preparing feature matrices & encoding target variable...")
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
        if status_container:
            status_container.write(f"✂️ **Step 2/5** — Performing train/test split ({int((1-test_frac)*100)}% train / {int(test_frac*100)}% test)...")
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
        if status_container:
            status_container.write("🛡️ **Step 3/5** — Fitting split-isolated mean/median imputers & feature scalers...")
        progress.progress(0.28, text="Step 3/5 — Leakage-free preprocessing…")
        X_train, pipeline = fit_transform_train(X_train, col_info)
        X_test             = transform_test(X_test, col_info, pipeline)

        # Step 4 — train
        if status_container:
            status_container.write(f"🤖 **Step 4/5** — Fitting up to 16 Machine Learning algorithms ({task_type.capitalize()})...")
        trained, failed = train_all_models(
            X_train, X_test, y_train, y_test,
            task_type, progress, run_cv=run_cv,
        )
        if not trained:
            st.error("All models failed. Please check your dataset.")
            return None

        # Step 5 — evaluate
        if status_container:
            status_container.write("📊 **Step 5/5** — Ranking leaderboard, calculating feature importances & confusion matrices...")
        progress.progress(0.92, text="Step 5/5 — Ranking & evaluating…")
        trained.sort(key=lambda r: r["_sort_key"], reverse=True)
        best      = trained[0]
        best_name = best["Model"]

        # Optional tuning
        tuning_info = None
        if run_tuning:
            if status_container:
                status_container.write(f"⚙️ **Tuning** — Hyperparameter tuning best model ({best_name})...")
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
        st.error(f"❌ **Validation Error**: {exc}")
        return None
    except Exception as exc:
        st.error(f"❌ **Pipeline Execution Error**: {exc}")
        return None


# Streamlit App Initialization
st.set_page_config(
    page_title="MLNexus — Automated ML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# Session state initialization & Callbacks
if "task_type" not in st.session_state:
    st.session_state["task_type"] = "classification"
if "use_sample_csv" not in st.session_state:
    st.session_state["use_sample_csv"] = False
if "use_sample_tsv" not in st.session_state:
    st.session_state["use_sample_tsv"] = False
if "should_run_automl" not in st.session_state:
    st.session_state["should_run_automl"] = False

def trigger_automl_callback():
    st.session_state["should_run_automl"] = True

NONE_UNSUPERVISED = "None (Unsupervised Learning)"

def update_task_from_sb():
    st.session_state["task_type"] = st.session_state["sb_task_radio"].lower()
    st.session_state["task_type_user_set"] = True
    st.session_state["automl_results"] = None

def update_task_from_main():
    st.session_state["task_type"] = st.session_state["main_landing_task_radio"].lower()
    st.session_state["task_type_user_set"] = True
    st.session_state["automl_results"] = None

def update_task_from_overview():
    st.session_state["task_type"] = st.session_state["main_overview_task_radio"].lower()
    st.session_state["task_type_user_set"] = True
    st.session_state["automl_results"] = None

def update_target_from_sb():
    st.session_state["target_col"] = st.session_state["sb_target_col"]
    st.session_state["automl_results"] = None
    if st.session_state["target_col"] == NONE_UNSUPERVISED:
        st.session_state["task_type"] = "unsupervised"
    elif not st.session_state.get("task_type_user_set", False):
        st.session_state["task_type"] = auto_detect_task_type(st.session_state.get("_active_df"), st.session_state["target_col"])

def update_target_from_main():
    st.session_state["target_col"] = st.session_state["main_overview_target_select"]
    if "sb_target_col" in st.session_state:
        st.session_state["sb_target_col"] = st.session_state["main_overview_target_select"]
    st.session_state["automl_results"] = None
    if st.session_state["target_col"] == NONE_UNSUPERVISED:
        st.session_state["task_type"] = "unsupervised"
    elif not st.session_state.get("task_type_user_set", False):
        st.session_state["task_type"] = auto_detect_task_type(st.session_state.get("_active_df"), st.session_state["target_col"])


# Sidebar Controls
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

    cur_task_name = st.session_state.get("task_type", "classification").capitalize()
    cur_task_idx = TASK_OPTIONS.index(cur_task_name) if cur_task_name in TASK_OPTIONS else 0

    st.markdown('<div class="sb-section">Task Mode</div>', unsafe_allow_html=True)
    sb_task = st.radio(
        "Task type", TASK_OPTIONS,
        index=cur_task_idx,
        horizontal=True, label_visibility="collapsed", key="sb_task_radio",
        on_change=update_task_from_sb
    ).lower()

    task_type = st.session_state.get("task_type", "classification")

    st.markdown('<div class="sb-section">Dataset Source (.csv / .tsv)</div>', unsafe_allow_html=True)
    uploaded_file_sb = st.file_uploader(
        "Upload CSV / TSV", type=["csv", "tsv"],
        label_visibility="collapsed", key="sb_file_uploader",
        help="Supports CSV and TSV formats with headers",
    )

    test_size_pct = 20
    run_cv = False
    run_tuning = False

    # ── Load dataset active state ─────────────────────────────────────────────
    df, uploaded_name = None, None

    main_uploaded_file = st.session_state.get("main_landing_file_uploader")
    active_uploaded_file = uploaded_file_sb if uploaded_file_sb is not None else main_uploaded_file

    if st.session_state.get("use_sample_csv"):
        try:
            df = pd.read_csv("sample_data.csv")
            uploaded_name = "sample_data.csv (Demo Dataset)"
        except FileNotFoundError:
            st.session_state["use_sample_csv"] = False

    elif st.session_state.get("use_sample_tsv"):
        try:
            df = pd.read_csv("sample_data.tsv", sep="\t")
            uploaded_name = "sample_data.tsv (Demo TSV Dataset)"
        except FileNotFoundError:
            st.session_state["use_sample_tsv"] = False

    elif active_uploaded_file is not None:
        st.session_state["use_sample_csv"] = False
        st.session_state["use_sample_tsv"] = False
        try:
            df = load_dataset(active_uploaded_file)
            uploaded_name = active_uploaded_file.name
        except ValueError as exc:
            st.error(str(exc))

    st.session_state["_active_df"] = df

    if df is not None:
        probs = validate_dataset(df)
        if probs:
            for p in probs:
                st.error(p)
            df = None

    # Clear previous pipeline results when new data is loaded
    if df is not None:
        cache_key = f"{uploaded_name}"
        if "loaded_key" not in st.session_state:
            st.session_state["loaded_key"] = cache_key
            st.session_state["task_type_user_set"] = False
        elif st.session_state["loaded_key"] != cache_key:
            st.session_state["loaded_key"]     = cache_key
            st.session_state["automl_results"] = None
            st.session_state["task_type_user_set"] = False
            if "target_col" in st.session_state:
                del st.session_state["target_col"]

        st.success(f"**{uploaded_name}**  \n{df.shape[0]:,} rows × {df.shape[1]} columns")

        target_options = [NONE_UNSUPERVISED] + list(df.columns)

        common_targets = ("target","class","label","y","species","diagnosis",
                          "outcome","survived","purchased","price","salary",
                          "value","score","result")
        default_col_idx = next(
            (i for i, c in enumerate(df.columns)
             if c.strip().lower() in common_targets),
            df.shape[1] - 1,
        )
        default_target = df.columns[default_col_idx]

        if "target_col" not in st.session_state or (st.session_state["target_col"] not in df.columns and st.session_state["target_col"] != NONE_UNSUPERVISED):
            st.session_state["target_col"] = default_target

        # Auto-detect task mode if not manually set by user
        if not st.session_state.get("task_type_user_set", False):
            st.session_state["task_type"] = auto_detect_task_type(df, st.session_state["target_col"])
            task_type = st.session_state["task_type"]

        cur_target = st.session_state.get("target_col", default_target)
        sel_idx = 0 if cur_target == NONE_UNSUPERVISED else (target_options.index(cur_target) if cur_target in target_options else target_options.index(default_target))

        st.markdown('<div class="sb-section">Target Variable</div>', unsafe_allow_html=True)
        sb_target = st.selectbox(
            "Target column", target_options,
            index=sel_idx,
            label_visibility="collapsed", key="sb_target_col",
            on_change=update_target_from_sb
        )

        if "last_target" not in st.session_state:
            st.session_state["last_target"] = st.session_state["target_col"]
        elif st.session_state["last_target"] != st.session_state["target_col"]:
            st.session_state["last_target"]    = st.session_state["target_col"]
            st.session_state["automl_results"] = None
            if not st.session_state.get("task_type_user_set", False):
                st.session_state["task_type"] = auto_detect_task_type(df, st.session_state["target_col"])
                task_type = st.session_state["task_type"]

        st.markdown("")
        start_training_sb = st.button(
            "⚡ Start AutoML Pipeline", type="primary", key="sb_start_automl_btn", on_click=trigger_automl_callback
        )
    else:
        st.info("Upload a CSV or TSV file to start.")
        start_training_sb = False

    target_col = st.session_state.get("target_col")


# NAVBAR
st.markdown(navbar_html(task_type), unsafe_allow_html=True)

# MAIN SCREEN BUTTON EVALUATION & STEPPER DETERMINATION
start_training = st.session_state.get("should_run_automl", False)
_results_pre = st.session_state.get("automl_results")

# Execute training pipeline immediately if Launch button was clicked
if start_training and df is not None:
    _pip_step = 2
elif _results_pre is not None:
    _pip_step = 5
elif df is not None:
    _pip_step = 1
else:
    _pip_step = 0

st.markdown(pipeline_html(_pip_step), unsafe_allow_html=True)


# MAIN CONTENT ROUTING
if df is None:
    st.markdown(
        '<div class="ag-hero">'
        '<div class="ag-hero-badge">⚡ Automated Machine Learning Platform</div>'
        '<h1>Train & Rank ML Models in <span class="ag-hero-gradient-text">1-Click</span></h1>'
        '<p>Upload your dataset. MLNexus performs zero-leakage data preparation, fits Classification, Regression, and Unsupervised algorithms, ranks leaderboard performance, and generates explainable AI metrics.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Prominent Upload & Configuration Card on Main Page
    st.markdown('<div class="ag-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ag-card-header">'
        '<div class="ag-card-title">'
        '<span style="color:#4f46e5;">' + ICON_UPLOAD + '</span>'
        '<span>Dataset Upload & Task Setup</span>'
        '</div>'
        '<span style="background:#f5f3ff; color:#4f46e5; border:1px solid #c7d2fe; font-size:0.75rem; font-weight:800; padding:0.25rem 0.6rem; border-radius:6px;">Step 1 of 5</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_up1, col_up2 = st.columns([1, 1.2])

    with col_up1:
        st.markdown('<p style="font-weight:800; color:#0f172a; font-size:0.92rem; margin-bottom:0.4rem;">Select ML Task Mode</p>', unsafe_allow_html=True)
        cur_landing_idx = TASK_OPTIONS.index(st.session_state.get("task_type", "classification").capitalize()) if st.session_state.get("task_type", "classification").capitalize() in TASK_OPTIONS else 0
        m_task = st.radio(
            "Task Type", TASK_OPTIONS,
            index=cur_landing_idx,
            horizontal=True, key="main_landing_task_radio",
            on_change=update_task_from_main
        ).lower()

        st.markdown('<p style="font-weight:700; color:#64748b; font-size:0.82rem; margin-top:1.1rem; margin-bottom:0.4rem;">Instant Demo Datasets:</p>', unsafe_allow_html=True)
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            if st.button("Load Sample CSV", key="btn_sample_csv", use_container_width=True):
                st.session_state["use_sample_csv"] = True
                st.session_state["use_sample_tsv"] = False
                st.session_state["automl_results"] = None
                st.rerun()
        with s_col2:
            if st.button("Load Sample TSV", key="btn_sample_tsv", use_container_width=True):
                st.session_state["use_sample_tsv"] = True
                st.session_state["use_sample_csv"] = False
                st.session_state["automl_results"] = None
                st.rerun()

    with col_up2:
        st.markdown('<p style="font-weight:800; color:#0f172a; font-size:0.92rem; margin-bottom:0.4rem;">Upload CSV or TSV File</p>', unsafe_allow_html=True)
        main_uploaded_file = st.file_uploader(
            "Upload CSV or TSV dataset",
            type=["csv", "tsv"],
            key="main_landing_file_uploader",
            help="Supports CSV and TSV formats with header rows",
        )
        if main_uploaded_file is not None:
            st.session_state["use_sample_csv"] = False
            st.session_state["use_sample_tsv"] = False
            try:
                df = load_dataset(main_uploaded_file)
                uploaded_name = main_uploaded_file.name
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        (ICON_UPLOAD, "1. Zero-Leakage Preprocessing", "Automatic column typing, split-fitted mean/median imputation, and feature scaling without data leakage."),
        (ICON_PREPROCESS, "2. Classification, Regression & Unsupervised", "Fits Supervised models plus Clustering (K-Means, DBSCAN), PCA, and Anomaly Detection."),
        (ICON_COMPARE, "3. Explainable AI & Leaderboard", "Ranks performance scores, feature importance, confusion matrix, residual/cluster plots, and exports reports."),
    ]
    for col, (icon_svg, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f'<div class="ag-feat-card">'
                f'<div class="ag-feat-icon">{icon_svg}</div>'
                f'<div class="ag-feat-title">{title}</div>'
                f'<div class="ag-feat-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# Execute active training pipeline if triggered
elif start_training and df is not None:
    st.session_state["should_run_automl"] = False
    st.markdown('<div class="ag-card" style="border-top: 4px solid #4f46e5;">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ag-card-header">'
        '<div class="ag-card-title"><span style="color:#4f46e5;">⚡</span><span>AutoML Engine Training Pipeline Active</span></div>'
        '<span style="background:#f5f3ff; color:#4f46e5; border:1px solid #c7d2fe; font-size:0.75rem; font-weight:800; padding:0.25rem 0.6rem; border-radius:6px;">Processing...</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.status("⚡ Running MLNexus AutoML Engine...", expanded=True) as status:
        res = run_automl(
            df, target_col, task_type,
            test_size_pct / 100, run_cv, run_tuning,
            status_container=status
        )
        if res is not None:
            st.session_state["automl_results"] = res
            status.update(label="✅ AutoML Pipeline Complete!", state="complete", expanded=False)
            st.rerun()
        else:
            status.update(label="❌ AutoML Pipeline Failed", state="error", expanded=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Dataset Overview & Launch Panel
elif _results_pre is None:
    st.markdown(section("Dataset Inspection"), unsafe_allow_html=True)
    st.markdown(stat_row_html(df), unsafe_allow_html=True)

    tab_prev, tab_info, tab_corr = st.tabs([
        "👁️ Data Preview (First 10 rows)",
        "📋 Column Health & Types",
        "📈 Correlation Matrix",
    ])
    with tab_prev:
        st.dataframe(df.head(10), use_container_width=True)
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
        st.dataframe(ci, use_container_width=True, hide_index=True)
    with tab_corr:
        try:
            fig_c, _ = plot_correlation_heatmap(df)
            st.plotly_chart(fig_c, use_container_width=True)
        except Exception:
            st.info("Not enough numeric columns for a correlation heatmap.")

    # PROMINENT START AUTOML LAUNCH CARD
    st.markdown('<div class="ag-card" style="margin-top:1.75rem; border-top: 4px solid #4f46e5;">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ag-card-header">'
        '<div class="ag-card-title">'
        '<span style="color:#4f46e5;">⚡</span>'
        '<span>Configure Target Variable & Task Setup</span>'
        '</div>'
        f'<span style="background:#f5f3ff; color:#4f46e5; border:1px solid #c7d2fe; font-size:0.75rem; font-weight:800; padding:0.25rem 0.6rem; border-radius:6px;">✨ Auto-Detected: {task_type.capitalize()}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    c_tgt, c_task, c_btn = st.columns([1.5, 1.8, 1.3])
    with c_tgt:
        st.markdown('<p style="font-weight:800; color:#0f172a; font-size:0.9rem; margin-bottom:0.3rem;">Target Variable (To Predict)</p>', unsafe_allow_html=True)
        target_options = [NONE_UNSUPERVISED] + list(df.columns)
        cur_target = st.session_state.get("target_col", df.columns[-1])
        sel_idx = 0 if cur_target == NONE_UNSUPERVISED else (target_options.index(cur_target) if cur_target in target_options else 1)
        m_target = st.selectbox(
            "Target Column", target_options,
            index=sel_idx,
            label_visibility="collapsed",
            key="main_overview_target_select",
            on_change=update_target_from_main
        )

    with c_task:
        st.markdown('<p style="font-weight:800; color:#0f172a; font-size:0.9rem; margin-bottom:0.3rem;">Select ML Task Mode</p>', unsafe_allow_html=True)
        cur_overview_idx = TASK_OPTIONS.index(st.session_state.get("task_type", "classification").capitalize()) if st.session_state.get("task_type", "classification").capitalize() in TASK_OPTIONS else 0
        o_task = st.radio(
            "Overview Task Mode", TASK_OPTIONS,
            index=cur_overview_idx,
            horizontal=True, label_visibility="collapsed",
            key="main_overview_task_radio",
            on_change=update_task_from_overview
        ).lower()

    with c_btn:
        st.markdown('<p style="font-weight:800; color:transparent; font-size:0.9rem; margin-bottom:0.3rem;">Run</p>', unsafe_allow_html=True)
        start_training_main = st.button("⚡ Launch AutoML Pipeline", type="primary", key="main_overview_start_automl_btn", use_container_width=True, on_click=trigger_automl_callback)

    st.markdown('</div>', unsafe_allow_html=True)


# RESULTS PANEL
results = st.session_state.get("automl_results")

if results is not None and not start_training:
    rt      = results["task_type"]
    rname   = results["best_name"]
    rdf     = results["results_df"]
    p_label = ("F1-Score" if rt == "classification"
               else ("R²" if rt == "regression"
                     else "Silhouette Score"))
    algo_d  = (ALGORITHM_INFO if rt == "classification"
               else (REGRESSION_INFO if rt == "regression"
                     else UNSUPERVISED_INFO))

    # Winner Hero Summary Card
    st.markdown(
        f'<div class="ag-card" style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #ffffff; padding: 1.5rem 1.75rem; border: none;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">'
        f'<div>'
        f'<span style="background:rgba(255,255,255,0.2); color:#ffffff; font-size:0.75rem; font-weight:800; padding:0.25rem 0.75rem; border-radius:9999px; text-transform:uppercase; letter-spacing:0.06em;">🏆 WINNING ALGORITHM</span>'
        f'<h2 style="font-size:1.8rem; font-weight:800; color:#ffffff; margin:0.4rem 0 0.2rem 0; letter-spacing:-0.02em;">{rname}</h2>'
        f'<p style="margin:0; color:rgba(255,255,255,0.85); font-size:0.9rem; font-weight:500;">Outperformed {len(rdf)-1} other algorithms on dataset</p>'
        f'</div>'
        f'<div style="display:flex; gap:1rem; align-items:center;">'
        f'<div style="background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25); border-radius:12px; padding:0.75rem 1.25rem; text-align:center;">'
        f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:1.6rem; font-weight:800; color:#ffffff; display:block;">{results["best_primary"]:.4f}</span>'
        f'<span style="font-size:0.7rem; font-weight:700; color:rgba(255,255,255,0.8); text-transform:uppercase;">Top {p_label}</span>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25); border-radius:12px; padding:0.75rem 1.25rem; text-align:center;">'
        f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:1.6rem; font-weight:800; color:#ffffff; display:block;">{results["best_time"]}s</span>'
        f'<span style="font-size:0.7rem; font-weight:700; color:rgba(255,255,255,0.8); text-transform:uppercase;">Train Time</span>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 1. Leaderboard
    st.markdown(section("Model Leaderboard"), unsafe_allow_html=True)
    st.markdown(leaderboard_html(rdf, rt, rname), unsafe_allow_html=True)

    left_dl, _, right_cv = st.columns([1, 2, 1])
    with left_dl:
        st.download_button(
            "📥 Export Leaderboard CSV",
            rdf.assign(Best=["Yes" if m == rname else "" for m in rdf["Model"]])
               .to_csv(index=False).encode("utf-8"),
            file_name=f"automl_leaderboard_{results['target']}.csv",
            mime="text/csv",
        )
    with right_cv:
        if results["run_cv"]:
            st.caption("CV column = 3-fold mean ± std on training rows")

    # 2. Explanation Card
    st.markdown(section("Explainable AI Insights"), unsafe_allow_html=True)
    st.markdown(explain_card_html(rname, p_label, results, algo_d), unsafe_allow_html=True)

    why_col, tip_col = st.columns(2)
    with why_col:
        if rt == "classification":
            st.info("**Why F1-Score?** Harmonic mean of Precision & Recall — robust to class imbalance, unlike raw accuracy.")
        elif rt == "regression":
            st.info("**Why R²?** Fraction of variance explained. MAE/RMSE give error magnitude in target units.")
        else:
            st.info("**Why Silhouette Score?** Measures how well samples are clustered with their own cluster versus neighbor clusters (-1 to +1).")
    with tip_col:
        if results.get("tuning_info"):
            ti = results["tuning_info"]
            st.success(
                f"Tuning improved **{rname}**  \n"
                f"Best params: `{ti['best_params']}`  \n"
                f"CV {ti['scoring']}: **{ti['cv_score']}**"
            )
        elif rt == "unsupervised":
            st.success(f"Evaluated 10 unsupervised algorithms (Clustering, PCA, Anomaly Detection). Top model: **{rname}**.")

    # 3. Performance comparison
    st.markdown(section("Performance Comparison"), unsafe_allow_html=True)
    st.markdown(chart_header(f"All Models · {rt.capitalize()} Evaluation",
                             "Higher is better for all primary metrics"), unsafe_allow_html=True)
    if rt == "classification":
        st.plotly_chart(plot_model_comparison_clf(rdf), use_container_width=True)
    elif rt == "regression":
        st.plotly_chart(plot_model_comparison_reg(rdf), use_container_width=True)
    else:
        st.plotly_chart(plot_model_comparison_unsupervised(rdf), use_container_width=True)

    if results["run_cv"]:
        cv_fig = plot_cv_comparison(rdf, rt)
        if cv_fig:
            st.markdown(chart_header("Test Score vs CV Score",
                                     "Large gap may indicate overfitting"),
                        unsafe_allow_html=True)
            st.plotly_chart(cv_fig, use_container_width=True)

    # 4. Detailed analysis grid
    st.markdown(section("Detailed Model Diagnostics"), unsafe_allow_html=True)

    if rt == "classification":
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(chart_header("Confusion Matrix Heatmap",
                                     f"Winning Model: {rname}"), unsafe_allow_html=True)
            st.plotly_chart(
                plot_confusion_matrix(
                    results["cm"], results["col_info"]["class_names"], rname),
                use_container_width=True,
            )
        with col_b:
            if results["importance"] is not None:
                st.markdown(chart_header("Top 5 Predictive Features"), unsafe_allow_html=True)
                top5 = results["importance"].head(5).copy()
                top5.insert(0, "#", range(1, len(top5) + 1))
                top5["Importance"] = top5["Importance"].round(4)
                st.dataframe(top5, hide_index=True, use_container_width=True)
                st.caption(results["importance_method"])
    elif rt == "regression":
        y_arr = (results["y_test"].values
                 if hasattr(results["y_test"], "values")
                 else np.asarray(results["y_test"]))
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(chart_header("Actual vs Predicted",
                                     "Points on the diagonal line = perfect prediction"),
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_actual_vs_predicted(y_arr, results["best_predictions"], rname),
                use_container_width=True,
            )
        with col_b:
            st.markdown(chart_header("Residual Distribution",
                                     "Ideal: random scatter centered on zero line"),
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_residuals(y_arr, results["best_predictions"], rname),
                use_container_width=True,
            )
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(chart_header("2D Cluster & PCA Projection", f"Winning Algorithm: {rname}"), unsafe_allow_html=True)
            st.plotly_chart(
                plot_cluster_pca_scatter(results["X_scaled"], results["best_predictions"], rname),
                use_container_width=True,
            )
        with col_b:
            st.markdown(chart_header("Unsupervised Group Sample Counts"), unsafe_allow_html=True)
            if results["best_predictions"] is not None:
                group_counts = pd.Series(results["best_predictions"]).value_counts().reset_index()
                group_counts.columns = ["Cluster / Group ID", "Sample Count"]
                st.dataframe(group_counts, use_container_width=True, hide_index=True)
            else:
                st.info("Dimensionality reduction model projects feature space into component axes.")

    # Feature importance full chart
    if results["importance"] is not None:
        st.markdown(chart_header("Feature Importance Ranking",
                                 results["importance_method"]), unsafe_allow_html=True)
        st.plotly_chart(
            plot_feature_importance(results["importance"],
                                    results["importance_method"]),
            use_container_width=True,
        )

    # Training time
    st.markdown(chart_header("Training Computational Efficiency",
                             "Log scale — lower time indicates faster execution"), unsafe_allow_html=True)
    st.plotly_chart(plot_training_time(rdf), use_container_width=True)

    # 5. Algorithm Reference Guide
    st.markdown(section("Algorithm Reference & Documentation"), unsafe_allow_html=True)
    with st.expander(f"📚 How each of the {len(rdf)} trained algorithms works (Technical Guide for Presentation)"):
        for name, desc in algo_d.items():
            trophy = " 🏆 (Winner)" if name == rname else ""
            st.markdown(f"**{name}{trophy}** — {desc}")

    if results["failed"]:
        st.warning(
            "Some models were skipped:\n\n"
            + "\n\n".join(f"• {f}" for f in results["failed"])
        )

    if df is not None:
        with st.expander("🔍 View Training Dataset Snapshot"):
            st.dataframe(df.head(10), use_container_width=True)
