"""
Plotly visualization functions for MLNexus platform (Light Theme Dashboard).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models import CV_FOLDS

MAX_CORRELATION_COLS = 15
MAX_IMPORTANCE_FEATURES = 15


def plot_model_comparison_clf(results_df):
    """Horizontal bar chart comparing classification metrics across models."""
    df = results_df.copy()
    df = df.sort_values("F1-Score", ascending=True)

    melted = df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric", value_name="Score",
    )
    fig = px.bar(
        melted, y="Model", x="Score", color="Metric",
        barmode="group", orientation="h",
        template="plotly_white",
        color_discrete_sequence=["#4f46e5", "#10b981", "#f59e0b", "#ec4899"],
        hover_data={"Score": ":.4f"},
    )
    fig.update_layout(
        title="Classification — Model Performance Comparison",
        xaxis_range=[0, 1.05], xaxis_title="Score",
        yaxis_title=None, legend_title="Metric",
        height=max(460, len(results_df) * 32),
        hovermode="y unified",
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def plot_confusion_matrix(cm, class_names, best_name):
    """Confusion matrix heatmap with explicit label ordering and annotations."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    else:
        class_names = [str(c) for c in class_names]

    x_labels = [f"Pred: {c}" for c in class_names]
    y_labels = [f"Actual: {c}" for c in class_names]

    total = np.sum(cm)
    text_matrix = []
    for i in range(cm.shape[0]):
        row_text = []
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total * 100) if total > 0 else 0
            row_text.append(f"<b>{val}</b><br>({pct:.1f}%)")
        text_matrix.append(row_text)

    fig = px.imshow(
        cm, x=x_labels, y=y_labels,
        labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
        color_continuous_scale="Blues", text_auto=False,
        template="plotly_white", aspect="auto",
    )
    fig.update_traces(text=text_matrix, texttemplate="%{text}")
    fig.update_layout(
        title=f"Confusion Matrix — {best_name}", height=460,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def plot_model_comparison_reg(results_df):
    """Bar chart comparing regression metrics across models."""
    df = results_df.copy().sort_values("R²", ascending=True)
    models = df["Model"].tolist()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="R² (Left axis, higher=better)",
        y=models,
        x=df["R²"],
        orientation="h",
        marker_color="#4f46e5",
        hovertemplate="<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_white",
        title="Regression — R² Metric Comparison",
        xaxis=dict(title="R² Score", range=[None, 1.05]),
        legend_title="Metric",
        height=max(440, len(results_df) * 30),
        yaxis_title=None,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def plot_actual_vs_predicted(y_test, best_preds, best_name):
    """Scatter plot of actual vs predicted values."""
    df = pd.DataFrame({"Actual": y_test, "Predicted": best_preds})
    fig = px.scatter(
        df, x="Actual", y="Predicted",
        template="plotly_white",
        opacity=0.6,
        color_discrete_sequence=["#4f46e5"],
        hover_data={"Actual": ":.4f", "Predicted": ":.4f"},
    )
    lo = float(min(df["Actual"].min(), df["Predicted"].min()))
    hi = float(max(df["Actual"].max(), df["Predicted"].max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color="#ef4444", dash="dash", width=2),
        name="Perfect Prediction Line",
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=f"Actual vs Predicted — {best_name}",
        height=440,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def plot_residuals(y_test, best_preds, best_name):
    """Residual plot (predicted vs residual)."""
    residuals = np.asarray(y_test) - np.asarray(best_preds)
    df = pd.DataFrame({"Predicted": best_preds, "Residual": residuals})
    fig = px.scatter(
        df, x="Predicted", y="Residual",
        template="plotly_white",
        opacity=0.6,
        color_discrete_sequence=["#10b981"],
        hover_data={"Predicted": ":.4f", "Residual": ":.4f"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=2)
    fig.update_layout(
        title=f"Residual Diagnostic Plot — {best_name}",
        height=420,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def plot_feature_importance(importance, method):
    """Horizontal bar chart of feature importances."""
    top = importance.head(MAX_IMPORTANCE_FEATURES).iloc[::-1]
    fig = px.bar(
        top, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Purples",
        template="plotly_white",
        hover_data={"Importance": ":.4f"},
    )
    fig.update_layout(
        title=f"Top {len(top)} Feature Importance Rankings",
        xaxis_title="Importance Score", yaxis_title=None,
        height=460, coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def plot_training_time(results_df):
    """Bar chart of training time per model with log scale support."""
    plot_df = results_df.copy()
    plot_df["Training Time (s)"] = plot_df["Training Time (s)"].clip(lower=0.001)
    plot_df = plot_df.sort_values("Training Time (s)", ascending=True)

    fig = px.bar(
        plot_df,
        y="Model", x="Training Time (s)", color="Family",
        orientation="h",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={"Training Time (s)": ":.3f"},
    )
    fig.update_layout(
        title="Training Execution Time per Model (Seconds, Log Scale)",
        xaxis_type="log", height=max(420, len(results_df) * 28),
        yaxis_title=None, xaxis_title="Training Time (seconds)",
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def plot_correlation_heatmap(df):
    """Correlation heatmap of numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > MAX_CORRELATION_COLS:
        numeric_df = numeric_df.iloc[:, :MAX_CORRELATION_COLS]
    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, template="plotly_white", aspect="auto",
        labels=dict(color="Correlation"),
    )
    fig.update_layout(title="Feature Correlation Heatmap", height=540, margin=dict(l=10, r=10, t=50, b=40))
    return fig, numeric_df.shape[1]


def plot_cv_comparison(results_df, task_type):
    """Bar chart comparing test score vs CV score."""
    cv_col = "CV F1 (mean±std)" if task_type == "classification" else "CV R² (mean±std)"
    test_col = "F1-Score" if task_type == "classification" else "R²"
    label = "F1-Score" if task_type == "classification" else "R²"

    def _parse_mean(s):
        if s in ("—", "error", None):
            return None
        try:
            return float(str(s).split("±")[0].strip())
        except Exception:
            return None

    df = results_df[["Model", test_col, cv_col]].copy()
    df["CV mean"] = df[cv_col].apply(_parse_mean)
    df = df.dropna(subset=["CV mean"])
    if df.empty:
        return None

    melted = df.melt(
        id_vars="Model",
        value_vars=[test_col, "CV mean"],
        var_name="Type", value_name="Score",
    )
    melted["Type"] = melted["Type"].replace(
        {test_col: "Test Score", "CV mean": f"CV Mean ({CV_FOLDS}-fold)"}
    )
    fig = px.bar(
        melted, y="Model", x="Score", color="Type",
        barmode="group", orientation="h", template="plotly_white",
        color_discrete_sequence=["#4f46e5", "#10b981"],
        hover_data={"Score": ":.4f"},
        category_orders={"Model": df["Model"].tolist()},
    )
    fig.update_layout(
        title=f"Held-out Test Score vs {CV_FOLDS}-Fold CV Score — {label}",
        xaxis_range=[0, 1.05] if task_type == "classification" else None,
        height=max(440, len(df) * 32),
        margin=dict(l=10, r=10, t=50, b=40),
        yaxis_title=None,
    )
    return fig


def plot_model_comparison_unsupervised(results_df):
    """Horizontal bar chart comparing unsupervised metrics across algorithms."""
    df = results_df.copy().sort_values("Silhouette Score", ascending=True)
    fig = px.bar(
        df, y="Model", x="Silhouette Score", color="Family",
        orientation="h",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={"Silhouette Score": ":.4f", "Clusters / Outliers": True},
    )
    fig.update_layout(
        title="Unsupervised — Silhouette & Variance Explained Quality Comparison",
        xaxis_title="Silhouette Score / Variance Ratio",
        yaxis_title=None, legend_title="Category",
        height=max(460, len(results_df) * 32),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def plot_cluster_pca_scatter(X_scaled, labels, best_name):
    """2D PCA Projection showing clusters or anomaly predictions."""
    from sklearn.decomposition import PCA
    n_comp = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X_scaled)

    if labels is None:
        labels_str = ["Point"] * len(X_scaled)
    else:
        labels_str = [f"Cluster {l}" if l != -1 else "Noise/Outlier" for l in labels]

    df_pca = pd.DataFrame({
        "PCA Component 1": coords[:, 0],
        "PCA Component 2": coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords)),
        "Group": labels_str
    })

    fig = px.scatter(
        df_pca, x="PCA Component 1", y="PCA Component 2", color="Group",
        template="plotly_white", opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(
        title=f"2D PCA Projection & Cluster Allocation — {best_name}",
        height=460,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig
