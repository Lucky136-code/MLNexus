"""
Plotly visualization functions for ML models.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models import CV_FOLDS

MAX_CORRELATION_COLS = 15
MAX_IMPORTANCE_FEATURES = 15


# Classification charts
def plot_model_comparison_clf(results_df):
    """Bar chart comparing classification metrics across models."""
    melted = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric", value_name="Score",
    )
    fig = px.bar(
        melted, x="Model", y="Score", color="Metric",
        barmode="group", template="plotly_white",
        category_orders={"Model": results_df["Model"].tolist()},
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={"Score": ":.4f"},
    )
    fig.update_layout(
        title="Classification — Model Performance Comparison",
        yaxis_range=[0, 1.05], yaxis_title="Score",
        xaxis_title="Model", legend_title="Metric",
        height=460, hovermode="x unified",
    )
    return fig


def plot_confusion_matrix(cm, class_names, best_name):
    """Confusion matrix heatmap for the best classifier."""
    fig = px.imshow(
        cm, x=class_names, y=class_names,
        labels=dict(x="Predicted label", y="Actual label", color="Count"),
        color_continuous_scale="Blues", text_auto=True,
        template="plotly_white", aspect="auto",
    )
    fig.update_layout(
        title=f"Confusion Matrix — {best_name}", height=460,
        xaxis_nticks=len(class_names), yaxis_nticks=len(class_names),
    )
    return fig


# Regression charts
def plot_model_comparison_reg(results_df):
    """Bar chart comparing regression metrics across models."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    models = results_df["Model"].tolist()

    for idx, (col, name) in enumerate([("R²", "R² (left axis)"),
                                        ("MAE", "MAE (right axis)"),
                                        ("RMSE", "RMSE (right axis)")]):
        yaxis = "y" if col == "R²" else "y2"
        fig.add_trace(go.Bar(
            name=name,
            x=models,
            y=results_df[col],
            yaxis=yaxis,
            marker_color=colors[idx % len(colors)],
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title="Regression — Model Performance Comparison",
        yaxis=dict(title="R²", range=[None, 1.05]),
        yaxis2=dict(title="MAE / RMSE", overlaying="y", side="right"),
        legend_title="Metric",
        height=460,
        hovermode="x unified",
        xaxis_title="Model",
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
        line=dict(color="red", dash="dash", width=1.5),
        name="Perfect prediction",
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=f"Actual vs Predicted — {best_name}",
        height=440,
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
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1.5)
    fig.update_layout(
        title=f"Residual Plot — {best_name}",
        height=420,
    )
    return fig


# Shared charts
def plot_feature_importance(importance, method):
    """Bar chart of top feature importances."""
    top = importance.head(MAX_IMPORTANCE_FEATURES).iloc[::-1]
    fig = px.bar(
        top, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Viridis",
        template="plotly_white",
        hover_data={"Importance": ":.4f"},
    )
    fig.update_layout(
        title=f"Top {len(top)} Feature Importance",
        xaxis_title="Importance score", yaxis_title=None,
        height=480, coloraxis_showscale=False,
    )
    return fig


def plot_training_time(results_df):
    """Bar chart of training time per model."""
    plot_df = results_df.copy()
    plot_df["Training Time (s)"] = plot_df["Training Time (s)"].clip(lower=0.001)
    fig = px.bar(
        plot_df.sort_values("Training Time (s)"),
        x="Model", y="Training Time (s)", color="Family",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={"Training Time (s)": ":.3f"},
    )
    fig.update_layout(
        title="Training Time per Model (seconds)",
        yaxis_type="log", height=420, xaxis_title=None,
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
    fig.update_layout(title="Correlation Heatmap", height=560)
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
        {test_col: "Test score", "CV mean": f"CV mean ({CV_FOLDS}-fold)"}
    )
    fig = px.bar(
        melted, x="Model", y="Score", color="Type",
        barmode="group", template="plotly_white",
        color_discrete_sequence=["#4f46e5", "#10b981"],
        hover_data={"Score": ":.4f"},
        category_orders={"Model": df["Model"].tolist()},
    )
    fig.update_layout(
        title=f"Test Score vs {CV_FOLDS}-Fold CV Score — {label}",
        yaxis_range=[0, 1.05] if task_type == "classification" else None,
        height=440, hovermode="x unified",
    )
    return fig


# Unsupervised charts
def plot_model_comparison_unsupervised(results_df):
    """Bar chart comparing unsupervised metrics across algorithms."""
    df = results_df.copy()
    fig = px.bar(
        df, x="Model", y="Silhouette Score", color="Family",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={"Silhouette Score": ":.4f", "Clusters / Outliers": True},
    )
    fig.update_layout(
        title="Unsupervised — Silhouette & Quality Score Comparison",
        yaxis_title="Silhouette Score / Variance Explained",
        xaxis_title="Algorithm", legend_title="Category",
        height=460, hovermode="x unified",
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
        title=f"2D Projection & Clustering Map — {best_name}",
        height=460,
    )
    return fig
