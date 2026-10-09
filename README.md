# ⚡ MLNexus — Automated Machine Learning Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mlnexus.streamlit.app/)
**Live Web Application**: [https://mlnexus.streamlit.app/](https://mlnexus.streamlit.app/)

MLNexus is an automated machine learning (AutoML) web application built with Streamlit, scikit-learn, and Plotly. It allows you to upload any tabular dataset (CSV/TSV), choose target variables or task modes, and automatically train, rank, and evaluate Classification, Regression, and Unsupervised Learning algorithms in 1-click.

## Features

- **Automated Pipeline**: Handles data validation, leakage-free preprocessing, feature scaling, and categorical encoding with smart task mode auto-detection.
- **Classification, Regression & Unsupervised**: Supports Classification (16 models), Regression (13 models), and Unsupervised Learning (10 models including Clustering, PCA, and Anomaly Detection).
- **Leaderboard**: Compares models based on key performance metrics (F1-score, Accuracy, Precision, Recall for classification; R², MAE, RMSE for regression; Silhouette score, Calinski-Harabasz, and Davies-Bouldin for unsupervised).
- **Interactive Visualizations**: Includes model comparison bar charts, confusion matrices, actual vs. predicted plots, residual plots, 2D PCA cluster maps, feature importance, and correlation heatmaps.
- **Model Explainability**: Highlights top features driving model predictions and provides technical algorithm performance guides.

## Tech Stack

- **Frontend / UI**: Streamlit
- **Machine Learning**: scikit-learn
- **Data Handling**: Pandas, NumPy
- **Plotting & Charts**: Plotly

## Project Structure

```text
MLNexus/
├── app.py              # Main Streamlit web application
├── preprocessing.py    # Data loading, validation, and preprocessing pipeline
├── models.py           # Model definitions, training, and feature importance
├── charts.py           # Plotly interactive chart builders
├── requirements.txt    # Python dependencies
└── sample_data.csv     # Sample dataset for testing
```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Lucky136-code/MLNexus.git
cd MLNexus
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.
