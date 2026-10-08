"""
Data preprocessing utilities.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(uploaded_file):
    """Load CSV or TSV file into pandas DataFrame."""
    if uploaded_file.size == 0:
        raise ValueError("Uploaded file is empty.")

    sep = "\t" if uploaded_file.name.lower().endswith(".tsv") else ","
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}")

    # Fallback separator check
    if df.shape[1] == 1:
        uploaded_file.seek(0)
        other_sep = "," if sep == "\t" else "\t"
        try:
            df_alt = pd.read_csv(uploaded_file, sep=other_sep)
            if df_alt.shape[1] > df.shape[1]:
                df = df_alt
        except Exception:
            pass

    if df.empty:
        raise ValueError("File contains no data rows.")

    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_dataset(df):
    """Validate dataset row and column counts."""
    problems = []
    if df.shape[0] < 10:
        problems.append(f"Dataset has only {df.shape[0]} row(s). At least 10 rows required.")
    if df.shape[1] < 2:
        problems.append("Dataset needs at least 2 columns (1 feature, 1 target).")
    return problems


def coerce_object_columns_to_numeric(X):
    """Cast numeric string columns to float64."""
    for col in X.columns:
        if X[col].dtype == object:
            converted = pd.to_numeric(X[col], errors="coerce")
            non_null = X[col].notna().sum()
            if non_null > 0 and converted.notna().sum() == non_null:
                X[col] = converted
    return X


def prepare_features_and_target(df, target_col, task_type="classification"):
    """Separate features and target, encode target column if categorical."""
    data = df.replace([np.inf, -np.inf], np.nan)
    rows_before = len(data)
    data = data.dropna(subset=[target_col])
    dropped_target_rows = rows_before - len(data)

    if data.shape[0] < 10:
        raise ValueError("Fewer than 10 rows remain after dropping missing target values.")

    # Target processing
    if task_type == "classification":
        y_raw = data[target_col].astype(str)
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y_raw), index=data.index)
        if len(target_encoder.classes_) < 2:
            raise ValueError(f"Target '{target_col}' must have at least 2 distinct classes.")
        n_classes = len(target_encoder.classes_)
        class_names = [str(c) for c in target_encoder.classes_]
    else:
        target_encoder = None
        numeric_y = pd.to_numeric(data[target_col], errors="coerce")
        if numeric_y.isna().all():
            raise ValueError(f"Target '{target_col}' must contain numeric values for regression.")
        y = pd.Series(numeric_y.values, index=data.index)
        n_classes, class_names = None, None

    # Feature processing
    X = data.drop(columns=[target_col])
    X = coerce_object_columns_to_numeric(X)
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("No feature columns remaining in dataset.")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    missing_before = int(X.isna().sum().sum())

    col_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_before": missing_before,
        "dropped_target_rows": dropped_target_rows,
        "n_classes": n_classes,
        "class_names": class_names,
    }

    return X, y, target_encoder, col_info


def fit_transform_train(X_train, col_info):
    """Fit imputer and scaler on training set, return transformed data and pipeline."""
    X_train = X_train.copy()
    numeric_cols = col_info["numeric_cols"]
    categorical_cols = col_info["categorical_cols"]
    pipeline = {}

    # Numeric imputation
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        pipeline["num_imputer"] = num_imputer

    # Categorical imputation & encoding
    if categorical_cols:
        X_train[categorical_cols] = X_train[categorical_cols].astype(str)
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        cat_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            cat_encoders[col] = le
        pipeline["cat_imputer"] = cat_imputer
        pipeline["cat_encoders"] = cat_encoders

    # Feature scaling
    if numeric_cols:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        pipeline["scaler"] = scaler

    return X_train, pipeline


def transform_test(X_test, col_info, pipeline):
    """Transform test set using fitted pipeline."""
    X_test = X_test.copy()
    numeric_cols = col_info["numeric_cols"]
    categorical_cols = col_info["categorical_cols"]

    if numeric_cols and "num_imputer" in pipeline:
        X_test[numeric_cols] = pipeline["num_imputer"].transform(X_test[numeric_cols])

    if categorical_cols and "cat_imputer" in pipeline:
        X_test[categorical_cols] = X_test[categorical_cols].astype(str)
        X_test[categorical_cols] = pipeline["cat_imputer"].transform(X_test[categorical_cols])
        cat_encoders = pipeline.get("cat_encoders", {})
        for col in categorical_cols:
            le = cat_encoders[col]
            known = set(le.classes_)
            fallback = le.classes_[0]
            X_test[col] = X_test[col].apply(
                lambda v, k=known, fb=fallback: v if v in k else fb
            )
            X_test[col] = le.transform(X_test[col])

    if numeric_cols and "scaler" in pipeline:
        X_test[numeric_cols] = pipeline["scaler"].transform(X_test[numeric_cols])

    return X_test
