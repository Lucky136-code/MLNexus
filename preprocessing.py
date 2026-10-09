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


def auto_detect_task_type(df, target_col=None):
    """Auto-detect whether dataset task should be classification, regression, or unsupervised."""
    if df is None or df.empty:
        return "classification"

    NONE_UNSUPERVISED = "None (Unsupervised Learning)"
    if target_col == NONE_UNSUPERVISED:
        return "unsupervised"

    # Check for known target candidate names in columns
    common_targets = ("target", "class", "label", "y", "species", "diagnosis",
                      "outcome", "survived", "purchased", "price", "salary",
                      "value", "score", "result")

    found_common = next((c for c in df.columns if c.strip().lower() in common_targets), None)

    # If no target specified or target column invalid
    if target_col is None or target_col not in df.columns:
        if found_common:
            target_col = found_common
        else:
            # Check if all columns are numeric features with no clear label column
            typed_df = coerce_object_columns_to_numeric(df.copy())
            num_cols = typed_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == df.shape[1]:
                return "unsupervised"
            target_col = df.columns[-1]

    col_data = df[target_col].dropna()
    if len(col_data) == 0:
        return "unsupervised"

    if col_data.dtype == object or col_data.dtype == bool or isinstance(col_data.dtype, pd.CategoricalDtype):
        return "classification"

    numeric_col = pd.to_numeric(col_data, errors="coerce")
    if numeric_col.isna().all():
        return "classification"

    n_unique = numeric_col.nunique()
    if n_unique <= 10:
        return "classification"
    else:
        return "regression"


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
        valid_mask = numeric_y.notna()
        if not valid_mask.any():
            raise ValueError(f"Target '{target_col}' does not contain valid numeric values for regression. Switch to Classification or select a numeric column.")
        data = data[valid_mask]
        y = pd.Series(numeric_y[valid_mask].values, index=data.index)
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


def prepare_unsupervised_features(df):
    """Prepare dataset for unsupervised learning (clustering, PCA, anomaly detection)."""
    data = df.replace([np.inf, -np.inf], np.nan).copy()
    if data.shape[0] < 5:
        raise ValueError("Dataset must have at least 5 rows for unsupervised learning.")

    data = coerce_object_columns_to_numeric(data)
    data = data.dropna(axis=1, how="all")

    if data.shape[1] == 0:
        raise ValueError("No valid feature columns remaining in dataset.")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in data.columns if c not in numeric_cols]
    missing_before = int(data.isna().sum().sum())

    X_processed = data.copy()

    # Numeric imputation
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_processed[numeric_cols] = num_imputer.fit_transform(X_processed[numeric_cols])

    # Categorical imputation & encoding
    if categorical_cols:
        X_processed[categorical_cols] = X_processed[categorical_cols].astype(str)
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_processed[categorical_cols] = cat_imputer.fit_transform(X_processed[categorical_cols])
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_processed),
        columns=X_processed.columns,
        index=X_processed.index,
    )

    col_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_before": missing_before,
        "dropped_target_rows": 0,
        "n_classes": None,
        "class_names": None,
    }

    return X_scaled, col_info


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
