"""
Data preprocessing, validation, and leakage-safe feature engineering pipeline.
"""

import io
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(uploaded_file_or_path):
    """
    Robust dataset reader supporting CSV and TSV files with various encodings and delimiters.
    Handles UTF-8, UTF-8 BOM (utf-8-sig), latin1, and cp1252.
    """
    if uploaded_file_or_path is None:
        raise ValueError("No file provided.")

    file_bytes = None
    file_name = ""

    # Streamlit UploadedFile or file-like object vs path string
    if hasattr(uploaded_file_or_path, "getvalue"):
        file_bytes = uploaded_file_or_path.getvalue()
        file_name = getattr(uploaded_file_or_path, "name", "").lower()
    elif isinstance(uploaded_file_or_path, str):
        with open(uploaded_file_or_path, "rb") as f:
            file_bytes = f.read()
        file_name = uploaded_file_or_path.lower()
    elif hasattr(uploaded_file_or_path, "read"):
        uploaded_file_or_path.seek(0)
        file_bytes = uploaded_file_or_path.read()
        file_name = getattr(uploaded_file_or_path, "name", "").lower()

    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    delimiters = ["\t" if file_name.endswith(".tsv") else ",", ",", "\t", ";", "|"]

    df = None
    last_error = None

    for enc in encodings:
        for sep in delimiters:
            try:
                buf = io.BytesIO(file_bytes)
                candidate_df = pd.read_csv(buf, sep=sep, encoding=enc)
                if candidate_df.shape[1] > 1 or (candidate_df.shape[1] == 1 and len(delimiters) == 1):
                    df = candidate_df
                    break
            except Exception as exc:
                last_error = exc
                continue
        if df is not None:
            break

    if df is None:
        # Fallback to python engine with sep=None auto-detection
        for enc in encodings:
            try:
                buf = io.BytesIO(file_bytes)
                df = pd.read_csv(buf, sep=None, engine="python", encoding=enc)
                if df is not None:
                    break
            except Exception as exc:
                last_error = exc

    if df is None or df.empty:
        raise ValueError(f"Could not parse file: {last_error or 'No valid data rows found'}")

    # Clean whitespace in column names
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]

    if df.shape[0] == 0:
        raise ValueError("File contains header but zero data rows.")

    if df.shape[1] == 0:
        raise ValueError("No columns to parse from file.")

    return df


def validate_dataset(df):
    """Validate dataset row and column counts."""
    problems = []
    if df is None or df.empty:
        return ["Dataset is empty."]
    if df.shape[0] < 10:
        problems.append(f"Dataset has only {df.shape[0]} row(s). At least 10 rows are required for AutoML.")
    if df.shape[1] < 2:
        problems.append("Dataset needs at least 2 columns (1 feature, 1 target/grouping).")
    return problems


def detect_id_and_date_columns(df):
    """
    Detect likely identifier columns (e.g., ID, uuid, index) and date/datetime columns.
    """
    id_cols = []
    date_cols = []

    for col in df.columns:
        col_lower = str(col).lower().strip()
        series = df[col].dropna()
        n_unique = series.nunique()
        n_total = len(series)

        # ID detection heuristiic
        if col_lower in ("id", "identifier", "uuid", "guid", "index", "row_id", "customer_id", "user_id"):
            id_cols.append(col)
        elif n_total > 20 and n_unique == n_total and (series.dtype == object or pd.api.types.is_integer_dtype(series)):
            if any(term in col_lower for term in ["id", "code", "number", "no", "num", "hash"]):
                id_cols.append(col)

        # Date detection heuristic
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
        elif series.dtype == object and any(term in col_lower for term in ["date", "time", "timestamp", "dt", "year"]):
            try:
                pd.to_datetime(series.iloc[:50], errors="raise")
                date_cols.append(col)
            except Exception:
                pass

    return id_cols, date_cols


def analyze_target_and_features(df, target_col, task_type="classification"):
    """
    Analyze target column distribution, class imbalance, and potential data leakage.
    """
    diagnostics = {
        "warnings": [],
        "class_distribution": None,
        "imbalance_ratio": None,
        "leakage_cols": [],
        "id_cols": [],
        "date_cols": [],
    }

    if df is None or target_col not in df.columns:
        return diagnostics

    id_cols, date_cols = detect_id_and_date_columns(df)
    diagnostics["id_cols"] = id_cols
    diagnostics["date_cols"] = date_cols

    target_series = df[target_col].dropna()

    if task_type == "classification":
        counts = target_series.value_counts()
        diagnostics["class_distribution"] = counts.to_dict()
        if len(counts) >= 2:
            maj = counts.iloc[0]
            min_c = counts.iloc[-1]
            ratio = maj / max(1, min_c)
            diagnostics["imbalance_ratio"] = round(ratio, 2)
            min_pct = (min_c / len(target_series)) * 100
            if ratio > 5.0 or min_pct < 10.0:
                diagnostics["warnings"].append(
                    f"Class imbalance detected: Minority class '{counts.index[-1]}' accounts for only {min_pct:.1f}% of data (Imbalance Ratio: {ratio:.1f}:1)."
                )
        elif len(counts) < 2:
            diagnostics["warnings"].append(f"Target '{target_col}' has fewer than 2 distinct classes.")

    elif task_type == "regression":
        num_target = pd.to_numeric(target_series, errors="coerce").dropna()
        if len(num_target) < len(target_series):
            diagnostics["warnings"].append(f"Target '{target_col}' contains {len(target_series) - len(num_target)} non-numeric values.")
        if num_target.nunique() <= 1:
            diagnostics["warnings"].append(f"Target '{target_col}' is constant (only 1 unique value). Regression cannot fit a constant target.")

    # Leakage detection check
    for col in df.columns:
        if col == target_col:
            continue
        try:
            if df[col].equals(df[target_col]):
                diagnostics["leakage_cols"].append(col)
                diagnostics["warnings"].append(f"Potential data leakage: Column '{col}' is identical to target column '{target_col}'.")
            elif pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                corr = abs(df[col].corr(df[target_col]))
                if corr > 0.98:
                    diagnostics["leakage_cols"].append(col)
                    diagnostics["warnings"].append(f"High correlation leakage risk: Column '{col}' has {corr:.3f} correlation with target '{target_col}'.")
        except Exception:
            pass

    return diagnostics


def coerce_object_columns_to_numeric(X):
    """Cast numeric string columns to float64 safely."""
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
    if target_col == NONE_UNSUPERVISED or target_col is None:
        return "unsupervised"

    if target_col not in df.columns:
        return "classification"

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


def extract_date_features(df, date_cols):
    """Convert date columns into explicit numeric features (year, month, day, dayofweek)."""
    df_out = df.copy()
    for col in date_cols:
        if col in df_out.columns:
            try:
                dt_series = pd.to_datetime(df_out[col], errors="coerce")
                if dt_series.notna().sum() > 0:
                    df_out[f"{col}_year"] = dt_series.dt.year
                    df_out[f"{col}_month"] = dt_series.dt.month
                    df_out[f"{col}_day"] = dt_series.dt.day
                    df_out[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                    df_out = df_out.drop(columns=[col])
            except Exception:
                pass
    return df_out


def prepare_features_and_target(df, target_col, task_type="classification", exclude_cols=None):
    """
    Separate features and target, handle date features and target encoding.
    """
    if exclude_cols is None:
        exclude_cols = []

    data = df.replace([np.inf, -np.inf], np.nan).copy()
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
            raise ValueError(f"Target '{target_col}' does not contain valid numeric values for regression.")
        data = data[valid_mask]
        y = pd.Series(numeric_y[valid_mask].values, index=data.index)
        n_classes, class_names = None, None

    # Feature processing
    cols_to_drop = [target_col] + [c for c in exclude_cols if c in data.columns and c != target_col]
    X = data.drop(columns=cols_to_drop)

    # Process date features
    id_cols, date_cols = detect_id_and_date_columns(X)
    if date_cols:
        X = extract_date_features(X, date_cols)

    X = coerce_object_columns_to_numeric(X)
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("No feature columns remaining in dataset after target & exclusion filters.")

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
        "excluded_cols": exclude_cols,
    }

    return X, y, target_encoder, col_info


def prepare_unsupervised_features(df, exclude_cols=None):
    """Prepare dataset for unsupervised learning (clustering, PCA, anomaly detection)."""
    if exclude_cols is None:
        exclude_cols = []

    data = df.replace([np.inf, -np.inf], np.nan).copy()
    if data.shape[0] < 5:
        raise ValueError("Dataset must have at least 5 rows for unsupervised learning.")

    cols_to_drop = [c for c in exclude_cols if c in data.columns]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    id_cols, date_cols = detect_id_and_date_columns(data)
    if date_cols:
        data = extract_date_features(data, date_cols)

    data = coerce_object_columns_to_numeric(data)
    data = data.dropna(axis=1, how="all")

    if data.shape[1] == 0:
        raise ValueError("No valid feature columns remaining in dataset.")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in data.columns if c not in numeric_cols]
    missing_before = int(data.isna().sum().sum())

    X_processed = data.copy()

    # Impute numeric
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_processed[numeric_cols] = num_imputer.fit_transform(X_processed[numeric_cols])

    # Impute & encode categorical
    if categorical_cols:
        X_processed[categorical_cols] = X_processed[categorical_cols].astype(str)
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_processed[categorical_cols] = cat_imputer.fit_transform(X_processed[categorical_cols])
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))

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
        "excluded_cols": exclude_cols,
    }

    return X_scaled, col_info


def fit_transform_train(X_train, col_info):
    """
    Fit imputer and scaler ONLY on training set, returning transformed DataFrame and pipeline object.
    Enforces strict zero data-leakage.
    """
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
    """
    Transform test set using fitted pipeline transformers without refitting (preventing leakage).
    Handles unseen categorical classes safely.
    """
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
            # Handle unseen categories gracefully
            X_test[col] = X_test[col].apply(lambda v, k=known, fb=fallback: v if v in k else fb)
            X_test[col] = le.transform(X_test[col])

    if numeric_cols and "scaler" in pipeline:
        X_test[numeric_cols] = pipeline["scaler"].transform(X_test[numeric_cols])

    return X_test
