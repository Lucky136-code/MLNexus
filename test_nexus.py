"""
Comprehensive Automated Unit & Integration Tests for MLNexus AutoML Engine.
"""

import io
import pickle
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import (
    load_dataset, validate_dataset, analyze_target_and_features,
    prepare_features_and_target, prepare_unsupervised_features,
    fit_transform_train, transform_test, auto_detect_task_type,
    detect_id_and_date_columns
)
from models import (
    train_all_models, train_unsupervised_models,
    build_classifiers, build_regressors, build_unsupervised_models,
    export_model_artifact, get_feature_importance
)


class TestMLNexusPipeline(unittest.TestCase):

    def setUp(self):
        # Deterministic synthetic datasets
        np.random.seed(42)
        n = 120

        # Classification dataset
        self.clf_df = pd.DataFrame({
            "user_id": [f"ID_{i}" for i in range(n)], # ID column
            "age": np.random.randint(18, 65, size=n),
            "salary": np.random.normal(50000, 15000, size=n),
            "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
            "category": np.random.choice(["A", "B", "C"], size=n),
            "target_leak": np.random.choice([0, 1], size=n), # Will be synced with target for leak test
            "purchased": np.random.choice([0, 1], size=n, p=[0.7, 0.3]) # Imbalanced
        })
        self.clf_df["target_leak"] = self.clf_df["purchased"]

        # Regression dataset
        self.reg_df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "cat_feature": np.random.choice(["X", "Y"], size=n),
            "house_price": 100000 + 50000 * np.random.randn(n)
        })

    # 1. File Upload & Delimiter / BOM Tests
    def test_load_dataset_csv_and_tsv(self):
        # CSV
        csv_bytes = b"col1,col2\n1,a\n2,b\n3,c\n4,d\n5,e\n6,f\n7,g\n8,h\n9,i\n10,j"
        f_csv = io.BytesIO(csv_bytes)
        f_csv.name = "data.csv"
        df_csv = load_dataset(f_csv)
        self.assertEqual(df_csv.shape, (10, 2))

        # TSV
        tsv_bytes = b"col1\tcol2\n1\ta\n2\tb\n3\tc\n4\td\n5\te\n6\tf\n7\tg\n8\th\n9\ti\n10\tj"
        f_tsv = io.BytesIO(tsv_bytes)
        f_tsv.name = "data.tsv"
        df_tsv = load_dataset(f_tsv)
        self.assertEqual(df_tsv.shape, (10, 2))

        # UTF-8 BOM
        bom_bytes = "\ufeffcol1,col2\n1,10\n2,20\n3,30\n4,40\n5,50\n6,60\n7,70\n8,80\n9,90\n10,100".encode("utf-8-sig")
        f_bom = io.BytesIO(bom_bytes)
        f_bom.name = "data_bom.csv"
        df_bom = load_dataset(f_bom)
        self.assertEqual(df_bom.shape, (10, 2))

    def test_load_dataset_empty_and_malformed_files(self):
        # Empty file (0 bytes)
        with self.assertRaises(ValueError):
            load_dataset(io.BytesIO(b""))

        # Header only (0 data rows)
        with self.assertRaises(ValueError):
            load_dataset(io.BytesIO(b"col1,col2\n"))

    # 2. Preprocessing & Leakage Tests
    def test_zero_data_leakage(self):
        X, y, encoder, col_info = prepare_features_and_target(self.clf_df, "purchased", "classification", exclude_cols=["user_id", "target_leak"])
        
        # Verify target is excluded from features
        self.assertNotIn("purchased", X.columns)
        self.assertNotIn("target_leak", X.columns)
        self.assertNotIn("user_id", X.columns)

        # Split
        X_tr = X.iloc[:80]
        X_te = X.iloc[80:]

        # Fit train, transform test
        X_tr_proc, pipeline = fit_transform_train(X_tr, col_info)
        X_te_proc = transform_test(X_te, col_info, pipeline)

        # Check shapes match
        self.assertEqual(X_tr_proc.shape[1], X_te_proc.shape[1])
        self.assertFalse(X_tr_proc.isna().any().any())
        self.assertFalse(X_te_proc.isna().any().any())

    def test_unseen_categorical_levels_handling(self):
        X, y, encoder, col_info = prepare_features_and_target(self.clf_df, "purchased", "classification")
        X_tr = X.iloc[:80].copy()
        X_te = X.iloc[80:].copy()

        # Add completely new categorical value in test set
        X_te.loc[X_te.index[0], "category"] = "UNSEEN_CATEGORY_999"

        X_tr_proc, pipeline = fit_transform_train(X_tr, col_info)
        # Should transform test without crashing
        X_te_proc = transform_test(X_te, col_info, pipeline)
        self.assertEqual(len(X_te_proc), len(X_te))

    # 3. Diagnostics & ID/Date Detection Tests
    def test_target_diagnostics_and_leakage(self):
        diag = analyze_target_and_features(self.clf_df, "purchased", "classification")
        self.assertIn("user_id", diag["id_cols"])
        self.assertIn("target_leak", diag["leakage_cols"])
        self.assertTrue(len(diag["warnings"]) > 0)

    # 4. BUG G REGRESSION TEST: Metric & Confusion Matrix Mathematical Alignment
    def test_bug_g_confusion_matrix_metric_alignment(self):
        # Synthetic binary test predictions & ground truth
        y_test = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        preds  = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        self.assertEqual((tn, fp, fn, tp), (5, 1, 2, 2))

        # Binary formulas
        expected_acc = (tp + tn) / (tp + tn + fp + fn) # 7/10 = 0.70
        expected_prec = tp / (tp + fp)                 # 2/3 = 0.6667
        expected_rec = tp / (tp + fn)                  # 2/4 = 0.50
        expected_f1 = 2 * tp / (2 * tp + fp + fn)      # 4/7 = 0.5714

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="binary", pos_label=1)
        rec = recall_score(y_test, preds, average="binary", pos_label=1)
        f1 = f1_score(y_test, preds, average="binary", pos_label=1)

        self.assertAlmostEqual(acc, expected_acc, places=4)
        self.assertAlmostEqual(prec, expected_prec, places=4)
        self.assertAlmostEqual(rec, expected_rec, places=4)
        self.assertAlmostEqual(f1, expected_f1, places=4)

    # 5. Supervised Model Pipelines Execution
    def test_train_all_classifiers(self):
        X, y, encoder, col_info = prepare_features_and_target(self.clf_df, "purchased", "classification")
        X_tr, X_te = X.iloc[:80], X.iloc[80:]
        y_tr, y_te = y.iloc[:80], y.iloc[80:]
        X_tr, pipe = fit_transform_train(X_tr, col_info)
        X_te = transform_test(X_te, col_info, pipe)

        trained, failed = train_all_models(X_tr, X_te, y_tr, y_te, "classification", col_info=col_info)
        self.assertGreater(len(trained), 0)
        self.assertEqual(len(failed), 0)

    def test_train_all_regressors(self):
        X, y, encoder, col_info = prepare_features_and_target(self.reg_df, "house_price", "regression")
        X_tr, X_te = X.iloc[:80], X.iloc[80:]
        y_tr, y_te = y.iloc[:80], y.iloc[80:]
        X_tr, pipe = fit_transform_train(X_tr, col_info)
        X_te = transform_test(X_te, col_info, pipe)

        trained, failed = train_all_models(X_tr, X_te, y_tr, y_te, "regression", col_info=col_info)
        self.assertGreater(len(trained), 0)
        self.assertEqual(len(failed), 0)

    # 6. Unsupervised Pipeline Execution
    def test_unsupervised_models(self):
        X_scaled, col_info = prepare_unsupervised_features(self.clf_df.drop(columns=["user_id", "signup_date"]))
        trained, failed = train_unsupervised_models(X_scaled)
        self.assertGreater(len(trained), 0)

    # 7. Model Pickle Export & Serialization Test
    def test_export_model_artifact(self):
        X, y, encoder, col_info = prepare_features_and_target(self.clf_df, "purchased", "classification")
        X_tr, X_te = X.iloc[:80], X.iloc[80:]
        y_tr, y_te = y.iloc[:80], y.iloc[80:]
        X_tr, pipe = fit_transform_train(X_tr, col_info)
        X_te = transform_test(X_te, col_info, pipe)

        trained, failed = train_all_models(X_tr, X_te, y_tr, y_te, "classification", col_info=col_info)
        best_model = trained[0]["_model"]

        artifact_bytes = export_model_artifact(best_model, pipe, encoder, col_info, "classification")
        self.assertTrue(len(artifact_bytes) > 0)

        # Verify unpickling
        unpickled = pickle.loads(artifact_bytes)
        self.assertIn("model", unpickled)
        self.assertIn("preprocessing_pipeline", unpickled)
        preds = unpickled["model"].predict(X_te)
        self.assertEqual(len(preds), len(X_te))


if __name__ == "__main__":
    unittest.main()
