"""
Model training and hyperparameter tuning definitions.
"""

import time

import numpy as np
import pandas as pd
from sklearn.cluster import (AgglomerativeClustering, Birch, DBSCAN, KMeans)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (Lasso, LinearRegression,
                                  LogisticRegression, Ridge,
                                  RidgeClassifier, SGDClassifier,
                                  SGDRegressor)
from sklearn.metrics import (accuracy_score, calinski_harabasz_score,
                             davies_bouldin_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor,
                               LocalOutlierFactor)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42
MAX_SVM_ROWS = 4_000
MAX_IMPORTANCE_FEATURES = 15
CV_FOLDS = 3
MAX_CV_ROWS = 5_000

# Algorithm descriptions
ALGORITHM_INFO = {
    "Logistic Regression": (
        "Calculates a weighted sum of input features and applies a sigmoid function to estimate class probabilities."
    ),
    "Ridge Classifier": (
        "Linear classifier with L2 regularization penalty to stabilize correlated features."
    ),
    "SGD Classifier": (
        "Linear model trained via Stochastic Gradient Descent. Fast and memory efficient."
    ),
    "Linear SVM": (
        "Finds a linear hyperplane maximizing the margin between classes."
    ),
    "SVM (RBF Kernel)": (
        "Uses Radial Basis Function kernel to project data into higher dimensions for non-linear decision boundaries."
    ),
    "Gaussian Naive Bayes": (
        "Probabilistic classifier applying Bayes theorem assuming feature independence."
    ),
    "K-Nearest Neighbors": (
        "Classifies samples based on majority vote of nearest training instances."
    ),
    "Decision Tree": (
        "Tree model splitting data on feature thresholds to maximize node purity."
    ),
    "Random Forest": (
        "Ensemble of decision trees trained on bootstrap samples with random feature selection."
    ),
    "Extra Trees": (
        "Extremely randomized trees with random split thresholds for higher variance reduction."
    ),
    "AdaBoost": (
        "Sequentially trains weak decision trees, placing higher weight on misclassified points."
    ),
    "Gradient Boosting": (
        "Builds trees sequentially to minimize the residual errors of prior iterations."
    ),
    "Hist Gradient Boosting": (
        "Histogram-based gradient boosting optimized for faster training speeds."
    ),
    "MLP Neural Network": (
        "Multi-layer perceptron neural network with non-linear activation functions."
    ),
    "Linear Discriminant Analysis": (
        "Models class distributions with shared covariance to find a linear decision boundary."
    ),
    "Quadratic Discriminant Analysis": (
        "Models each class with its own covariance matrix for quadratic decision boundaries."
    ),
}

REGRESSION_INFO = {
    "Linear Regression": (
        "Fits a linear model minimizing ordinary least squares residual sum of squares."
    ),
    "Ridge Regression": (
        "Linear regression with L2 regularization to prevent overfitting on collinear features."
    ),
    "Lasso Regression": (
        "Linear regression with L1 regularization that can shrink coefficients to zero for feature selection."
    ),
    "SGD Regressor": (
        "Linear regression model fitted using Stochastic Gradient Descent."
    ),
    "SVR (RBF Kernel)": (
        "Support vector regressor using RBF kernel to capture non-linear relationships."
    ),
    "K-Nearest Neighbors": (
        "Predicts target values by averaging the targets of the k-nearest neighbors."
    ),
    "Decision Tree": (
        "Tree regressor splitting features to minimize mean squared error."
    ),
    "Random Forest": (
        "Ensemble of decision tree regressors averaged to reduce variance."
    ),
    "Extra Trees": (
        "Randomized tree ensemble regressor providing fast and stable predictions."
    ),
    "AdaBoost": (
        "Sequential ensemble fitting decision trees on current prediction residuals."
    ),
    "Gradient Boosting": (
        "Boosting regressor optimizing pseudo-residuals via gradient descent."
    ),
    "Hist Gradient Boosting": (
        "Histogram binned gradient boosting regressor for fast performance."
    ),
    "MLP Neural Network": (
        "Multi-layer perceptron regressor for complex continuous target patterns."
    ),
}

UNSUPERVISED_INFO = {
    "K-Means Clustering": (
        "Partitions dataset into K distinct clusters by minimizing variance within each cluster (sum of squared Euclidean distances to centroids)."
    ),
    "Agglomerative Clustering": (
        "Hierarchical bottom-up clustering algorithm that recursively merges pairs of clusters based on distance metric."
    ),
    "DBSCAN Clustering": (
        "Density-based spatial clustering that groups closely packed points and detects low-density noise/outlier points."
    ),
    "Gaussian Mixture Model": (
        "Probabilistic generative clustering model fitting a mixture of multi-dimensional Gaussian distributions."
    ),
    "BIRCH Clustering": (
        "Memory-efficient hierarchical clustering using Clustering Feature Trees, ideal for fast scalable clustering."
    ),
    "Principal Component Analysis": (
        "Linear dimensionality reduction projecting data into orthogonal principal components to maximize captured variance."
    ),
    "Truncated SVD": (
        "Dimensionality reduction using Singular Value Decomposition on feature matrices without mean centering."
    ),
    "Isolation Forest": (
        "Tree-based anomaly detection algorithm isolating outliers by randomly partitioning feature values."
    ),
    "One-Class SVM": (
        "Unsupervised support vector model learning a tight decision boundary surrounding normal data points to detect novelties."
    ),
    "Local Outlier Factor": (
        "Measures local density deviation of a sample relative to its k-nearest neighbors to detect isolated outliers."
    ),
}

CLASSIFICATION_PARAM_GRIDS = {
    "Logistic Regression":            {"C": [0.1, 1.0, 10.0]},
    "Ridge Classifier":               {"alpha": [0.1, 1.0, 10.0]},
    "SGD Classifier":                 {"alpha": [1e-4, 1e-3, 1e-2]},
    "Linear SVM":                     {"C": [0.1, 1.0, 10.0]},
    "SVM (RBF Kernel)":               {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "Gaussian Naive Bayes":           {},
    "K-Nearest Neighbors":            {"n_neighbors": [3, 5, 7, 11]},
    "Decision Tree":                  {"max_depth": [None, 5, 10]},
    "Random Forest":                  {"n_estimators": [100, 200]},
    "Extra Trees":                    {"n_estimators": [100, 200]},
    "AdaBoost":                       {"n_estimators": [50, 100]},
    "Gradient Boosting":              {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "Hist Gradient Boosting":         {"max_iter": [100, 200]},
    "MLP Neural Network":              {"hidden_layer_sizes": [(64, 32), (128, 64)]},
    "Linear Discriminant Analysis":    {},
    "Quadratic Discriminant Analysis": {"reg_param": [0.0, 0.1]},
}

REGRESSION_PARAM_GRIDS = {
    "Linear Regression":      {},
    "Ridge Regression":       {"alpha": [0.1, 1.0, 10.0]},
    "Lasso Regression":       {"alpha": [0.1, 1.0, 10.0]},
    "SGD Regressor":          {"alpha": [1e-4, 1e-3, 1e-2]},
    "SVR (RBF Kernel)":       {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "K-Nearest Neighbors":    {"n_neighbors": [3, 5, 7, 11]},
    "Decision Tree":          {"max_depth": [None, 5, 10]},
    "Random Forest":          {"n_estimators": [100, 200]},
    "Extra Trees":            {"n_estimators": [100, 200]},
    "AdaBoost":               {"n_estimators": [50, 100]},
    "Gradient Boosting":      {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "Hist Gradient Boosting": {"max_iter": [100, 200]},
    "MLP Neural Network":     {"hidden_layer_sizes": [(64, 32), (128, 64)]},
}

CLASSIFICATION_RESULT_COLUMNS = [
    "Rank", "Model", "Family",
    "Accuracy", "Precision", "Recall", "F1-Score",
    "CV F1 (mean±std)",
    "Training Time (s)",
]

REGRESSION_RESULT_COLUMNS = [
    "Rank", "Model", "Family",
    "R²", "MAE", "RMSE",
    "CV R² (mean±std)",
    "Training Time (s)",
]

UNSUPERVISED_RESULT_COLUMNS = [
    "Rank", "Model", "Family",
    "Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin",
    "Clusters / Outliers",
    "Training Time (s)",
]


def build_classifiers(n_train):
    """Return classification model specifications."""
    specs = [
        ("Logistic Regression",
         LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
         "Linear Model", None),
        ("Ridge Classifier",
         RidgeClassifier(random_state=RANDOM_STATE),
         "Linear Model", None),
        ("SGD Classifier",
         SGDClassifier(max_iter=1000, random_state=RANDOM_STATE),
         "Linear Model", None),
        ("Linear SVM",
         LinearSVC(max_iter=5000, random_state=RANDOM_STATE),
         "Support Vector Machine", None),
        ("SVM (RBF Kernel)",
         SVC(random_state=RANDOM_STATE),
         "Support Vector Machine", MAX_SVM_ROWS),
        ("Gaussian Naive Bayes", GaussianNB(), "Bayesian", None),
        ("K-Nearest Neighbors",
         KNeighborsClassifier(n_neighbors=min(5, max(1, n_train))),
         "Instance-Based", None),
        ("Decision Tree",
         DecisionTreeClassifier(random_state=RANDOM_STATE),
         "Tree / Ensemble", None),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
         "Tree / Ensemble", None),
        ("Extra Trees",
         ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
         "Tree / Ensemble", None),
        ("AdaBoost",
         AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
         "Tree / Ensemble", None),
        ("Gradient Boosting",
         GradientBoostingClassifier(random_state=RANDOM_STATE),
         "Tree / Ensemble", None),
        ("Hist Gradient Boosting",
         HistGradientBoostingClassifier(random_state=RANDOM_STATE),
         "Tree / Ensemble", None),
        ("MLP Neural Network",
         MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                       early_stopping=(n_train >= 100),
                       random_state=RANDOM_STATE),
         "Neural Network", None),
        ("Linear Discriminant Analysis",
         LinearDiscriminantAnalysis(), "Discriminant Analysis", None),
        ("Quadratic Discriminant Analysis",
         QuadraticDiscriminantAnalysis(), "Discriminant Analysis", None),
    ]
    return [{"name": n, "estimator": e, "family": f, "cap": c}
            for n, e, f, c in specs]


def build_regressors(n_train):
    """Return regression model specifications."""
    specs = [
        ("Linear Regression",
         LinearRegression(), "Linear Model", None),
        ("Ridge Regression",
         Ridge(random_state=RANDOM_STATE), "Linear Model", None),
        ("Lasso Regression",
         Lasso(max_iter=3000, random_state=RANDOM_STATE), "Linear Model", None),
        ("SGD Regressor",
         SGDRegressor(max_iter=1000, random_state=RANDOM_STATE), "Linear Model", None),
        ("SVR (RBF Kernel)",
         SVR(), "Support Vector Machine", MAX_SVM_ROWS),
        ("K-Nearest Neighbors",
         KNeighborsRegressor(n_neighbors=min(5, max(1, n_train))),
         "Instance-Based", None),
        ("Decision Tree",
         DecisionTreeRegressor(random_state=RANDOM_STATE), "Tree / Ensemble", None),
        ("Random Forest",
         RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
         "Tree / Ensemble", None),
        ("Extra Trees",
         ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
         "Tree / Ensemble", None),
        ("AdaBoost",
         AdaBoostRegressor(n_estimators=100, random_state=RANDOM_STATE),
         "Tree / Ensemble", None),
        ("Gradient Boosting",
         GradientBoostingRegressor(random_state=RANDOM_STATE), "Tree / Ensemble", None),
        ("Hist Gradient Boosting",
         HistGradientBoostingRegressor(random_state=RANDOM_STATE), "Tree / Ensemble", None),
        ("MLP Neural Network",
         MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                       early_stopping=(n_train >= 100),
                       random_state=RANDOM_STATE),
          "Neural Network", None),
    ]
    return [{"name": n, "estimator": e, "family": f, "cap": c}
            for n, e, f, c in specs]


def build_unsupervised_models(n_samples, n_features):
    """Return unsupervised model specifications."""
    k = min(5, max(2, n_samples // 10))
    specs = [
        ("K-Means Clustering", KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10), "Clustering", "clustering"),
        ("Agglomerative Clustering", AgglomerativeClustering(n_clusters=k), "Clustering", "clustering"),
        ("DBSCAN Clustering", DBSCAN(eps=0.5, min_samples=3), "Clustering", "clustering"),
        ("Gaussian Mixture Model", GaussianMixture(n_components=k, random_state=RANDOM_STATE), "Clustering", "clustering"),
        ("BIRCH Clustering", Birch(n_clusters=k), "Clustering", "clustering"),
        ("Principal Component Analysis", PCA(n_components=min(3, max(1, n_features)), random_state=RANDOM_STATE), "Dimensionality Reduction", "dim_reduction"),
        ("Truncated SVD", TruncatedSVD(n_components=min(3, max(1, n_features - 1 if n_features > 1 else 1)), random_state=RANDOM_STATE), "Dimensionality Reduction", "dim_reduction"),
        ("Isolation Forest", IsolationForest(random_state=RANDOM_STATE, contamination=0.05), "Anomaly Detection", "anomaly"),
        ("One-Class SVM", OneClassSVM(gamma="scale", nu=0.05), "Anomaly Detection", "anomaly"),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=min(20, max(2, n_samples - 1)), novelty=True), "Anomaly Detection", "anomaly"),
    ]
    return [{"name": n, "estimator": e, "family": f, "category": cat} for n, e, f, cat in specs]


def train_unsupervised_models(X_scaled, progress_bar=None):
    """Train unsupervised models and evaluate cluster/reduction/anomaly metrics."""
    n_samples, n_features = X_scaled.shape
    specs = build_unsupervised_models(n_samples, n_features)
    trained, failed = [], []
    total = len(specs)

    for i, spec in enumerate(specs):
        if progress_bar:
            progress_bar.progress(
                0.30 + 0.60 * (i / total),
                text=f"Training {spec['name']} ({i + 1}/{total})…",
            )
        try:
            start = time.perf_counter()
            model = spec["estimator"]
            cat = spec["category"]
            preds = None
            sil, ch, db = 0.0, 0.0, 0.0
            cluster_desc = "—"

            if cat == "clustering":
                if hasattr(model, "fit_predict"):
                    preds = model.fit_predict(X_scaled)
                else:
                    model.fit(X_scaled)
                    preds = model.predict(X_scaled)

                valid_labels = set(preds) - {-1}
                n_clusters = len(valid_labels)
                cluster_desc = f"{n_clusters} Clusters"

                if 2 <= n_clusters < len(X_scaled):
                    try:
                        sil = float(silhouette_score(X_scaled, preds))
                        ch = float(calinski_harabasz_score(X_scaled, preds))
                        db = float(davies_bouldin_score(X_scaled, preds))
                    except Exception:
                        pass
                sort_val = sil

            elif cat == "dim_reduction":
                model.fit(X_scaled)
                exp_var = float(np.sum(model.explained_variance_ratio_))
                sil = exp_var
                ch = np.nan
                db = np.nan
                cluster_desc = f"{model.n_components_} Components"
                sort_val = exp_var

            elif cat == "anomaly":
                if hasattr(model, "fit_predict"):
                    preds = model.fit_predict(X_scaled)
                else:
                    model.fit(X_scaled)
                    preds = model.predict(X_scaled)

                n_outliers = int(np.sum(preds == -1))
                pct = round(100 * n_outliers / len(X_scaled), 1)
                cluster_desc = f"{n_outliers} Outliers ({pct}%)"
                sil = 1.0 - (n_outliers / len(X_scaled))
                ch = np.nan
                db = np.nan
                sort_val = sil

            elapsed = time.perf_counter() - start

            record = {
                "Model": spec["name"],
                "Family": spec["family"],
                "Silhouette Score": round(sil, 4) if not np.isnan(sil) else 0.0,
                "Calinski-Harabasz": round(ch, 2) if not np.isnan(ch) else np.nan,
                "Davies-Bouldin": round(db, 4) if not np.isnan(db) else np.nan,
                "Clusters / Outliers": cluster_desc,
                "Training Time (s)": round(elapsed, 3),
                "_predictions": preds,
                "_model": model,
                "_sort_key": sort_val,
                "_category": cat,
            }
            trained.append(record)

        except Exception as exc:
            failed.append(f"{spec['name']}: {exc}")

    return trained, failed


def train_all_models(X_train, X_test, y_train, y_test,
                     task_type, progress_bar, run_cv=False):
    """Train all models, compute metrics and optional CV scores."""
    specs = (build_classifiers(len(X_train))
             if task_type == "classification"
             else build_regressors(len(X_train)))
    trained, failed = [], []
    total = len(specs)

    cv_n = min(MAX_CV_ROWS, len(X_train))
    X_cv = X_train.iloc[:cv_n]
    y_cv = y_train.iloc[:cv_n]
    cv_scoring = "f1_weighted" if task_type == "classification" else "r2"

    for i, spec in enumerate(specs):
        progress_bar.progress(
            0.30 + 0.60 * (i / total),
            text=f"Training {spec['name']} ({i + 1}/{total})…",
        )
        try:
            Xtr, ytr = X_train, y_train
            if spec["cap"] and len(X_train) > spec["cap"]:
                Xtr = X_train.sample(spec["cap"], random_state=RANDOM_STATE)
                ytr = y_train.loc[Xtr.index]

            start = time.perf_counter()
            model = spec["estimator"]
            model.fit(Xtr, ytr)
            preds = model.predict(X_test)
            elapsed = time.perf_counter() - start

            cv_mean_str = "—"
            if run_cv:
                try:
                    fresh = spec["estimator"].__class__(**spec["estimator"].get_params())
                    cv_scores = cross_val_score(
                        fresh, X_cv, y_cv,
                        cv=CV_FOLDS, scoring=cv_scoring, n_jobs=1,
                    )
                    cv_mean_str = f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                except Exception:
                    cv_mean_str = "error"

            if task_type == "classification":
                record = {
                    "Model": spec["name"],
                    "Family": spec["family"],
                    "Accuracy": accuracy_score(y_test, preds),
                    "Precision": precision_score(y_test, preds,
                                                 average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, preds,
                                           average="weighted", zero_division=0),
                    "F1-Score": f1_score(y_test, preds,
                                         average="weighted", zero_division=0),
                    "CV F1 (mean±std)": cv_mean_str,
                    "Training Time (s)": round(elapsed, 3),
                    "_predictions": preds,
                    "_model": model,
                    "_sort_key": f1_score(y_test, preds,
                                           average="weighted", zero_division=0),
                }
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                record = {
                    "Model": spec["name"],
                    "Family": spec["family"],
                    "R²": r2_score(y_test, preds),
                    "MAE": mean_absolute_error(y_test, preds),
                    "RMSE": rmse,
                    "CV R² (mean±std)": cv_mean_str,
                    "Training Time (s)": round(elapsed, 3),
                    "_predictions": preds,
                    "_model": model,
                    "_sort_key": r2_score(y_test, preds),
                }

            trained.append(record)

        except Exception as exc:
            failed.append(f"{spec['name']}: {exc}")

    return trained, failed


def tune_best_model(best_model, best_name, X_train, y_train,
                    task_type, progress_bar):
    """Tune hyperparameters for the winning model using GridSearchCV."""
    progress_bar.progress(
        0.95,
        text=f"Tuning hyperparameters for {best_name}…",
    )

    grids = (CLASSIFICATION_PARAM_GRIDS
             if task_type == "classification"
             else REGRESSION_PARAM_GRIDS)
    param_grid = grids.get(best_name, {})

    if not param_grid:
        return best_model, None

    scoring = "f1_weighted" if task_type == "classification" else "r2"

    n_tune = min(MAX_CV_ROWS, len(X_train))
    X_t = X_train.iloc[:n_tune]
    y_t = y_train.iloc[:n_tune]

    try:
        gs = GridSearchCV(
            best_model, param_grid,
            cv=CV_FOLDS, scoring=scoring,
            n_jobs=-1, refit=True,
        )
        gs.fit(X_t, y_t)

        best_params_all = {**best_model.get_params(), **gs.best_params_}
        tuned = best_model.__class__(**best_params_all)
        tuned.fit(X_train, y_train)

        tuning_info = {
            "best_params": gs.best_params_,
            "cv_score": round(gs.best_score_, 4),
            "scoring": scoring,
        }
        return tuned, tuning_info

    except Exception:
        return best_model, None


def get_feature_importance(model, X_test, y_test, feature_names, task_type):
    """Extract feature importance from trained estimator."""
    y_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        method = "Feature importance based on tree impurity reduction"
    elif hasattr(model, "coefs_"):
        values = np.abs(model.coefs_[0]).mean(axis=1)
        method = "Mean absolute weights of first neural network layer"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        method = "Mean absolute value of model coefficients"
    else:
        sample_n = min(300, len(X_test))
        scoring = ("accuracy" if task_type == "classification" else "r2")
        perm = permutation_importance(
            model,
            X_test.iloc[:sample_n],
            y_arr[:sample_n],
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring=scoring,
            n_jobs=1,
        )
        values = perm.importances_mean
        method = "Permutation importance score drop"

    importance = (
        pd.DataFrame({"Feature": feature_names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance, method
