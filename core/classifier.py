"""
core/classifier.py
Domain-integrated water quality classification and benchmarking.

This module supports:
1) Rule-based environmental screening (safe/moderate/poor)
2) Supervised ML benchmarking across multiple model families
3) Ensemble learning (voting + stacking)
4) Feature-importance and correlation-based interpretability
5) Recommendation generation for degraded quality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Sklearn is optional at runtime; rule-based mode still works without it.
try:
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
        StackingClassifier,
        VotingClassifier,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    SKLEARN_AVAILABLE = False


FEATURE_COLS = [
    "do",
    "ph",
    "ammonia",
    "bod",
    "temp",
    "nitrogen",
    "nitrate",
    "orthophosphate",
    "ccme_values",
]

WQI_LABELS = ["Excellent", "Good", "Marginal", "Fair", "Poor"]
_DEGRADED_WQI = {"Marginal", "Fair", "Poor"}

_CCME_TO_STATUS = {
    "Excellent": "Safe",
    "Good": "Safe",
    "Marginal": "Moderate",
    "Fair": "Moderate",
    "Poor": "Poor",
}

THRESHOLDS = {
    "do": {"safe": (6, None), "poor": (4, None)},
    "ph": {"safe": (6.5, 8.5), "poor": (5, 9.5)},
    "ammonia": {"safe": (None, 0.5), "poor": (None, 1.0)},
    "bod": {"safe": (None, 3), "poor": (None, 6)},
    "temp": {"safe": (None, 28), "poor": (None, 35)},
    "nitrogen": {"safe": (None, 1), "poor": (None, 5)},
    "nitrate": {"safe": (None, 10), "poor": (None, 50)},
    "orthophosphate": {"safe": (None, 0.1), "poor": (None, 0.5)},
    "ccme_values": {"safe": (80, None), "poor": (45, None)},
}

POLLUTION_RULES = {
    "nutrient_runoff_reduction": (
        lambda r: _v(r.get("nitrogen")) >= 1.0
        or _v(r.get("nitrate")) >= 10.0
        or _v(r.get("orthophosphate")) >= 0.1,
        "Reduce nutrient runoff (fertilizer management, riparian buffers, stormwater control).",
    ),
    "wastewater_treatment_improvement": (
        lambda r: _v(r.get("bod")) >= 3.0 or _v(r.get("ammonia")) >= 0.5,
        "Improve wastewater treatment performance (BOD/ammonia removal and process optimization).",
    ),
    "artificial_aeration": (
        lambda r: _v(r.get("do")) <= 5.0,
        "Consider artificial aeration or hydraulic mixing to improve dissolved oxygen.",
    ),
    "industrial_discharge_regulation": (
        lambda r: _v(r.get("temp")) >= 30.0 or _v(r.get("ph")) <= 6.0 or _v(r.get("ph")) >= 9.0,
        "Tighten industrial discharge controls (thermal load, pH correction, pretreatment compliance).",
    ),
}


def _v(value: Any) -> float:
    """Safe numeric conversion helper."""
    try:
        if value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _score_param(key: str, value: Any) -> int:
    """Return 1 if value is within safe range, 0 otherwise."""
    if key not in THRESHOLDS:
        return 0
    value_f = _v(value)
    if np.isnan(value_f):
        return 0
    lo_safe, hi_safe = THRESHOLDS[key]["safe"]
    in_safe = (lo_safe is None or value_f >= lo_safe) and (hi_safe is None or value_f <= hi_safe)
    return 1 if in_safe else 0


def _rule_based_classification(row: dict[str, Any]) -> dict[str, Any]:
    """
    Rule-based classification fallback.
    Returns status in legacy Safe/Moderate/Poor classes.
    """
    keys = [k for k in THRESHOLDS if k != "ccme_values"]
    total = len(keys)
    score = sum(_score_param(k, row.get(k)) for k in keys)

    ccme_wqi = row.get("ccme_wqi")
    if ccme_wqi in _CCME_TO_STATUS:
        status = _CCME_TO_STATUS[ccme_wqi]
    elif score >= total * 0.75:
        status = "Safe"
    elif score >= total * 0.4:
        status = "Moderate"
    else:
        status = "Poor"

    color = {"Safe": "green", "Moderate": "orange", "Poor": "red"}.get(status, "grey")
    recs = generate_recommendations(row, predicted_wqi=ccme_wqi)

    return {
        "status": status,
        "color": color,
        "score": score,
        "total": total,
        "messages": recs,
        "method": "rule_based",
    }


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare ML training matrix X and label y from cleaned dataframe.
    Uses WQI labels as supervised target.
    """
    d = df.copy()
    if "ccme_wqi" not in d.columns:
        raise ValueError("Dataframe must contain `ccme_wqi` for supervised classification.")

    d = d[d["ccme_wqi"].isin(WQI_LABELS)]
    if d.empty:
        raise ValueError("No valid WQI labels found for training.")

    X = d[[c for c in FEATURE_COLS if c in d.columns]].copy()
    y = d["ccme_wqi"].astype(str)
    return X, y


def _build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """
    Shared preprocessor:
    - median imputation
    - standard scaling
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                columns,
            )
        ],
        remainder="drop",
    )


def _model_zoo(random_state: int = 42) -> dict[str, Any]:
    """Build base and ensemble classifiers."""
    if not SKLEARN_AVAILABLE:
        return {}

    # Single-model baselines
    linear_model = LogisticRegression(max_iter=1500, random_state=random_state)
    distance_model = KNeighborsClassifier(n_neighbors=9, weights="distance")
    tree_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    boosting_model = GradientBoostingClassifier(random_state=random_state)
    svm_model = SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=random_state)

    # Ensembles
    voting_hard = VotingClassifier(
        estimators=[
            ("rf", clone(tree_model)),
            ("svm", clone(svm_model)),
            ("gb", clone(boosting_model)),
        ],
        voting="hard",
    )
    voting_soft = VotingClassifier(
        estimators=[
            ("rf", clone(tree_model)),
            ("svm", clone(svm_model)),
            ("gb", clone(boosting_model)),
        ],
        voting="soft",
    )
    stacking = StackingClassifier(
        estimators=[
            ("rf", clone(tree_model)),
            ("svm", clone(svm_model)),
            ("gb", clone(boosting_model)),
        ],
        final_estimator=LogisticRegression(max_iter=1200, random_state=random_state),
        stack_method="predict_proba",
        cv=3,
        n_jobs=1,
    )

    return {
        "linear_logreg": linear_model,
        "distance_knn": distance_model,
        "tree_random_forest": tree_model,
        "boosting_gradient_boosting": boosting_model,
        "svm_rbf": svm_model,
        "ensemble_voting_hard": voting_hard,
        "ensemble_voting_soft": voting_soft,
        "ensemble_stacking": stacking,
    }


def benchmark_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train and benchmark models under identical split conditions.
    Returns trained artifacts + benchmark table + interpretability outputs.
    """
    if not SKLEARN_AVAILABLE:
        return {
            "available": False,
            "reason": "scikit-learn is not installed",
            "models": {},
            "benchmark": pd.DataFrame(),
        }

    X, y = prepare_training_data(df)
    feature_cols = X.columns.tolist()

    # Common split for objective comparison.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # Fallback when class counts are too small for stratification.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    models = _model_zoo(random_state=random_state)
    preprocessor = _build_preprocessor(feature_cols)
    rows = []
    trained_pipelines: dict[str, Any] = {}

    for name, estimator in models.items():
        pipe = Pipeline(
            steps=[
                ("prep", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rows.append(
            {
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
                "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
                "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            }
        )
        trained_pipelines[name] = pipe

    benchmark_df = pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    best_model_name = benchmark_df.iloc[0]["model"] if not benchmark_df.empty else None

    # Refit all models on full labeled data to maximize downstream inference quality.
    full_pipelines: dict[str, Any] = {}
    for name, estimator in models.items():
        full_pipe = Pipeline(steps=[("prep", clone(preprocessor)), ("model", estimator)])
        full_pipe.fit(X, y)
        full_pipelines[name] = full_pipe

    # Interpretability: RF importance + correlation matrix.
    feature_importance = {}
    if "tree_random_forest" in full_pipelines:
        rf_model = full_pipelines["tree_random_forest"].named_steps["model"]
        if hasattr(rf_model, "feature_importances_"):
            fi = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            feature_importance = {k: round(float(v), 6) for k, v in fi.items()}

    corr = (
        df[[c for c in FEATURE_COLS if c in df.columns]]
        .apply(pd.to_numeric, errors="coerce")
        .corr(numeric_only=True)
    )

    return {
        "available": True,
        "feature_columns": feature_cols,
        "models": full_pipelines,
        "benchmark": benchmark_df,
        "best_model_name": best_model_name,
        "best_model": full_pipelines.get(best_model_name),
        "feature_importance": feature_importance,
        "correlation_matrix": corr,
        "meta": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "n_samples": int(len(X)),
            "n_features": int(len(feature_cols)),
        },
    }


def _predict_wqi_from_model(model_bundle: dict[str, Any], row: dict[str, Any]) -> tuple[str | None, float | None]:
    """Predict WQI label and confidence from trained best model."""
    if not model_bundle or not model_bundle.get("available"):
        return None, None

    best_model = model_bundle.get("best_model")
    if best_model is None:
        return None, None

    feature_cols = model_bundle.get("feature_columns", FEATURE_COLS)
    X_row = pd.DataFrame([{c: row.get(c, np.nan) for c in feature_cols}])
    try:
        pred = best_model.predict(X_row)[0]
    except Exception:
        return None, None

    confidence = None
    if hasattr(best_model, "predict_proba"):
        try:
            proba = best_model.predict_proba(X_row)[0]
            confidence = float(np.max(proba))
        except Exception:
            confidence = None

    return str(pred), confidence


def generate_recommendations(row: dict[str, Any], predicted_wqi: str | None = None) -> list[str]:
    """
    Recommendation engine for degraded quality conditions.
    """
    recommendations = []

    is_degraded = predicted_wqi in _DEGRADED_WQI or _CCME_TO_STATUS.get(predicted_wqi) in {"Moderate", "Poor"}
    if not is_degraded and "ccme_wqi" in row:
        is_degraded = row["ccme_wqi"] in _DEGRADED_WQI

    for _, (rule, text) in POLLUTION_RULES.items():
        try:
            if rule(row):
                recommendations.append(text)
        except Exception:
            continue

    if is_degraded and not recommendations:
        recommendations.append("Increase monitoring frequency and investigate local pollution sources.")

    # Return unique recommendations while preserving order.
    deduped = list(dict.fromkeys(recommendations))
    return deduped[:4]


def get_shap_values(model_bundle: dict[str, Any], X_sample: pd.DataFrame) -> dict[str, Any]:
    """
    Optional SHAP analysis hook.
    Returns empty payload if SHAP is unavailable.
    """
    try:  # pragma: no cover - optional dependency
        import shap  # type: ignore
    except Exception:
        return {"available": False, "reason": "shap not installed"}

    best_model = model_bundle.get("best_model") if model_bundle else None
    if best_model is None:
        return {"available": False, "reason": "no trained best model"}

    try:
        transformed = best_model.named_steps["prep"].transform(X_sample)
        estimator = best_model.named_steps["model"]
        explainer = shap.Explainer(estimator)
        shap_values = explainer(transformed)
    except Exception as exc:
        return {"available": False, "reason": f"shap failed: {exc}"}

    return {"available": True, "values": shap_values}


def classify(row: dict[str, Any], model_bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Classify water quality from a parameter row.

    Backward-compatible output keys:
      status, color, score, total

    Extended keys:
      wqi_label, confidence, messages, method
    """
    # ML-first path if model bundle exists.
    wqi_label, confidence = _predict_wqi_from_model(model_bundle or {}, row)
    if wqi_label is not None:
        status = _CCME_TO_STATUS.get(wqi_label, "Poor")
        color = {"Safe": "green", "Moderate": "orange", "Poor": "red"}.get(status, "grey")
        keys = [k for k in THRESHOLDS if k != "ccme_values"]
        total = len(keys)
        score = sum(_score_param(k, row.get(k)) for k in keys)
        return {
            "status": wqi_label,  # keep CCME semantics for UI badge mapping
            "wqi_label": wqi_label,
            "color": color,
            "score": score,
            "total": total,
            "confidence": confidence,
            "messages": generate_recommendations(row, predicted_wqi=wqi_label),
            "method": f"ml:{model_bundle.get('best_model_name', 'unknown')}",
        }

    return _rule_based_classification(row)
