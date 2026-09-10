"""
Blood Pressure Estimation Model Training.

Trains classical ML models (Random Forest, XGBoost) on PPG features
to estimate systolic (SBP) and diastolic (DBP) blood pressure.
Evaluates against BHS (British Hypertension Society) and AAMI standards.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


# ── BHS evaluation ────────────────────────────────────────────────────────
def bhs_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    British Hypertension Society (BHS) protocol metrics.

    BHS Grade A: ≥60% within 5 mmHg, ≥85% within 10 mmHg, ≥95% within 15 mmHg
    BHS Grade B: ≥50% within 5 mmHg, ≥75% within 10 mmHg, ≥90% within 15 mmHg
    BHS Grade C: ≥40% within 5 mmHg, ≥65% within 10 mmHg, ≥85% within 15 mmHg
    """
    abs_errors = np.abs(y_true - y_pred)
    n = len(y_true)

    p5 = np.mean(abs_errors <= 5) * 100
    p10 = np.mean(abs_errors <= 10) * 100
    p15 = np.mean(abs_errors <= 15) * 100

    # Determine BHS grade
    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        grade = "A"
    elif p5 >= 50 and p10 >= 75 and p15 >= 90:
        grade = "B"
    elif p5 >= 40 and p10 >= 65 and p15 >= 85:
        grade = "C"
    else:
        grade = "D"

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "within_5mmHg": round(p5, 1),
        "within_10mmHg": round(p10, 1),
        "within_15mmHg": round(p15, 1),
        "bhs_grade": grade,
        "mean_error": float(np.mean(y_pred - y_true)),
        "std_error": float(np.std(y_pred - y_true)),
    }


def aami_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    AAMI standard: mean error ≤ 5 mmHg, std of error ≤ 8 mmHg.
    """
    errors = y_pred - y_true
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))
    passed = abs(mean_err) <= 5 and std_err <= 8
    return {
        "mean_error": mean_err,
        "std_error": std_err,
        "aami_pass": passed,
    }


# ── model training ────────────────────────────────────────────────────────
def train_evaluate_rf(
    X: np.ndarray,
    y_sbp: np.ndarray,
    y_dbp: np.ndarray,
    n_folds: int = 5,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Dict:
    """
    Train Random Forest with k-fold cross-validation.
    Returns evaluation metrics for SBP and DBP.
    """
    model_sbp = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=10,
        min_samples_leaf=3, random_state=random_state, n_jobs=-1,
    )
    model_dbp = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=10,
        min_samples_leaf=3, random_state=random_state, n_jobs=-1,
    )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # SBP
    pred_sbp = cross_val_predict(model_sbp, X, y_sbp, cv=kf)
    # DBP
    pred_dbp = cross_val_predict(model_dbp, X, y_dbp, cv=kf)

    results = {
        "model": "RandomForest",
        "n_samples": len(y_sbp),
        "n_features": X.shape[1],
        "n_folds": n_folds,
        "sbp": bhs_metrics(y_sbp, pred_sbp),
        "dbp": bhs_metrics(y_dbp, pred_dbp),
        "sbp_aami": aami_metrics(y_sbp, pred_sbp),
        "dbp_aami": aami_metrics(y_dbp, pred_dbp),
    }

    # Feature importance (fit once on full data)
    model_sbp.fit(X, y_sbp)
    model_dbp.fit(X, y_dbp)
    results["sbp_feature_importance"] = _top_features(
        model_sbp.feature_importances_, None, top_k=10
    )
    results["dbp_feature_importance"] = _top_features(
        model_dbp.feature_importances_, None, top_k=10
    )

    return results


def train_evaluate_xgb(
    X: np.ndarray,
    y_sbp: np.ndarray,
    y_dbp: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict:
    """Train XGBoost with k-fold cross-validation."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return {"error": "XGBoost not installed. Run: pip install xgboost"}

    model_sbp = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=random_state, verbosity=0,
    )
    model_dbp = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=random_state, verbosity=0,
    )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    pred_sbp = cross_val_predict(model_sbp, X, y_sbp, cv=kf)
    pred_dbp = cross_val_predict(model_dbp, X, y_dbp, cv=kf)

    results = {
        "model": "XGBoost",
        "n_samples": len(y_sbp),
        "n_features": X.shape[1],
        "n_folds": n_folds,
        "sbp": bhs_metrics(y_sbp, pred_sbp),
        "dbp": bhs_metrics(y_dbp, pred_dbp),
        "sbp_aami": aami_metrics(y_sbp, pred_sbp),
        "dbp_aami": aami_metrics(y_dbp, pred_dbp),
    }

    model_sbp.fit(X, y_sbp)
    model_dbp.fit(X, y_dbp)
    results["sbp_feature_importance"] = _top_features(
        model_sbp.feature_importances_, None, top_k=10
    )
    results["dbp_feature_importance"] = _top_features(
        model_dbp.feature_importances_, None, top_k=10
    )

    return results


# ── helper ────────────────────────────────────────────────────────────────
def _top_features(importances: np.ndarray, feature_names: Optional[list],
                  top_k: int = 10) -> list:
    """Return top-k feature indices sorted by importance."""
    idx = np.argsort(importances)[::-1][:top_k]
    if feature_names is not None:
        return [(int(i), float(importances[i]), feature_names[i]) for i in idx]
    return [(int(i), float(importances[i])) for i in idx]


def _top_features_named(importances: np.ndarray, feature_names: list,
                        top_k: int = 10) -> list:
    return [(feature_names[i], float(importances[i]))
            for i in np.argsort(importances)[::-1][:top_k]]


# ── full pipeline ─────────────────────────────────────────────────────────
def run_ml_pipeline(
    X: np.ndarray,
    y_sbp: np.ndarray,
    y_dbp: np.ndarray,
    feature_names: Optional[list] = None,
    n_folds: int = 5,
) -> Dict:
    """
    Run full ML training pipeline: RF + XGBoost with cross-validation.

    Returns:
        {
            "random_forest": {...},
            "xgboost": {...},
        }
    """
    print(f"Training on {len(y_sbp)} samples, {X.shape[1]} features, {n_folds}-fold CV\n")

    results = {}

    # Random Forest
    print("=== Random Forest ===")
    rf = train_evaluate_rf(X, y_sbp, y_dbp, n_folds=n_folds)
    results["random_forest"] = rf
    _print_results(rf)

    # XGBoost
    print("\n=== XGBoost ===")
    xgb = train_evaluate_xgb(X, y_sbp, y_dbp, n_folds=n_folds)
    results["xgboost"] = xgb
    if "error" not in xgb:
        _print_results(xgb)
    else:
        print(f"  Skipped: {xgb['error']}")

    # Feature importance
    if feature_names is not None:
        for bp_type in ["sbp", "dbp"]:
            key = f"{bp_type}_feature_importance"
            if key in rf:
                print(f"\n=== Top {bp_type.upper()} Features (RF) ===")
                for idx, val in rf[key][:10]:
                    if idx < len(feature_names):
                        print(f"  {feature_names[idx]:30s} {val:.4f}")

    return results


def _print_results(r: Dict):
    """Print evaluation metrics for a model."""
    for bp_type in ["sbp", "dbp"]:
        m = r[bp_type]
        aami = r.get(f"{bp_type}_aami", {})
        print(f"  {bp_type.upper()}: MAE={m['mae']:.1f}, RMSE={m['rmse']:.1f}, "
              f"R²={m['r2']:.3f}, ≤5mmHg={m['within_5mmHg']:.0f}%, "
              f"BHS={m['bhs_grade']}, AAMI={aami.get('aami_pass','?')}")
