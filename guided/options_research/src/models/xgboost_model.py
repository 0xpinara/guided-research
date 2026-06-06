"""XGBoost regressor and classifier for return prediction."""

import numpy as np
import xgboost as xgb

from src.utils.logger import setup_logger
from src.utils.io_helpers import RESULTS_DIR

log = setup_logger(__name__)


def train_xgb_regressor(
    X_train, y_train, X_val, y_val, params: dict
) -> xgb.XGBRegressor:
    """Train an XGBoost regressor with early stopping."""
    model = xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.01),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        eval_metric=params.get("eval_metric_reg", "rmse"),
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    log.info("XGB Regressor: best iteration %d", model.best_iteration)
    return model


def train_xgb_classifier(
    X_train, y_train, X_val, y_val, params: dict
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with early stopping."""
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.01),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        eval_metric=params.get("eval_metric_clf", "logloss"),
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    log.info("XGB Classifier: best iteration %d", model.best_iteration)
    return model


def compute_shap_values(model, X_test):
    """Compute SHAP values for an XGBoost model."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values


def run_xgb_experiment(
    X_train, y_train_ret, y_train_dir,
    X_val, y_val_ret, y_val_dir,
    X_test, y_test_ret, y_test_dir,
    params: dict,
    feature_set_name: str,
    horizon: int,
) -> dict:
    """Run XGBoost regression + classification for one feature set and horizon.

    Returns dict with models, predictions, and SHAP values.
    """
    log.info("=== XGBoost: %s, %d-day ===", feature_set_name, horizon)

    # Regression
    reg_model = train_xgb_regressor(X_train, y_train_ret, X_val, y_val_ret, params)
    reg_pred = reg_model.predict(X_test)

    # Classification
    clf_model = train_xgb_classifier(X_train, y_train_dir, X_val, y_val_dir, params)
    clf_pred = clf_model.predict(X_test)
    clf_proba = clf_model.predict_proba(X_test)[:, 1]

    # SHAP
    shap_reg = compute_shap_values(reg_model, X_test)

    # Save models
    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reg_model.save_model(str(ckpt_dir / f"xgb_{feature_set_name}_{horizon}d_reg.json"))
    clf_model.save_model(str(ckpt_dir / f"xgb_{feature_set_name}_{horizon}d_clf.json"))

    return {
        "reg_model": reg_model,
        "clf_model": clf_model,
        "reg_pred": reg_pred,
        "clf_pred": clf_pred,
        "clf_proba": clf_proba,
        "shap_values": shap_reg,
    }
