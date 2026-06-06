"""TabNet model with built-in feature attention."""

import numpy as np

from src.utils.logger import setup_logger
from src.utils.io_helpers import RESULTS_DIR

log = setup_logger(__name__)


def train_tabnet_regressor(X_train, y_train, X_val, y_val, params: dict):
    """Train TabNet regressor."""
    from pytorch_tabnet.tab_model import TabNetRegressor

    model = TabNetRegressor(
        n_d=params.get("n_d", 64),
        n_a=params.get("n_a", 64),
        n_steps=params.get("n_steps", 5),
        gamma=params.get("gamma", 1.5),
        lambda_sparse=params.get("lambda_sparse", 0.001),
        seed=42,
    )
    model.fit(
        X_train, y_train.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        max_epochs=params.get("max_epochs", 200),
        patience=params.get("patience", 20),
        batch_size=params.get("batch_size", 512),
        eval_metric=["mse"],
    )
    return model


def train_tabnet_classifier(X_train, y_train, X_val, y_val, params: dict):
    """Train TabNet classifier."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    model = TabNetClassifier(
        n_d=params.get("n_d", 64),
        n_a=params.get("n_a", 64),
        n_steps=params.get("n_steps", 5),
        gamma=params.get("gamma", 1.5),
        lambda_sparse=params.get("lambda_sparse", 0.001),
        seed=42,
    )
    model.fit(
        X_train, y_train.astype(int),
        eval_set=[(X_val, y_val.astype(int))],
        max_epochs=params.get("max_epochs", 200),
        patience=params.get("patience", 20),
        batch_size=params.get("batch_size", 512),
        eval_metric=["logloss"],
    )
    return model


def get_tabnet_attention(model, X) -> np.ndarray:
    """Extract per-sample attention masks from TabNet.

    Returns ndarray of shape (n_samples, n_features).
    """
    masks, _ = model.explain(X)
    return masks


def run_tabnet_experiment(
    X_train, y_train_ret, y_train_dir,
    X_val, y_val_ret, y_val_dir,
    X_test, y_test_ret, y_test_dir,
    params: dict,
    feature_set_name: str,
    horizon: int,
) -> dict:
    """Run TabNet regression + classification."""
    log.info("=== TabNet: %s, %d-day ===", feature_set_name, horizon)

    reg_model = train_tabnet_regressor(X_train, y_train_ret, X_val, y_val_ret, params)
    reg_pred = reg_model.predict(X_test).ravel()

    clf_model = train_tabnet_classifier(X_train, y_train_dir, X_val, y_val_dir, params)
    clf_pred = clf_model.predict(X_test).ravel()
    clf_proba = clf_model.predict_proba(X_test)[:, 1]

    # Attention masks
    reg_attention = get_tabnet_attention(reg_model, X_test)
    feature_importance = reg_model.feature_importances_

    # Save
    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reg_model.save_model(str(ckpt_dir / f"tabnet_{feature_set_name}_{horizon}d_reg"))
    clf_model.save_model(str(ckpt_dir / f"tabnet_{feature_set_name}_{horizon}d_clf"))

    return {
        "reg_model": reg_model,
        "clf_model": clf_model,
        "reg_pred": reg_pred,
        "clf_pred": clf_pred,
        "clf_proba": clf_proba,
        "attention_masks": reg_attention,
        "feature_importance": feature_importance,
    }
