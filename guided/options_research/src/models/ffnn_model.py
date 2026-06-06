"""Feed-Forward Neural Network for return prediction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import setup_logger
from src.utils.io_helpers import save_checkpoint, RESULTS_DIR

log = setup_logger(__name__)


class FFNN(nn.Module):
    def __init__(self, input_dim, layers=(256, 128, 64, 32), dropout=0.3):
        super().__init__()
        modules = []
        prev_dim = input_dim
        for dim in layers:
            modules.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.backbone = nn.Sequential(*modules)
        self.reg_head = nn.Linear(prev_dim, 1)
        self.clf_head = nn.Linear(prev_dim, 1)

    def forward(self, x, task="regression"):
        h = self.backbone(x)
        if task == "regression":
            return self.reg_head(h).squeeze(-1)
        return self.clf_head(h).squeeze(-1)


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_ffnn(
    X_train, y_train, X_val, y_val,
    params: dict,
    task: str = "regression",
) -> tuple[FFNN, list[float]]:
    """Train FFNN with early stopping.

    Parameters
    ----------
    task : str
        'regression' or 'classification'.

    Returns
    -------
    model : trained FFNN.
    val_losses : list of per-epoch validation losses.
    """
    device = _get_device()
    log.info("Training FFNN (%s) on %s", task, device)

    layers = params.get("layers", [256, 128, 64, 32])
    if isinstance(layers, list):
        layers = tuple(layers)

    model = FFNN(
        input_dim=X_train.shape[1],
        layers=layers,
        dropout=params.get("dropout", 0.3),
    ).to(device)

    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.get("learning_rate", 0.001),
        weight_decay=params.get("weight_decay", 0.0001),
    )

    epochs = params.get("epochs", 200)
    patience = params.get("early_stopping_patience", 20)
    batch_size = params.get("batch_size", 512)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Data loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    val_losses = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, task=task)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, task=task)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)
        
        scheduler.step()

        if epoch % 20 == 0:
            log.info("Epoch %d: train=%.6f val=%.6f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return model, val_losses


def predict_ffnn(model: FFNN, X: np.ndarray, task: str = "regression") -> np.ndarray:
    """Generate predictions from a trained FFNN."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(xt, task=task).cpu().numpy()
    if task == "classification":
        return torch.sigmoid(torch.tensor(pred)).numpy()
    return pred


def run_ffnn_experiment(
    X_train, y_train_ret, y_train_dir,
    X_val, y_val_ret, y_val_dir,
    X_test, y_test_ret, y_test_dir,
    params: dict,
    feature_set_name: str,
    horizon: int,
) -> dict:
    """Run FFNN regression + classification."""
    log.info("=== FFNN: %s, %d-day ===", feature_set_name, horizon)

    reg_model, _ = train_ffnn(X_train, y_train_ret, X_val, y_val_ret, params, "regression")
    reg_pred = predict_ffnn(reg_model, X_test, "regression")

    clf_model, _ = train_ffnn(X_train, y_train_dir, X_val, y_val_dir, params, "classification")
    clf_proba = predict_ffnn(clf_model, X_test, "classification")
    clf_pred = (clf_proba > 0.5).astype(int)

    # Save checkpoints
    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(reg_model.state_dict(), ckpt_dir / f"ffnn_{feature_set_name}_{horizon}d_reg.pt")
    save_checkpoint(clf_model.state_dict(), ckpt_dir / f"ffnn_{feature_set_name}_{horizon}d_clf.pt")

    return {
        "reg_model": reg_model,
        "clf_model": clf_model,
        "reg_pred": reg_pred,
        "clf_pred": clf_pred,
        "clf_proba": clf_proba,
    }
