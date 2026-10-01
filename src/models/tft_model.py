"""Temporal Fusion Transformer for sequential return prediction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils.logger import setup_logger
from src.utils.io_helpers import save_checkpoint, RESULTS_DIR

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset: sliding window over the feature panel
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Create (lookback_window, n_features) samples from a panel.

    IMPORTANT: This must be constructed per-ticker to avoid windows that
    span ticker boundaries.  Use ``build_multi_ticker_dataset`` to safely
    create a dataset from a multi-ticker panel.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int = 20):
        """
        Parameters
        ----------
        features : ndarray, shape (T, n_features) — sorted by time for ONE ticker.
        targets : ndarray, shape (T,)
        lookback : int
        """
        self.lookback = lookback
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.valid_indices = list(range(lookback, len(features)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x_seq = self.features[t - self.lookback: t]  # (lookback, n_features)
        y = self.targets[t]
        return torch.from_numpy(x_seq), torch.tensor(y)


class MultiTickerTimeSeriesDataset(Dataset):
    """Safely build sliding windows per-ticker, then concatenate.

    Prevents windows from spanning across different tickers.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        tickers: np.ndarray,
        lookback: int = 20,
    ):
        self.lookback = lookback
        self.sequences = []  # list of (x_seq, y) tuples
        self.global_indices = []  # row index in the original arrays

        global_idx = np.arange(len(features))
        for ticker in np.unique(tickers):
            mask = tickers == ticker
            feat_t = features[mask].astype(np.float32)
            tgt_t = targets[mask].astype(np.float32)
            gi = global_idx[mask]

            # Each ticker's data must already be sorted by date
            for t in range(lookback, len(feat_t)):
                self.sequences.append((feat_t[t - lookback: t], tgt_t[t]))
                self.global_indices.append(int(gi[t]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x_seq, y = self.sequences[idx]
        return torch.from_numpy(x_seq), torch.tensor(y)


# ---------------------------------------------------------------------------
# Simplified TFT architecture
# ---------------------------------------------------------------------------

class GatedResidualNetwork(nn.Module):
    """GRN from Lim et al. 2021."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None

    def forward(self, x, context=None):
        residual = self.skip(x)
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.elu(h)
        h = self.dropout(self.fc2(h))
        gate = torch.sigmoid(self.gate(h))
        return self.layer_norm(gate * h + residual)


class VariableSelectionNetwork(nn.Module):
    """Selects and weights input features."""

    def __init__(self, input_dim, n_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.per_feature_grn = nn.ModuleList([
            GatedResidualNetwork(input_dim // n_features, hidden_dim, hidden_dim, dropout)
            for _ in range(n_features)
        ])
        self.softmax_grn = GatedResidualNetwork(input_dim, hidden_dim, n_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, seq, input_dim) or (batch, input_dim)
        has_seq = x.dim() == 3

        # Variable selection weights
        if has_seq:
            flat = x.reshape(x.size(0) * x.size(1), -1)
        else:
            flat = x
        weights = self.softmax(self.softmax_grn(flat))  # (batch*seq, n_features)

        # Per-feature processing
        feat_size = x.size(-1) // self.n_features
        processed = []
        for i, grn in enumerate(self.per_feature_grn):
            if has_seq:
                fi = flat[:, i * feat_size: (i + 1) * feat_size]
            else:
                fi = flat[:, i * feat_size: (i + 1) * feat_size]
            processed.append(grn(fi))
        processed = torch.stack(processed, dim=-1)  # (batch*seq, hidden, n_features)

        # Weight and sum
        weights_expanded = weights.unsqueeze(1)  # (batch*seq, 1, n_features)
        combined = (processed * weights_expanded).sum(dim=-1)  # (batch*seq, hidden)

        if has_seq:
            combined = combined.reshape(x.size(0), x.size(1), -1)
            weights = weights.reshape(x.size(0), x.size(1), -1)

        return combined, weights


class SimpleTFT(nn.Module):
    """Simplified Temporal Fusion Transformer."""

    def __init__(
        self, n_features, hidden_size=64, attention_heads=4,
        dropout=0.1, lookback=20,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=2,
            batch_first=True, dropout=dropout,
        )

        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Output
        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, x_seq):
        """
        Parameters
        ----------
        x_seq : (batch, lookback, n_features)

        Returns
        -------
        prediction : (batch,)
        attn_weights : (batch, 1, lookback)
        """
        h = self.input_proj(x_seq)  # (batch, lookback, hidden)
        lstm_out, _ = self.lstm(h)  # (batch, lookback, hidden)

        # Self-attention over time
        # Use only last position as query
        query = lstm_out[:, -1:, :]  # (batch, 1, hidden)
        attn_out, attn_weights = self.attn(query, lstm_out, lstm_out)
        attn_out = self.attn_norm(query + attn_out)

        out = self.output_grn(attn_out.squeeze(1))
        pred = self.output_head(out).squeeze(-1)

        return pred, attn_weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_tft(
    features_train, targets_train,
    features_val, targets_val,
    params: dict,
    task: str = "regression",
    tickers_train: np.ndarray | None = None,
    tickers_val: np.ndarray | None = None,
) -> tuple[SimpleTFT, list[float]]:
    """Train a simplified TFT.

    Parameters
    ----------
    features_train/val : ndarray (N, n_features), sorted by (ticker, date).
    targets_train/val : ndarray (N,).
    tickers_train/val : ndarray (N,) of ticker strings.
        If provided, uses per-ticker windowing to prevent cross-ticker
        contamination.  If None, assumes single-ticker data.
    """
    device = _get_device()
    lookback = params.get("lookback_window", 20)
    n_features = features_train.shape[1]

    model = SimpleTFT(
        n_features=n_features,
        hidden_size=params.get("hidden_size", 64),
        attention_heads=params.get("attention_heads", 4),
        dropout=params.get("dropout", 0.1),
        lookback=lookback,
    ).to(device)

    # Use per-ticker windowing when tickers are provided
    if tickers_train is not None:
        train_ds = MultiTickerTimeSeriesDataset(features_train, targets_train, tickers_train, lookback)
        val_ds = MultiTickerTimeSeriesDataset(features_val, targets_val, tickers_val, lookback)
    else:
        train_ds = TimeSeriesDataset(features_train, targets_train, lookback)
        val_ds = TimeSeriesDataset(features_val, targets_val, lookback)

    batch_size = params.get("batch_size", 128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("learning_rate", 0.001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.get("max_epochs", 100)
    )

    patience = params.get("patience", 15)
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    val_losses = []

    for epoch in range(params.get("max_epochs", 100)):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)

        scheduler.step()

        if epoch % 10 == 0:
            log.info("TFT epoch %d: train=%.6f val=%.6f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info("TFT early stop at epoch %d", epoch)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return model, val_losses


def predict_tft(model, features, tickers=None, lookback=20):
    """Generate predictions for a time series.

    Parameters
    ----------
    features : ndarray (N, n_features), sorted by (ticker, date).
    tickers : ndarray (N,) of ticker strings, optional.
    """
    device = next(model.parameters()).device
    model.eval()

    if tickers is not None:
        ds = MultiTickerTimeSeriesDataset(features, np.zeros(len(features)), tickers, lookback)
    else:
        ds = TimeSeriesDataset(features, np.zeros(len(features)), lookback)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    preds, attn_list = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred, attn = model(xb)
            preds.append(pred.cpu().numpy())
            attn_list.append(attn.cpu().numpy())

    return np.concatenate(preds), np.concatenate(attn_list)


def run_tft_experiment(
    features_train, targets_train,
    features_val, targets_val,
    features_test, targets_test,
    params: dict,
    feature_set_name: str,
    horizon: int,
    tickers_train=None, tickers_val=None, tickers_test=None,
) -> dict:
    """Run TFT experiment.

    Pass ticker arrays to enable safe per-ticker windowing.
    """
    log.info("=== TFT: %s, %d-day ===", feature_set_name, horizon)

    model, _ = train_tft(
        features_train, targets_train,
        features_val, targets_val,
        params,
        tickers_train=tickers_train,
        tickers_val=tickers_val,
    )
    preds, attn_weights = predict_tft(
        model, features_test, tickers=tickers_test,
        lookback=params.get("lookback_window", 20),
    )

    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model.state_dict(), ckpt_dir / f"tft_{feature_set_name}_{horizon}d.pt")

    return {
        "model": model,
        "predictions": preds,
        "temporal_attention": attn_weights,
    }
