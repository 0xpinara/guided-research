"""Set Transformer for variable-size option contract sets (Resolution 3)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils.logger import setup_logger
from src.utils.io_helpers import save_checkpoint, RESULTS_DIR

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Architecture components
# ---------------------------------------------------------------------------

class MultiheadAttentionBlock(nn.Module):
    """Multi-head attention + LayerNorm + FFN."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, key_mask=None):
        key_padding_mask = ~key_mask if key_mask is not None else None
        attn_out, attn_weights = self.attn(
            query, key, value, key_padding_mask=key_padding_mask,
        )
        h = self.norm1(query + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h, attn_weights


class ISAB(nn.Module):
    """Induced Set Attention Block — O(n*m) complexity."""

    def __init__(self, embed_dim, num_heads, num_induced, dropout=0.1):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_induced, embed_dim))
        self.mab1 = MultiheadAttentionBlock(embed_dim, num_heads, dropout)
        self.mab2 = MultiheadAttentionBlock(embed_dim, num_heads, dropout)

    def forward(self, x, mask=None):
        inducing = self.inducing_points.expand(x.size(0), -1, -1)
        h, _ = self.mab1(inducing, x, x, key_mask=mask)
        out, _ = self.mab2(x, h, h)
        return out


class PMA(nn.Module):
    """Pooling by Multi-head Attention."""

    def __init__(self, embed_dim, num_heads, num_seeds=1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, embed_dim))
        self.mab = MultiheadAttentionBlock(embed_dim, num_heads, 0.0)

    def forward(self, x, mask=None):
        seeds = self.seeds.expand(x.size(0), -1, -1)
        out, attn_weights = self.mab(seeds, x, x, key_mask=mask)
        return out, attn_weights


class SetTransformerEncoder(nn.Module):
    """Encode a variable-size set of contracts into a fixed-size embedding."""

    def __init__(self, in_dim=10, embed_dim=64, num_heads=4,
                 num_induced=32, num_blocks=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.isab_blocks = nn.ModuleList([
            ISAB(embed_dim, num_heads, num_induced, dropout)
            for _ in range(num_blocks)
        ])
        self.pool = PMA(embed_dim, num_heads, num_seeds=1)

    def forward(self, contracts, mask):
        h = self.input_proj(contracts)
        for isab in self.isab_blocks:
            h = isab(h, mask)
        pooled, attn_weights = self.pool(h, mask)
        return pooled.squeeze(1), attn_weights  # (batch, embed_dim), (batch, 1, max_contracts)


class OptionsReturnPredictor(nn.Module):
    """Full model: Set Transformer on contracts + stock features -> prediction."""

    def __init__(self, contract_in_dim=10, stock_feature_dim=22,
                 embed_dim=64, num_heads=4, num_induced=32,
                 num_encoder_blocks=2, dropout=0.1):
        super().__init__()
        self.set_encoder = SetTransformerEncoder(
            in_dim=contract_in_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_induced=num_induced,
            num_blocks=num_encoder_blocks,
            dropout=dropout,
        )
        combined_dim = embed_dim + stock_feature_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, contracts, mask, stock_features):
        options_emb, attn_weights = self.set_encoder(contracts, mask)
        combined = torch.cat([options_emb, stock_features], dim=-1)
        pred = self.prediction_head(combined).squeeze(-1)
        return pred, attn_weights


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContractDataset(Dataset):
    def __init__(self, tensors, masks, stock_features, targets):
        self.tensors = torch.tensor(tensors, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.stock_features = torch.tensor(stock_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.tensors[idx], self.masks[idx],
                self.stock_features[idx], self.targets[idx])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_set_transformer(
    tensors_train, masks_train, stock_train, targets_train,
    tensors_val, masks_val, stock_val, targets_val,
    params: dict,
    task: str = "regression",
) -> tuple[OptionsReturnPredictor, list[float]]:
    """Train the Set Transformer model."""
    device = _get_device()
    log.info("Training Set Transformer on %s", device)

    contract_dim = tensors_train.shape[2]
    stock_dim = stock_train.shape[1]

    model = OptionsReturnPredictor(
        contract_in_dim=contract_dim,
        stock_feature_dim=stock_dim,
        embed_dim=params.get("embed_dim", 64),
        num_heads=params.get("num_heads", 4),
        num_induced=params.get("num_induced_points", 32),
        num_encoder_blocks=params.get("num_encoder_blocks", 2),
        dropout=params.get("dropout", 0.1),
    ).to(device)

    train_ds = ContractDataset(tensors_train, masks_train, stock_train, targets_train)
    val_ds = ContractDataset(tensors_val, masks_val, stock_val, targets_val)

    batch_size = params.get("batch_size", 128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("learning_rate", 0.001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.get("max_epochs", 100)
    )

    patience = params.get("patience", 15)
    best_val = float("inf")
    best_state = None
    wait = 0
    val_losses = []

    for epoch in range(params.get("max_epochs", 100)):
        model.train()
        tloss = 0.0
        for ct, mk, sf, tgt in train_loader:
            ct, mk, sf, tgt = ct.to(device), mk.to(device), sf.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred, _ = model(ct, mk, sf)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            tloss += loss.item() * len(tgt)
        tloss /= len(train_ds)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for ct, mk, sf, tgt in val_loader:
                ct, mk, sf, tgt = ct.to(device), mk.to(device), sf.to(device), tgt.to(device)
                pred, _ = model(ct, mk, sf)
                vloss += criterion(pred, tgt).item() * len(tgt)
        vloss /= len(val_ds)
        val_losses.append(vloss)

        scheduler.step()

        if epoch % 10 == 0:
            log.info("SetTransformer epoch %d: train=%.6f val=%.6f", epoch, tloss, vloss)

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info("SetTransformer early stop at epoch %d", epoch)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return model, val_losses


def predict_set_transformer(model, tensors, masks, stock_features):
    """Generate predictions and attention weights."""
    device = next(model.parameters()).device
    model.eval()
    ds = ContractDataset(tensors, masks, stock_features, np.zeros(len(tensors)))
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    preds, attns = [], []
    with torch.no_grad():
        for ct, mk, sf, _ in loader:
            ct, mk, sf = ct.to(device), mk.to(device), sf.to(device)
            pred, attn = model(ct, mk, sf)
            preds.append(pred.cpu().numpy())
            attns.append(attn.cpu().numpy())

    return np.concatenate(preds), np.concatenate(attns)


def run_set_transformer_experiment(
    tensors_train, masks_train, stock_train, targets_train,
    tensors_val, masks_val, stock_val, targets_val,
    tensors_test, masks_test, stock_test, targets_test,
    params: dict,
    feature_set_name: str,
    horizon: int,
) -> dict:
    """Run Set Transformer experiment."""
    log.info("=== Set Transformer: %s, %d-day ===", feature_set_name, horizon)

    model, _ = train_set_transformer(
        tensors_train, masks_train, stock_train, targets_train,
        tensors_val, masks_val, stock_val, targets_val,
        params,
    )
    preds, attn_weights = predict_set_transformer(model, tensors_test, masks_test, stock_test)

    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model.state_dict(), ckpt_dir / f"set_transformer_{feature_set_name}_{horizon}d.pt")

    return {
        "model": model,
        "predictions": preds,
        "contract_attention": attn_weights,
    }
