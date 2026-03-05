#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import QuantileLoss, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor


def get_feature_cols(df: pd.DataFrame, start_idx: int, end_idx: int) -> list[str]:
    cols: list[str] = []
    for i in range(start_idx, end_idx + 1):
        prefix = f"feat_{i:02d}_"
        cols.extend([c for c in df.columns if c.startswith(prefix)])
    return sorted(cols)


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.square(y_true)))
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum(np.square(y_pred - y_true)) / denom)


def simulate_long_flat(initial_capital: float, actual_returns: np.ndarray, pred_signal: np.ndarray) -> float:
    cap = float(initial_capital)
    for r, s in zip(actual_returns, pred_signal, strict=False):
        cap *= 1.0 + float(r) * (1.0 if float(s) > 0 else 0.0)
    return cap


def sector_of_ticker(ticker: str) -> str:
    tech = {"AAPL", "MSFT", "NVDA", "AVGO", "AMD", "PLTR", "QQQ"}
    comm = {"GOOGL", "META", "NFLX"}
    cons = {"AMZN", "TSLA"}
    fin = {"JPM"}
    broad = {"SPY"}
    if ticker in tech:
        return "tech"
    if ticker in comm:
        return "comm"
    if ticker in cons:
        return "cons"
    if ticker in fin:
        return "fin"
    if ticker in broad:
        return "broad"
    return "other"


class FFNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FFNNTrainOutput:
    model: FFNN
    scaler: StandardScaler


def train_ffnn_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
) -> FFNNTrainOutput:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    tr_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
    )
    va_x = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    va_y = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)
    tr_dl = DataLoader(tr_ds, batch_size=256, shuffle=True)

    model = FFNN(in_dim=X_train.shape[1], out_dim=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad = 0
    for _ in range(120):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(va_x), va_y).item())
        sched.step(val_loss)
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 10:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FFNNTrainOutput(model=model, scaler=scaler)


def train_ffnn_cls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
) -> FFNNTrainOutput:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    tr_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
    )
    va_x = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    va_y = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)
    tr_dl = DataLoader(tr_ds, batch_size=256, shuffle=True)

    model = FFNN(in_dim=X_train.shape[1], out_dim=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_val = float("inf")
    bad = 0
    for _ in range(120):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(va_x), va_y).item())
        sched.step(val_loss)
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 10:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FFNNTrainOutput(model=model, scaler=scaler)


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "oos_r2": float(oos_r2(y_true, y_pred)),
        "direction_accuracy_from_reg_sign": float(accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int))),
        "pnl_final_capital_long_flat_usd": float(simulate_long_flat(1000.0, y_true, (y_pred > 0).astype(int))),
        "buy_hold_final_capital_usd": float(simulate_long_flat(1000.0, y_true, np.ones_like(y_true))),
    }


def eval_classification(y_true: np.ndarray, prob_up: np.ndarray) -> dict[str, float]:
    pred = (prob_up >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, prob_up)),
    }


def run_xgb_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    reg_rows: list[dict] = []
    cls_rows: list[dict] = []
    y_test_reg = test_df["target_r_t_plus_1"].to_numpy(dtype=float)
    y_test_cls = test_df["target_direction_t_plus_1"].to_numpy(dtype=int)
    for set_name, cols in feature_sets.items():
        X_tr = train_df[cols].to_numpy(dtype=float)
        X_va = val_df[cols].to_numpy(dtype=float)
        X_te = test_df[cols].to_numpy(dtype=float)
        y_tr_reg = train_df["target_r_t_plus_1"].to_numpy(dtype=float)
        y_va_reg = val_df["target_r_t_plus_1"].to_numpy(dtype=float)
        y_tr_cls = train_df["target_direction_t_plus_1"].to_numpy(dtype=int)
        y_va_cls = val_df["target_direction_t_plus_1"].to_numpy(dtype=int)

        reg = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=6,
        )
        reg.fit(X_tr, y_tr_reg, eval_set=[(X_va, y_va_reg)], verbose=False)
        pred_reg = reg.predict(X_te)
        rr = eval_regression(y_test_reg, pred_reg)
        rr.update({"architecture": "XGBoost", "feature_set": set_name, "n_features": len(cols)})
        reg_rows.append(rr)

        clf = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=6,
        )
        clf.fit(X_tr, y_tr_cls, eval_set=[(X_va, y_va_cls)], verbose=False)
        prob = clf.predict_proba(X_te)[:, 1]
        cr = eval_classification(y_test_cls, prob)
        cr.update({"architecture": "XGBoost", "feature_set": set_name, "n_features": len(cols)})
        cls_rows.append(cr)

        if set_name == "Model_C_all_01_48":
            bg = shap.sample(pd.DataFrame(X_tr, columns=cols), 500, random_state=42)
            ex = shap.TreeExplainer(reg)
            sv = ex.shap_values(bg)
            if isinstance(sv, list):
                sv = sv[0]
            imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(np.asarray(sv)).mean(axis=0)}).sort_values(
                "mean_abs_shap", ascending=False
            )
            imp.to_csv(out_dir / "xgb_model_c_shap_importance.csv", index=False)
    return reg_rows, cls_rows


def run_ffnn_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    reg_rows: list[dict] = []
    cls_rows: list[dict] = []
    device = "cpu"
    y_test_reg = test_df["target_r_t_plus_1"].to_numpy(dtype=float)
    y_test_cls = test_df["target_direction_t_plus_1"].to_numpy(dtype=int)
    for set_name, cols in feature_sets.items():
        X_tr = train_df[cols].to_numpy(dtype=float)
        X_va = val_df[cols].to_numpy(dtype=float)
        X_te = test_df[cols].to_numpy(dtype=float)
        y_tr_reg = train_df["target_r_t_plus_1"].to_numpy(dtype=float)
        y_va_reg = val_df["target_r_t_plus_1"].to_numpy(dtype=float)
        y_tr_cls = train_df["target_direction_t_plus_1"].to_numpy(dtype=int)
        y_va_cls = val_df["target_direction_t_plus_1"].to_numpy(dtype=int)

        reg_obj = train_ffnn_reg(X_tr, y_tr_reg, X_va, y_va_reg, device=device)
        X_te_s = reg_obj.scaler.transform(X_te)
        with torch.no_grad():
            pr = (
                reg_obj.model(torch.tensor(X_te_s, dtype=torch.float32, device=device))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        rr = eval_regression(y_test_reg, pr)
        rr.update({"architecture": "FFNN", "feature_set": set_name, "n_features": len(cols)})
        reg_rows.append(rr)

        cls_obj = train_ffnn_cls(X_tr, y_tr_cls.astype(float), X_va, y_va_cls.astype(float), device=device)
        X_te_s2 = cls_obj.scaler.transform(X_te)
        with torch.no_grad():
            prob = torch.sigmoid(
                cls_obj.model(torch.tensor(X_te_s2, dtype=torch.float32, device=device))
            ).detach().cpu().numpy().reshape(-1)
        cr = eval_classification(y_test_cls, prob)
        cr.update({"architecture": "FFNN", "feature_set": set_name, "n_features": len(cols)})
        cls_rows.append(cr)

        if set_name == "Model_C_all_01_48":
            bg_idx = np.random.default_rng(42).choice(len(X_tr), size=min(300, len(X_tr)), replace=False)
            te_idx = np.random.default_rng(43).choice(len(X_te), size=min(200, len(X_te)), replace=False)
            bg = torch.tensor(reg_obj.scaler.transform(X_tr[bg_idx]), dtype=torch.float32)
            ex = shap.GradientExplainer(reg_obj.model, bg)
            sv = ex.shap_values(torch.tensor(reg_obj.scaler.transform(X_te[te_idx]), dtype=torch.float32))
            if isinstance(sv, list):
                sv = sv[0]
            sv_arr = np.asarray(sv)
            # Handle possible shapes: (n, d), (n, d, 1), or (1, n, d)
            if sv_arr.ndim == 3 and sv_arr.shape[-1] == 1:
                sv_arr = sv_arr[..., 0]
            if sv_arr.ndim == 3 and sv_arr.shape[0] == 1:
                sv_arr = sv_arr[0]
            if sv_arr.ndim != 2:
                sv_arr = sv_arr.reshape(sv_arr.shape[0], -1)
            imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv_arr).mean(axis=0)}).sort_values(
                "mean_abs_shap", ascending=False
            )
            imp.to_csv(out_dir / "ffnn_model_c_shap_importance.csv", index=False)
    return reg_rows, cls_rows


def run_tabnet_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    reg_rows: list[dict] = []
    cls_rows: list[dict] = []
    y_test_reg = test_df["target_r_t_plus_1"].to_numpy(dtype=float)
    y_test_cls = test_df["target_direction_t_plus_1"].to_numpy(dtype=int)
    for set_name, cols in feature_sets.items():
        X_tr = train_df[cols].to_numpy(dtype=np.float32)
        X_va = val_df[cols].to_numpy(dtype=np.float32)
        X_te = test_df[cols].to_numpy(dtype=np.float32)
        y_tr_reg = train_df["target_r_t_plus_1"].to_numpy(dtype=np.float32).reshape(-1, 1)
        y_va_reg = val_df["target_r_t_plus_1"].to_numpy(dtype=np.float32).reshape(-1, 1)
        y_tr_cls = train_df["target_direction_t_plus_1"].to_numpy(dtype=np.int64)
        y_va_cls = val_df["target_direction_t_plus_1"].to_numpy(dtype=np.int64)

        reg = TabNetRegressor(seed=42, verbose=0)
        reg.fit(
            X_tr,
            y_tr_reg,
            eval_set=[(X_va, y_va_reg)],
            eval_name=["val"],
            eval_metric=["rmse"],
            max_epochs=120,
            patience=12,
            batch_size=2048,
            virtual_batch_size=256,
        )
        pred_reg = reg.predict(X_te).reshape(-1)
        rr = eval_regression(y_test_reg, pred_reg)
        rr.update({"architecture": "TabNet", "feature_set": set_name, "n_features": len(cols)})
        reg_rows.append(rr)

        clf = TabNetClassifier(seed=42, verbose=0)
        clf.fit(
            X_tr,
            y_tr_cls,
            eval_set=[(X_va, y_va_cls)],
            eval_name=["val"],
            eval_metric=["auc"],
            max_epochs=120,
            patience=12,
            batch_size=2048,
            virtual_batch_size=256,
        )
        prob = clf.predict_proba(X_te)[:, 1]
        cr = eval_classification(y_test_cls, prob)
        cr.update({"architecture": "TabNet", "feature_set": set_name, "n_features": len(cols)})
        cls_rows.append(cr)

        if set_name == "Model_C_all_01_48":
            masks = clf.explain(X_te[:3000])[0]
            imp = pd.DataFrame({"feature": cols, "mean_mask": np.mean(masks, axis=0)}).sort_values("mean_mask", ascending=False)
            imp.to_csv(out_dir / "tabnet_model_c_attention_importance.csv", index=False)
    return reg_rows, cls_rows


def run_tft_model_c(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_c_cols: list[str],
    out_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True).copy()
    all_df["date"] = pd.to_datetime(all_df["date"], utc=True)
    all_df = all_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    # Keep a modern window for tractable TFT runtime on CPU.
    all_df = all_df[all_df["date"] >= pd.Timestamp("2012-01-01", tz="UTC")].copy()
    all_df["sector"] = all_df["ticker"].map(sector_of_ticker)
    all_df["time_idx"] = all_df.groupby("ticker").cumcount()
    all_df["target"] = all_df["target_r_t_plus_1"].astype(float)

    train_cut = train_df["date"].max()
    val_cut = val_df["date"].max()
    known_reals = [
        "feat_43_days_to_next_earnings",
        "feat_44_earnings_within_7d_flag",
        "feat_45_days_to_ex_dividend",
        "feat_46_day_of_week",
        "feat_47_days_to_monthly_opex",
        "feat_48_quarter_end_flag",
    ]
    unknown_reals = [c for c in model_c_cols if c not in known_reals]
    training = TimeSeriesDataSet(
        all_df[all_df["date"] <= train_cut],
        time_idx="time_idx",
        target="target",
        group_ids=["ticker"],
        max_encoder_length=20,
        max_prediction_length=1,
        static_categoricals=["ticker", "sector"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals + ["target"],
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        categorical_encoders={"ticker": NaNLabelEncoder(add_nan=True), "sector": NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, all_df[all_df["date"] <= val_cut], stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, all_df, stop_randomization=True)

    train_dl = training.to_dataloader(train=True, batch_size=256, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=256, num_workers=0)
    test_dl = testing.to_dataloader(train=False, batch_size=256, num_workers=0)

    early = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=3,
        loss=QuantileLoss([0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=3,
    )
    trainer = Trainer(
        max_epochs=8,
        accelerator="cpu",
        devices=1,
        callbacks=[early],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
    )
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    pred = tft.predict(test_dl).detach().cpu().numpy()
    # Median quantile if present, otherwise fallback to single output.
    if pred.ndim == 2 and pred.shape[1] > 1:
        p50 = pred[:, 1]
    elif pred.ndim == 2 and pred.shape[1] == 1:
        p50 = pred[:, 0]
    else:
        p50 = pred.reshape(-1)
    y_true = all_df["target"].to_numpy(dtype=float)[-len(p50) :]
    rr = eval_regression(y_true, p50)
    rr.update({"architecture": "TFT", "feature_set": "Model_C_all_01_48", "n_features": len(model_c_cols)})
    rows.append(rr)

    # Variable selection + attention summaries (best effort across library versions).
    try:
        raw_out = tft.predict(test_dl, mode="raw", return_x=True)
        raw_preds = raw_out[0] if isinstance(raw_out, tuple) else raw_out
        interp = tft.interpret_output(raw_preds, reduction="mean")
        if "encoder_variables" in interp:
            enc_var = interp["encoder_variables"]
            enc_var = enc_var.detach().cpu().numpy() if hasattr(enc_var, "detach") else np.asarray(enc_var)
            pd.DataFrame({"feature": unknown_reals + known_reals + ["target"], "mean_weight": enc_var}).sort_values(
                "mean_weight", ascending=False
            ).to_csv(out_dir / "tft_variable_selection_importance.csv", index=False)
        if "attention" in interp:
            att = interp["attention"]
            att = att.detach().cpu().numpy() if hasattr(att, "detach") else np.asarray(att)
            pd.DataFrame({"lag_idx": list(range(len(att))), "mean_attention": att}).to_csv(
                out_dir / "tft_attention_by_lag.csv", index=False
            )
    except Exception:
        pass
    return rows


def run_benchmarks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_cols: list[str],
) -> list[dict]:
    rows: list[dict] = []
    y_tr = train_df["target_r_t_plus_1"].to_numpy(dtype=float)
    y_te = test_df["target_r_t_plus_1"].to_numpy(dtype=float)
    zero = np.zeros_like(y_te)
    rows.append({"benchmark": "naive_zero", **eval_regression(y_te, zero)})

    mean_pred = np.full_like(y_te, fill_value=float(np.mean(y_tr)))
    rows.append({"benchmark": "historical_mean", **eval_regression(y_te, mean_pred)})

    lr_full = LinearRegression()
    lr_full.fit(train_df[all_cols].to_numpy(dtype=float), y_tr)
    p_full = lr_full.predict(test_df[all_cols].to_numpy(dtype=float))
    rows.append({"benchmark": "ols_full", **eval_regression(y_te, p_full)})

    bs_cols = [c for c in all_cols if c.startswith(("feat_04_", "feat_05_", "feat_06_", "feat_10_", "feat_12_"))]
    if len(bs_cols) > 0:
        lr_bs = LinearRegression()
        lr_bs.fit(train_df[bs_cols].to_numpy(dtype=float), y_tr)
        p_bs = lr_bs.predict(test_df[bs_cols].to_numpy(dtype=float))
        rows.append({"benchmark": "ols_bs_inputs_only", **eval_regression(y_te, p_bs)})

    for feat_name, label in [
        ("feat_01_put_call_volume_ratio", "single_put_call_ratio"),
        ("feat_10_atm_iv_change_1d", "single_iv_change"),
        ("feat_18_volume_weighted_avg_spread", "single_parity_deviation_proxy"),
    ]:
        if feat_name in train_df.columns:
            lr = LinearRegression()
            xtr = train_df[[feat_name]].to_numpy(dtype=float)
            xte = test_df[[feat_name]].to_numpy(dtype=float)
            lr.fit(xtr, y_tr)
            pred = lr.predict(xte)
            rows.append({"benchmark": label, **eval_regression(y_te, pred)})
    return rows


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    in_dir = root / "data/processed/panel/training_ready"
    out_dir = root / "results/panel_model_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(in_dir / "panel_train_no_sentiment.parquet")
    val_df = pd.read_parquet(in_dir / "panel_val_no_sentiment.parquet")
    test_df = pd.read_parquet(in_dir / "panel_test_no_sentiment.parquet")
    for d in (train_df, val_df, test_df):
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")

    feature_sets = {
        "Model_A_stock_only_27_42": get_feature_cols(train_df, 27, 42),
        "Model_B_options_only_01_26": get_feature_cols(train_df, 1, 26),
        "Model_C_all_01_48": get_feature_cols(train_df, 1, 48),
    }

    reg_rows: list[dict] = []
    cls_rows: list[dict] = []
    reg_xgb, cls_xgb = run_xgb_suite(train_df, val_df, test_df, feature_sets, out_dir)
    reg_rows.extend(reg_xgb)
    cls_rows.extend(cls_xgb)
    reg_ffnn, cls_ffnn = run_ffnn_suite(train_df, val_df, test_df, feature_sets, out_dir)
    reg_rows.extend(reg_ffnn)
    cls_rows.extend(cls_ffnn)
    reg_tab, cls_tab = run_tabnet_suite(train_df, val_df, test_df, feature_sets, out_dir)
    reg_rows.extend(reg_tab)
    cls_rows.extend(cls_tab)
    tft_error = None
    try:
        reg_tft = run_tft_model_c(train_df, val_df, test_df, feature_sets["Model_C_all_01_48"], out_dir)
        reg_rows.extend(reg_tft)
    except Exception as exc:
        tft_error = str(exc)

    benchmarks = run_benchmarks(train_df, test_df, feature_sets["Model_C_all_01_48"])

    reg_df = pd.DataFrame(reg_rows).sort_values(["feature_set", "architecture"]).reset_index(drop=True)
    cls_df = pd.DataFrame(cls_rows).sort_values(["feature_set", "architecture"]).reset_index(drop=True)
    bench_df = pd.DataFrame(benchmarks)

    reg_df.to_csv(out_dir / "regression_summary.csv", index=False)
    cls_df.to_csv(out_dir / "classification_summary.csv", index=False)
    bench_df.to_csv(out_dir / "benchmark_summary.csv", index=False)

    meta = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "outputs": {
            "regression_summary": str(out_dir / "regression_summary.csv"),
            "classification_summary": str(out_dir / "classification_summary.csv"),
            "benchmark_summary": str(out_dir / "benchmark_summary.csv"),
            "xgb_shap": str(out_dir / "xgb_model_c_shap_importance.csv"),
            "ffnn_shap": str(out_dir / "ffnn_model_c_shap_importance.csv"),
            "tabnet_attention": str(out_dir / "tabnet_model_c_attention_importance.csv"),
            "tft_variable_selection": str(out_dir / "tft_variable_selection_importance.csv"),
            "tft_attention": str(out_dir / "tft_attention_by_lag.csv"),
        },
        "tft_error": tft_error,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("=== Regression ===")
    print(reg_df.to_string(index=False))
    print("\n=== Classification ===")
    print(cls_df.to_string(index=False))
    print("\n=== Benchmarks ===")
    print(bench_df.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
