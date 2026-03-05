#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import norm
from xgboost import XGBClassifier, XGBRegressor


def get_feature_cols(df: pd.DataFrame, start_idx: int, end_idx: int) -> list[str]:
    cols: list[str] = []
    for i in range(start_idx, end_idx + 1):
        prefix = f"feat_{i:02d}_"
        cols.extend([c for c in df.columns if c.startswith(prefix)])
    return sorted(cols)


def simulate_long_flat(initial_capital: float, actual_returns: np.ndarray, pred_signal: np.ndarray) -> float:
    cap = float(initial_capital)
    for r, s in zip(actual_returns, pred_signal, strict=False):
        cap *= 1.0 + float(r) * (1.0 if int(s) == 1 else 0.0)
    return cap


def choose_threshold_for_pnl(prob: np.ndarray, returns: np.ndarray) -> float:
    best_t = 0.5
    best_cap = -np.inf
    for t in np.linspace(0.2, 0.8, 61):
        sig = (prob >= t).astype(int)
        cap = simulate_long_flat(1000.0, returns, sig)
        if cap > best_cap:
            best_cap = cap
            best_t = float(t)
    return best_t


def newey_west_var(x: np.ndarray, lag: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    if n <= 1:
        return np.nan
    gamma0 = np.nanmean(x * x)
    v = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        cov = np.nanmean(x[l:] * x[:-l])
        v += 2.0 * w * cov
    return float(v)


def dm_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, lag: int = 5) -> dict[str, float]:
    e1 = y - p1
    e2 = y - p2
    d = e1 * e1 - e2 * e2
    n = len(d)
    md = float(np.mean(d))
    v = newey_west_var(d, lag=lag)
    stat = md / np.sqrt(v / n) if np.isfinite(v) and v > 0 else np.nan
    p = 2.0 * (1.0 - norm.cdf(abs(stat))) if np.isfinite(stat) else np.nan
    return {"dm_stat": float(stat), "dm_pvalue": float(p)}


def clark_west_test(y: np.ndarray, p_small: np.ndarray, p_large: np.ndarray, lag: int = 5) -> dict[str, float]:
    e_small = y - p_small
    e_large = y - p_large
    f = e_small * e_small - (e_large * e_large - (p_small - p_large) ** 2)
    n = len(f)
    mf = float(np.mean(f))
    v = newey_west_var(f, lag=lag)
    stat = mf / np.sqrt(v / n) if np.isfinite(v) and v > 0 else np.nan
    p_one_sided = (1.0 - norm.cdf(stat)) if np.isfinite(stat) else np.nan
    return {"cw_stat": float(stat), "cw_pvalue_one_sided": float(p_one_sided)}


def rolling_windows(df: pd.DataFrame, train_len: int = 504, val_len: int = 63, test_len: int = 63, step: int = 63):
    dates = np.array(sorted(df["date"].dropna().unique()))
    i = 0
    while i + train_len + val_len + test_len <= len(dates):
        tr = dates[i : i + train_len]
        va = dates[i + train_len : i + train_len + val_len]
        te = dates[i + train_len + val_len : i + train_len + val_len + test_len]
        yield set(tr), set(va), set(te)
        i += step


def fit_xgb_reg(X: np.ndarray, y: np.ndarray) -> XGBRegressor:
    m = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=900,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=6,
    )
    m.fit(X, y, verbose=False)
    return m


def fit_xgb_cls(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    m = XGBClassifier(
        objective="binary:logistic",
        n_estimators=900,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=6,
    )
    m.fit(X, y, verbose=False)
    return m


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    in_dir = root / "data/processed/panel/training_ready"
    out_dir = root / "results/panel_future_plans"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(in_dir / "panel_train_no_sentiment.parquet")
    val_df = pd.read_parquet(in_dir / "panel_val_no_sentiment.parquet")
    test_df = pd.read_parquet(in_dir / "panel_test_no_sentiment.parquet")
    clean_df = pd.read_parquet(in_dir / "panel_model_dataset_no_sentiment_clean.parquet")
    for d in (train_df, val_df, test_df, clean_df):
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")

    target_r = "target_r_t_plus_1"
    target_d = "target_direction_t_plus_1"
    feature_sets = {
        "A_stock": get_feature_cols(train_df, 27, 42),
        "B_options": get_feature_cols(train_df, 1, 26),
        "C_all": get_feature_cols(train_df, 1, 48),
    }

    # 1) Rolling-window evaluation (Priority: temporal robustness)
    print("Running rolling-window evaluation ...")
    roll_rows: list[dict] = []
    for w_idx, (tr_dates, va_dates, te_dates) in enumerate(rolling_windows(clean_df), start=1):
        if w_idx > 20:
            break
        tr = clean_df[clean_df["date"].isin(tr_dates)].copy()
        va = clean_df[clean_df["date"].isin(va_dates)].copy()
        te = clean_df[clean_df["date"].isin(te_dates)].copy()
        if len(tr) < 1000 or len(te) < 200:
            continue
        y_te = te[target_r].to_numpy(dtype=float)
        for set_name, cols in feature_sets.items():
            reg = fit_xgb_reg(tr[cols].to_numpy(dtype=float), tr[target_r].to_numpy(dtype=float))
            pred = reg.predict(te[cols].to_numpy(dtype=float))
            cls = fit_xgb_cls(tr[cols].to_numpy(dtype=float), tr[target_d].to_numpy(dtype=int))
            p_val = cls.predict_proba(va[cols].to_numpy(dtype=float))[:, 1]
            t_star = choose_threshold_for_pnl(p_val, va[target_r].to_numpy(dtype=float))
            p_te = cls.predict_proba(te[cols].to_numpy(dtype=float))[:, 1]
            sig = (p_te >= t_star).astype(int)
            roll_rows.append(
                {
                    "window": w_idx,
                    "feature_set": set_name,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "rmse": float(np.sqrt(np.mean((pred - y_te) ** 2))),
                    "auc": float(np.nan if len(np.unique(te[target_d])) < 2 else __import__("sklearn.metrics").metrics.roc_auc_score(te[target_d], p_te)),
                    "pnl_capital": float(simulate_long_flat(1000.0, y_te, sig)),
                }
            )
    roll_df = pd.DataFrame(roll_rows)
    roll_df.to_csv(out_dir / "rolling_window_panel_xgb_abc.csv", index=False)
    if len(roll_df):
        roll_df.groupby("feature_set", as_index=False).agg({"rmse": ["mean", "std"], "auc": ["mean", "std"], "pnl_capital": ["median", "mean"]}).to_csv(
            out_dir / "rolling_window_panel_xgb_abc_summary.csv", index=False
        )

    print("Fitting final A/B/C XGBoost models ...")
    # Fit final XGB models on train split for downstream analyses.
    preds_reg: dict[str, np.ndarray] = {}
    preds_cls_prob: dict[str, np.ndarray] = {}
    xgb_models: dict[str, XGBRegressor] = {}
    for set_name, cols in feature_sets.items():
        reg = fit_xgb_reg(train_df[cols].to_numpy(dtype=float), train_df[target_r].to_numpy(dtype=float))
        xgb_models[set_name] = reg
        preds_reg[set_name] = reg.predict(test_df[cols].to_numpy(dtype=float))
        cls = fit_xgb_cls(train_df[cols].to_numpy(dtype=float), train_df[target_d].to_numpy(dtype=int))
        preds_cls_prob[set_name] = cls.predict_proba(test_df[cols].to_numpy(dtype=float))[:, 1]

    y_test = test_df[target_r].to_numpy(dtype=float)

    print("Running DM/CW statistical tests ...")
    # 2) Statistical significance tests (DM + CW) and single-signal comparisons done in prior suite.
    tests = []
    tests.append({"comparison": "A_vs_B", **dm_test(y_test, preds_reg["A_stock"], preds_reg["B_options"])})
    tests.append({"comparison": "A_vs_C", **dm_test(y_test, preds_reg["A_stock"], preds_reg["C_all"])})
    tests.append({"comparison": "B_vs_C", **dm_test(y_test, preds_reg["B_options"], preds_reg["C_all"])})
    tests.append({"comparison": "CW_A_nested_in_C", **clark_west_test(y_test, preds_reg["A_stock"], preds_reg["C_all"])})
    pd.DataFrame(tests).to_csv(out_dir / "stat_tests_dm_cw.csv", index=False)

    print("Running SHAP analyses (global/dependence/interaction/local/bootstrap) ...")
    # 3) SHAP global/dependence/interaction/local and bootstrap rank CI (Model C, XGB)
    cols_c = feature_sets["C_all"]
    model_c = xgb_models["C_all"]
    X_test_c = test_df[cols_c].to_numpy(dtype=float)
    ex = shap.TreeExplainer(model_c)
    sv = ex.shap_values(X_test_c)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    global_shap = pd.DataFrame({"feature": cols_c, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    global_shap.to_csv(out_dir / "xgb_model_c_global_shap.csv", index=False)

    # Dependence stats for top features: Spearman correlation(feature, shap(feature))
    dep_rows = []
    top_features = global_shap["feature"].head(15).tolist()
    for f in top_features:
        j = cols_c.index(f)
        x = test_df[f].to_numpy(dtype=float)
        y = sv[:, j]
        rank_x = pd.Series(x).rank().to_numpy()
        rank_y = pd.Series(y).rank().to_numpy()
        corr = float(np.corrcoef(rank_x, rank_y)[0, 1]) if np.isfinite(rank_x).all() and np.isfinite(rank_y).all() else np.nan
        dep_rows.append({"feature": f, "spearman_like_corr_feature_vs_shap": corr})
    pd.DataFrame(dep_rows).to_csv(out_dir / "xgb_model_c_shap_dependence_stats.csv", index=False)

    # Interaction values (subset for tractability)
    idx_int = np.random.default_rng(42).choice(len(X_test_c), size=min(300, len(X_test_c)), replace=False)
    inter = ex.shap_interaction_values(X_test_c[idx_int])
    if isinstance(inter, list):
        inter = inter[0]
    inter = np.asarray(inter)
    pair_rows = []
    d = inter.shape[1]
    for i in range(d):
        for j in range(i + 1, d):
            pair_rows.append((cols_c[i], cols_c[j], float(np.mean(np.abs(inter[:, i, j])))))
    pd.DataFrame(pair_rows, columns=["feature_i", "feature_j", "mean_abs_interaction"]).sort_values(
        "mean_abs_interaction", ascending=False
    ).head(200).to_csv(out_dir / "xgb_model_c_top_shap_interactions.csv", index=False)

    # Local explanations for worst residual points
    resid = np.abs(y_test - preds_reg["C_all"])
    top_idx = np.argsort(-resid)[:25]
    local_rows = []
    for k in top_idx:
        row = {"row_idx": int(k), "date": str(test_df.iloc[k]["date"]), "ticker": str(test_df.iloc[k]["ticker"]), "abs_residual": float(resid[k])}
        cabs = np.abs(sv[k])
        topj = np.argsort(-cabs)[:5]
        for n, j in enumerate(topj, start=1):
            row[f"top{n}_feature"] = cols_c[j]
            row[f"top{n}_shap"] = float(sv[k, j])
            row[f"top{n}_value"] = float(test_df.iloc[k][cols_c[j]])
        local_rows.append(row)
    pd.DataFrame(local_rows).to_csv(out_dir / "xgb_model_c_local_explanations_top_residuals.csv", index=False)

    # Bootstrap CI for SHAP rank
    rng = np.random.default_rng(123)
    n = len(sv)
    ranks = np.zeros((100, len(cols_c)), dtype=float)
    for b in range(100):
        idx = rng.integers(0, n, size=n)
        s = np.abs(sv[idx]).mean(axis=0)
        order = np.argsort(-s)
        rk = np.empty_like(order, dtype=float)
        rk[order] = np.arange(1, len(order) + 1)
        ranks[b] = rk
    ci_df = pd.DataFrame(
        {
            "feature": cols_c,
            "rank_mean": ranks.mean(axis=0),
            "rank_p2_5": np.quantile(ranks, 0.025, axis=0),
            "rank_p97_5": np.quantile(ranks, 0.975, axis=0),
        }
    ).sort_values("rank_mean")
    ci_df.to_csv(out_dir / "xgb_model_c_shap_rank_bootstrap_ci.csv", index=False)

    print("Running regime-conditional analyses ...")
    # 4) Regime-conditional analysis with 5 splits (using Model C SHAP)
    td = test_df.copy().reset_index(drop=True)
    td["_pred_c"] = preds_reg["C_all"]
    td["_sig_c"] = (preds_cls_prob["C_all"] >= 0.5).astype(int)
    td["_abs_ret_1d"] = td[target_r].abs()
    # proxy 20d trend from rolling sum of 1d returns by ticker
    td = td.sort_values(["ticker", "date"]).reset_index(drop=True)
    td["_trend20_proxy"] = td.groupby("ticker")[target_r].rolling(20, min_periods=10).sum().reset_index(level=0, drop=True)

    regimes = {
        "vol_high_vs_low": ((td["feat_38_vix_close"] > 25), (td["feat_38_vix_close"] < 15)),
        "earnings_within7_vs_normal": ((td["feat_44_earnings_within_7d_flag"] == 1), (td["feat_44_earnings_within_7d_flag"] == 0)),
        "options_liquidity_high_vs_low": (
            td["feat_03_options_to_stock_volume_ratio"] >= td["feat_03_options_to_stock_volume_ratio"].quantile(0.75),
            td["feat_03_options_to_stock_volume_ratio"] <= td["feat_03_options_to_stock_volume_ratio"].quantile(0.25),
        ),
        "market_trend_bull_vs_bear": ((td["_trend20_proxy"] > 0), (td["_trend20_proxy"] < 0)),
        "dealer_gex_pos_vs_neg": ((td["feat_21_estimated_dealer_gamma_gex"] > 0), (td["feat_21_estimated_dealer_gamma_gex"] < 0)),
    }
    reg_rows = []
    for name, (ma, mb) in regimes.items():
        ia = np.where(ma.to_numpy())[0]
        ib = np.where(mb.to_numpy())[0]
        if len(ia) > 20 and len(ib) > 20:
            sha = np.abs(sv[ia]).mean(axis=0)
            shb = np.abs(sv[ib]).mean(axis=0)
            dfm = pd.DataFrame({"feature": cols_c, "mean_abs_shap_A": sha, "mean_abs_shap_B": shb})
            dfm["delta_A_minus_B"] = dfm["mean_abs_shap_A"] - dfm["mean_abs_shap_B"]
            dfm = dfm.sort_values("delta_A_minus_B", ascending=False)
            dfm.to_csv(out_dir / f"regime_{name}_feature_shift.csv", index=False)
            reg_rows.append(
                {
                    "regime": name,
                    "n_A": int(len(ia)),
                    "n_B": int(len(ib)),
                    "top_delta_feature": str(dfm.iloc[0]["feature"]),
                    "top_delta_value": float(dfm.iloc[0]["delta_A_minus_B"]),
                }
            )
    pd.DataFrame(reg_rows).to_csv(out_dir / "regime_summary.csv", index=False)

    print("Running conformal prediction ...")
    # 5) Conformal prediction (split conformal using val as calibration)
    reg_conf = fit_xgb_reg(train_df[cols_c].to_numpy(dtype=float), train_df[target_r].to_numpy(dtype=float))
    cal_pred = reg_conf.predict(val_df[cols_c].to_numpy(dtype=float))
    cal_resid = np.abs(val_df[target_r].to_numpy(dtype=float) - cal_pred)
    test_pred = reg_conf.predict(test_df[cols_c].to_numpy(dtype=float))
    conf_rows = []
    for alpha in [0.1, 0.2]:
        q = float(np.quantile(cal_resid, min(0.999, np.ceil((len(cal_resid) + 1) * (1 - alpha)) / len(cal_resid))))
        lo = test_pred - q
        hi = test_pred + q
        y = test_df[target_r].to_numpy(dtype=float)
        cover = ((y >= lo) & (y <= hi)).mean()
        conf_rows.append({"alpha": alpha, "q_hat": q, "target_coverage": 1 - alpha, "empirical_coverage": float(cover), "avg_width": float(np.mean(hi - lo))})
        pd.DataFrame({"date": test_df["date"], "ticker": test_df["ticker"], "y_true": y, "y_pred": test_pred, "lo": lo, "hi": hi}).to_csv(
            out_dir / f"conformal_intervals_alpha_{alpha:.1f}.csv", index=False
        )
    pd.DataFrame(conf_rows).to_csv(out_dir / "conformal_summary.csv", index=False)

    print("Building cross-method explainability consensus ...")
    # 6) Cross-method explainability comparison: XGB SHAP, FFNN SHAP, TabNet attention
    suite_dir = root / "results/panel_model_suite"
    xgb_imp = pd.read_csv(suite_dir / "xgb_model_c_shap_importance.csv")[["feature", "mean_abs_shap"]].rename(
        columns={"mean_abs_shap": "xgb_score"}
    )
    ffnn_imp = pd.read_csv(suite_dir / "ffnn_model_c_shap_importance.csv")[["feature", "mean_abs_shap"]].rename(
        columns={"mean_abs_shap": "ffnn_score"}
    )
    tab_imp = pd.read_csv(suite_dir / "tabnet_model_c_attention_importance.csv")[["feature", "mean_mask"]].rename(
        columns={"mean_mask": "tabnet_score"}
    )
    m = xgb_imp.merge(ffnn_imp, on="feature", how="outer").merge(tab_imp, on="feature", how="outer").fillna(0.0)
    m["xgb_rank"] = m["xgb_score"].rank(ascending=False, method="average")
    m["ffnn_rank"] = m["ffnn_score"].rank(ascending=False, method="average")
    m["tabnet_rank"] = m["tabnet_score"].rank(ascending=False, method="average")
    m["rank_mean"] = m[["xgb_rank", "ffnn_rank", "tabnet_rank"]].mean(axis=1)
    m = m.sort_values("rank_mean")
    m.to_csv(out_dir / "cross_method_explainability_consensus.csv", index=False)

    summary = {
        "outputs_dir": str(out_dir),
        "completed_items": [
            "rolling_window_evaluation_panel",
            "shap_global_dependence_interaction_local",
            "regime_conditional_analysis_5_splits",
            "diebold_mariano_and_clark_west_tests",
            "shap_rank_bootstrap_confidence_intervals",
            "split_conformal_prediction_intervals",
            "cross_method_explainability_consensus",
        ],
        "files": sorted([p.name for p in out_dir.glob("*")]),
    }
    (out_dir / "future_plans_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
