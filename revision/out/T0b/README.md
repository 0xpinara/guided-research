# Task 0b — Correctness rebuild (canonical dataset, pipeline, predictions and evaluation)

**Status:** complete, 2026-10-01; stopped at the decision gate. Task 1 has not started and the manuscript is unchanged.

**Scope:** BIR-era files are never modified. Everything here was produced by `revision/T0b/` from real project data only: no observation was created, patched or invented. Imputation happens only as the documented per-window preprocessing in Section 3.

**Restarted once:** the first grid run (18:35, 39 of 54 jobs finished) still used the future-event features removed by rule R9. It was stopped, and its outputs are kept only as diagnostics in `diagnostic_with_future_event_features/`. **Every result below comes from the second, canonical run.**

---

## 1. Data layer (0b.1): `revision/T0b/build_panel.py` → `panel/`, `data_audit.csv`, `data_checks/`

| Rule | Task 0 finding | Correction | Rows before → after | Values changed |
|---|---|---|---|---|
| R0 | — | start: BIR raw feature panel (`features_panel.parquet`, pre-preprocessing) | 134,368 | — |
| R1 | D2 | **META excluded** (see below) | 134,368 → 133,576 | — |
| R2 | D3 | price-based technicals recomputed on CRSP prices adjusted **only at the 15 corporate-action events** | — | feat_28 75 · feat_29 150 · feat_33 4,285 · feat_34 4,353 · feat_35 347 · feat_36 735 |
| R3 | D6 | GOOG earnings dates from Yahoo (feeds only the ex-post diagnostic file after R9) | — | 2,264 (feat_43/44) · 768 (feat_51) |
| R4 | D4 | feat_25 / feat_26 recomputed with textbook BKM | — | 128,142 / 128,169 (rank corr. with BIR 0.985 / 0.990) |
| R5 | D5 | feat_50 recomputed with contracts keyed on expiration date | — | 133,136 (equals feat_01 on 0.1% of stock rows, was 99.8%) |
| R6 | D12 | realised-volatility targets built on the full CRSP calendar, before any row is dropped | — | — |
| R7 | D11 | no join with the legacy 60/20/20 split file; its 10 buffer dates are kept | — | — |
| **R9** | **D15 (new)** | **future-event features removed** (Section 2) | — | — |
| R8 | D7 | **primary panel**: ticker-days with fewer than 50 cleaned contracts dropped | 133,576 → **129,852** | 3,724 ticker-days |

**Primary panel:** 129,852 ticker-days, 56 stocks + 3 ETFs, 2,261 dates (2016-01-04 .. 2024-12-31), 49–59 names per date (median 58). Three dates drop out because every ticker is thin on them.
- **Robustness panel** (`panel_unfiltered.parquet`): the same corrections with thin days kept, 133,576 rows.
- **Where the thin days are:** concentrated in DUK (1,250 of 2,264 days), NEE (737), TMO (473), HON (293) and SO (210); `data_checks/thin_days_by_ticker.csv`.

**Data sources, as actually used:**
- **CRSP via WRDS** (BIR pull): returns, prices, volume, shares.
- **Compustat via WRDS:** earnings report dates. These now feed only the ex-post diagnostic file.
- **Yahoo Finance:** VIX, VIX3M, ^IRX (T-bill), sector-ETF returns, dividend ex-dates (now used by no canonical feature), and GOOG earnings dates (diagnostic file only).
- **Option source:** the community option files, with vendor-supplied IVs and greeks.

**META (R1).**
- The CRSP pull by ticker maps "META" to two securities:
  - PERMNO 21413 (2021-06-30..2022-01-28, a ~$15 security)
  - PERMNO 13407 (Meta), present only from 2022-06-09
- The Facebook-era history of PERMNO 13407 is not on disk. A re-pull failed: a non-interactive WRDS login with the stored credentials (`~/.pgpass`, user `aksoyp`) timed out after 90 s, probably waiting for Duo approval.
- Your rule was to exclude META if it could not be verified with real data, so it is excluded rather than patched.
- A WRDS session would allow PERMNO 13407 to be used across the ticker change.
- Correction to Task 0: the 148 bad rows had **no** option data (see the Task 0 Erratum).

**Corporate-action adjustment (R2).**
- The factor f_t = |prc_{t−1}|(1+ret_t)/|prc_t| comes from CRSP. Prices before an event with |log f| > log 1.2 are divided by f.
- The 15 events (`data_checks/corporate_action_events.csv`):
  - splits: AAPL 4:1, AMZN 20:1, AVGO 10:1, CMCSA 2:1, GOOG 20:1, NEE 4:1, NVDA 4:1 and 10:1, TSLA 5:1 and 3:1, WMT 3:1
  - GE 1:8 reverse split (f = 0.125)
  - spin-off distributions: GE 2023 (1.287) and 2024 (1.256), T 2022 (1.305)
- Every window without an event keeps exactly the BIR value: raw price, dividends not added back. feat_28 changes on exactly 15 × 5 = 75 rows.

**Validation checks** (`data_checks/`):
- One PERMNO per ticker except META.
- The option-file underlying price agrees with the CRSP price within 5% on every compared day, so there is no other identifier collision.
- One GVKEY per Compustat ticker.
- Yahoo and Compustat report dates agree on **352 of 352** quarters for the 10 tickers that have both.
- BKM self-test on Black–Scholes prices: skew −0.0004 and kurtosis 3.025 (theory: 0 and 3).
- `ret_1d/3d/5d` equal compounded CRSP returns over t+1..t+h (max |diff| 7e-16).

## 2. Additional blocking correction R9 / D15: future-event features removed

| Feature | Definition | Why removed |
|---|---|---|
| feat_43 days_to_next_earnings | days to the next realised Compustat report date (Yahoo for GOOG) | needs the future report date; no point-in-time calendar |
| feat_44 earnings_flag | 1{feat_43 ≤ 7} | derived from feat_43 |
| feat_45 days_to_ex_div | days to the next realised Yahoo ex-dividend date (≤ 90, else 999) | Yahoo has ex-dates and amounts but **no declaration dates**, so nothing shows the ex-date was known at t |
| feat_51 implied_earnings_move | ATM straddle at the expiry nearest the next earnings date, within 30 days of it | defined only through feat_43 |

- **No replacement features were added.**
- **The other features were audited and kept:**
  - every option feature that uses history looks only at the current and earlier dates (IV rank, volume spike, OI/spread/GEX changes, SVI dynamics, feat_50)
  - the stock technicals use trailing windows
  - the calendar features (day of week, days to the monthly option-expiry Friday, quarter-end flag) are deterministic rules known in advance
- The removed columns survive only in `panel/event_features_expost.parquet`, for any **ex-post descriptive** diagnostic.
- **Task 4:** the earnings-proximity conditional test is removed from the primary predictive analysis. It may appear only as a clearly labelled ex-post descriptive diagnostic, unless a point-in-time earnings calendar is obtained. VRP and VIX are unaffected.

**Canonical feature sets** (`feature_sets_canonical.yaml`: the BIR sets minus these four features):

| Set | A | B | C | candidate_6 | D | repr_svi | repr_grid | repr_grid_raw | repr_bkm |
|---|---|---|---|---|---|---|---|---|---|
| BIR count | 21 | 34 | 49 | 6 | 55 | 31 | 45 | 41 | 23 |
| canonical | **18** | **30** | **45** | 6 | **52** | **28** | **42** | **38** | **20** |

Every set also receives 58 ticker dummies (59 tickers, one dropped).

## 3. Walk-forward pipeline (0b.2–0b.3): `revision/T0b/run_grid.py`, `common.py`

**Grid:** 2 targets × 3 horizons × 9 sets × 2 schemes × 3 models = **324 cells**. The window geometry is the same as BIR: expanding uses a 504-date minimum and 63-date step (27 windows); rolling uses 189/63/63 (32 windows). The last h training dates are purged.

**Preprocessing, fitted on each window's training rows only:**
- **Winsorisation:** each feature is clipped to its training 0.1% / 99.9% quantiles; NaN stays NaN. Features entirely missing in a window's training rows are dropped for that window.
- **OLS and ElasticNet:**
  - imputation with the training per-ticker median, then the training global median
  - one missingness indicator per distinct training missingness pattern (about 8–9 per window), which covers the surface block
  - no zero-filling anywhere
- **ElasticNet:** additionally standardised by a `StandardScaler` fitted on the training block (as in BIR).
- **XGBoost:** receives the winsorised features with NaN left as missing (native handling); no imputation.

**Models:**
- **OLS:** float64 `LinearRegression`, full rank (e.g. 119–126 directions on Set D, vs 6–8 in BIR).
- **ElasticNet:** the BIR settings (l1_ratio 0.5, α ∈ {1e-4, 1e-3, 1e-2, 1e-1}, 3-fold CV, random_state 42, max_iter 1000, tol 1e-3), float64.
- **XGBoost:** the BIR hyperparameters (500 trees, depth 6, lr 0.01, subsample 0.8, colsample 0.8, early stopping 50, rmse), plus `tree_method="hist"`, random_state 42 and n_jobs 1, for determinism.

**Training data:**
- OLS and ElasticNet are fitted on the **full** purged training block. BIR used only its first 80% (finding P6).
- XGBoost is fitted on the first 80% of training dates **minus the last h** (purge, P7) and early-stopped on the last 20%.

**Target and saved output:**
- The target is demeaned per date over the names present; the ETFs stay in the training and IC cross-section, as in the manuscript.
- **Every test prediction is saved** for all three models (Section 9).

## 4. Canonical IC (0b.4)

- **Per-window IC:** the mean, over the window's test dates, of the **daily cross-sectional Spearman IC** across names (average ranks for ties).
- **Constant forecasts:** a date on which a model's predictions are constant across names scores **IC = 0** (zero skill).
- **Cell statistic:** the mean per-window IC, with the Newey–West HAC t (Bartlett, lag ⌊n^{1/3}⌋) on the per-window series.
- **Secondary only:** the BIR pooled per-window IC is reported for comparison. The revised paper must not mix the two definitions.

## 5. Trading book and economics (0b.6): `revision/T0b/economics.py`

- **Search family:** the **162** return configurations (OLS/EN/XGB × 9 sets × 2 schemes × 3 horizons). FFNN and TFT were not rerun and are not in the family.
- **Book:**
  - **stocks only** (56 names; SPY, QQQ, IWM excluded), the manuscript's stated design
  - long the top 6 and short the bottom 6 at ±1/6 (dollar-neutral), reopened every h-th test date on non-overlapping holds
  - costs as in BIR: 5 bp × 15% on turnover, plus 50 bp/yr borrow on the short leg
  - the all-names book is computed only for comparison with BIR
- **Ties:** names tied at a cut-off share the remaining slots equally, so no name is chosen by ticker or row order. If all predictions are tied, or the tie groups overlap, the book is **flat**.
- **Sharpe:** arithmetic mean/sd × √(252/h).
- **SR0:** the expected maximum of N = 162 zero-skill Sharpes.
- **DSR:** Bailey–López de Prado PSR(SR0), using each configuration's own skewness and kurtosis.
- **PBO:** CSCV with S = 8 (70 splits) and ω = rank/(N+1).
  - family-level: all 162 configurations, on calendar-month returns over the common months
  - group-level: each scheme × horizon group of 27 configurations

## 6. Bootstrap (0b.7): `revision/T0b/bootstrap.py`

- **Method:** a moving-block bootstrap over the cell's trading-date calendar.
  - block length **21** trading dates, pre-specified and longer than the 5-day horizon
  - 10,000 replications, seed 20261001
  - all daily series of a cell are indexed by the same resampled dates, so same-day pairing (high/low, D/A) is kept
  - percentile 95% intervals; p = 2·min(P(stat* ≤ 0), P(stat* ≥ 0))
- **Self-test on an AR(1) series** (ρ = 0.6): the block interval is 1.85× the i.i.d. width (theory 2.0).
- **Terminology:** the BIR regime procedure resampled single days, so it is described as an *i.i.d. day bootstrap*, not a block bootstrap. Task 4 uses this module unchanged.

## 7. Reproducibility infrastructure (0b.8)

- **`.gitignore`:** `data/` → `/data/`, so `src/data/` is no longer excluded. The large regenerable outputs are kept out of git, with checksums in `MANIFEST_large_files.sha256`.
- **Exact versions** in `revision/requirements.txt` (Python 3.9.6, numpy 2.0.2, pandas 2.2.3, scipy 1.13.1, scikit-learn 1.6.1, xgboost 2.1.4). Seeds: random_state 42; the bootstrap seed is above.
- **Run order:** `revision/T0b/run_all.sh` (it never deletes BIR files).
- **Figures:** BIR's `scripts/make_paper_figures.py` stays frozen (BIR-era file); its problems are findings R1/R2. Revision figures will be produced by new scripts under `revision/` that read every number from `revision/out/` and write to `paper/figures/`; none are needed for Task 0b.
- **Number → script map:** Section 10.


---

## 8. Corrected return benchmark (0b.5)

Table 3 layout: rolling 9m/3m, h = 5, mean per-window **daily cross-sectional** rank IC; † = HAC t > 3. Full grid: `benchmark/cells.csv`; this slice: `benchmark/table3_corrected.csv`.

| Set | ret OLS | ret EN | ret XGB | vol OLS | vol EN | vol XGB |
|---|---|---|---|---|---|---|
| A | +0.0052 | +0.0042 | +0.0099 | +0.522† | +0.515† | +0.477† |
| B | +0.0137 | +0.0066 | +0.0089 | +0.539† | +0.566† | +0.536† |
| C | +0.0057 | -0.0012 | +0.0006 | +0.546† | +0.575† | +0.551† |
| candidate_6 | +0.0148 | +0.0229† | +0.0201† | +0.542† | +0.568† | +0.529† |
| D | +0.0062 | +0.0058 | +0.0057 | +0.541† | +0.573† | +0.552† |
| repr_svi | +0.0038 | +0.0022 | +0.0025 | +0.538† | +0.557† | +0.537† |
| repr_grid | +0.0064 | +0.0028 | +0.0072 | +0.543† | +0.573† | +0.551† |
| repr_grid_raw | +0.0057 | +0.0084 | +0.0131 | +0.547† | +0.574† | +0.552† |
| repr_bkm | +0.0043 | -0.0026 | +0.0077 | +0.526† | +0.522† | +0.493† |

**Grid-wide (162 return cells)**
- **7 cells clear t > 3.** All are on the rolling scheme; **none is a surface set on the Set-A control**:

  | Cell | IC | t |
  |---|---|---|
  | XGB / B / h1 | +0.0150 | 5.51 |
  | XGB / candidate_6 / h3 | +0.0183 | 5.21 |
  | EN / candidate_6 / h3 | +0.0195 | 4.22 |
  | XGB / repr_grid_raw / h3 | +0.0188 | 3.69 |
  | EN / candidate_6 / h5 | +0.0229 | 3.38 |
  | XGB / candidate_6 / h5 | +0.0201 | 3.26 |
  | XGB / C / h1 | +0.0122 | 3.15 |

- **Standard error:** the median HAC SE of the mean IC is 0.0061, so 3 × SE = **0.0182**.
- **Return lift over Set A** (mean over EN + XGB × 2 schemes × 3 horizons; `benchmark/lifts_over_A.csv`):

  | Set | Lift | Per-cell paired HAC t | Note |
  |---|---|---|---|
  | D | +0.0005 | −0.8 .. +1.5 | nested |
  | repr_svi | −0.0027 | −1.3 .. +0.7 | nested |
  | repr_grid | +0.0019 | −0.5 .. +2.1 | nested |
  | repr_grid_raw | +0.0017 | −0.9 .. +1.7 | nested |
  | repr_bkm | −0.0011 | −1.3 .. +1.1 | nested |
  | C | −0.0004 | −1.2 .. +2.1 | nested |
  | B | +0.0037 | −1.0 .. +2.1 | non-nested |
  | candidate_6 | +0.0076 | −1.0 .. +3.2 | non-nested |

- **Undefined t:** for C, D, repr_svi and repr_bkm, one cell has an undefined t, because both ElasticNet fits are null models in every window, so the lift is 0 throughout.

## 9. Corrected volatility benchmark

- **All 162 volatility cells clear t > 3**; the largest HAC t is 62.3, and the median SE is 0.0101.
- **Set A**, EN rolling 5d: 0.515.
- **Lift over Set A** (EN + XGB, full grid):

  | Set | Lift | Per-cell paired t (median) |
  |---|---|---|
  | D | **+0.0421** | 5.8 .. 21.4 (11.9) |
  | repr_grid | +0.0418 | 5.8 .. 19.1 |
  | repr_grid_raw | +0.0432 | 6.2 .. 18.9 |
  | repr_svi | +0.0315 | 3.5 .. 13.9 |
  | C | +0.0447 | 7.9 .. 22.2 |
  | candidate_6 | +0.0335 | 3.3 .. 7.3 |
  | B | +0.0318 | 2.6 .. 7.7 |
  | repr_bkm | +0.0042 | 0.0 .. 6.3 |

- **No constant predictions:** no volatility test date has constant predictions.

## 10. Old (paper) vs corrected headline results

`benchmark/headline_numbers.csv`; cell level in `benchmark/old_vs_new_cells.csv` and `benchmark/old_vs_new_headline_cells.csv` (sets A, D, repr_svi, repr_grid, repr_grid_raw, repr_bkm × both schemes × h = 1, 3, 5 × both targets × 3 models).

| Quantity | Paper (BIR pipeline) | Corrected (Task 0b) |
|---|---|---|
| return cells with t > 3 (of 162) | 7 | 7 |
| volatility cells with t > 3 (of 162) | 158 | 162 |
| largest volatility HAC t | 58.7 | 62.3 |
| median HAC SE of the mean return IC | 0.0064 (198 cells incl. FFNN/TFT) | 0.0061 (162 cells) |
| t>3 detection threshold (3 x SE) | 0.019 | 0.0182 |
| return lift D over A (EN+XGB, full grid) | +0.0027 (paper: about 0.003) | +0.0005 |
| volatility lift D over A (EN+XGB, full grid) | 0.0362 ('more than ten SEs') | 0.0421 (per-cell paired t 5.8..21.4) |
| volatility lift repr_svi over A | 0.0235 | 0.0315 |
| Set A volatility IC, EN rolling 5d | 0.520 | 0.515 |
| return IC ElasticNet/repr_grid/rolling_9m_3m/h5 | 0.0346 (t 4.28) | +0.0028 (t +0.47) |
| return IC ElasticNet/D/rolling_9m_3m/h5 | 0.0331 (t 4.81) | +0.0058 (t +1.24) |
| return IC ElasticNet/A/rolling_9m_3m/h5 | 0.0311 (t 3.21) | +0.0042 (t +1.32) |
| return IC ElasticNet/repr_bkm/rolling_9m_3m/h5 | 0.0312 (t 3.24) | -0.0026 (t -0.70) |
| return IC XGBoost/A/expanding/h1 | 0.0136 (t 3.64) | +0.0057 (t +1.21) |
| return IC XGBoost/D/expanding/h3 | 0.0118 (t 1.47) | +0.0048 (t +0.61) |
| return IC XGBoost/D/rolling_9m_3m/h5 | -0.0025 (t -0.39) | +0.0057 (t +0.63) |
| headline long/short Sharpe (EN/D/rolling/5d) | 1.15 (geometric formula, all 60 names) | 0.403 stocks only, arithmetic (0.368 all names) |
| headline final $ from $1,000 | 5,228 | 1,437 stocks (1,381 all names) |
| headline periods flat (all predictions tied) | 0 (ties broken arbitrarily) | 78.2% |
| ETF-free headline (Sharpe / $) | 1.07 / 4,585 (geometric) | 0.403 / 1,437 |
| SR0 (expected-maximum Sharpe) | 1.153 (N = 198) | 0.631 (N = 162) |
| configurations above SR0 | 2 of 198 | 11 of 162 |
| DSR of the headline cell | about 0.497 (referee_fixes.txt) | 0.252 |
| PBO | 0.571 (expanding/3d group) | 0.600 (family of 162); headline group rolling h5 0.429 |

**Across the 108 headline return cells** (sets A, D, repr_* × 2 schemes × 3 horizons × 3 models):
- BIR and corrected ICs are **uncorrelated** (correlation −0.26).
- Cells clearing t > 3 go from 5 (BIR) to 1 (corrected).

**Across the 108 volatility cells:** the correlation is 0.69, and all 108 clear t > 3 under both pipelines.

## 11. IC-definition comparison (0b.4)

Mean per-window IC and HAC t for the two headline cells, under three definitions, using both the BIR predictions and the corrected ones. For ElasticNet, the BIR predictions are the Task 0 re-fit, which is identical to the BIR run. Source: `benchmark/ic_definition_comparison.csv`.

| Cell | Pipeline | Pooled, NaN dropped (BIR) | Pooled, NaN = 0 | **Daily cross-sectional (canonical)** |
|---|---|---|---|---|
| ElasticNet/D/rolling_9m_3m/5 | BIR | +0.0331 (t 4.81; 10 win) | +0.0103 (t 2.31; 32 win) | +0.0092 (t 2.16; 32 win) |
| ElasticNet/D/rolling_9m_3m/5 | corrected | +0.0313 (t 1.97; 7 win) | +0.0068 (t 1.16; 32 win) | +0.0058 (t 1.24; 32 win) |
| XGBoost/D/expanding/3 | BIR | +0.0118 (t 1.47; 27 win) | +0.0118 (t 1.47; 27 win) | +0.0088 (t 1.20; 27 win) |
| XGBoost/D/expanding/3 | corrected | +0.0038 (t 0.41; 27 win) | +0.0038 (t 0.41; 27 win) | +0.0048 (t 0.61; 27 win) |

**How to read it:**
- **ElasticNet:** almost all of the BIR number comes from dropping the constant-forecast windows (0.0331 on 10 windows → 0.0103 on 32). The pooled → daily switch itself moves little (0.0103 → 0.0092), and the corrected pipeline lowers it further (0.0058, t 1.24).
- **XGBoost:** the definition change takes 0.0118 → 0.0088 on the BIR predictions; the corrected pipeline gives 0.0048 (t 0.61).

The revised paper uses only the daily cross-sectional definition.

## 12. Economics (0b.6)

Source: `economics/strategy_summary.csv`, `pbo.csv`, `economics.txt`, `period_returns_stocks.parquet`.

**SR0 and DSR**
- **Search family:** the 162 OLS/EN/XGB configurations, stocks-only book, arithmetic Sharpe.
- **Five configurations never trade:** ElasticNet, expanding, h1, on Sets A, C, D, repr_bkm and repr_svi. Their predictions are constant in all 27 windows, so the book is always flat.
- **Primary convention (approved by the authors, 2026-10-02):** SR0 is computed over the full pre-specified family of 162 configurations. The five always-flat configurations are valid, pre-specified strategies that earn zero economic return, i.e. zero skill, so they enter at Sharpe = 0. This matches the IC = 0 rule for constant forecasts.

  | | Primary (N = 162) | Sensitivity (N = 157, undefined Sharpes excluded) |
  |---|---|---|
  | SR0 | **0.631** | 0.624 |
  | configurations above SR0 | 11 | 11 |
  | configurations with DSR > 0.95 | **0** | 0 |
  | headline-cell DSR | 0.252 | 0.258 |
  | best-configuration DSR | 0.783 | 0.789 |

  **The conclusion is unchanged:** under either convention no configuration's deflated Sharpe ratio reaches 0.95, so there is no tradeable-strategy claim.
- **Best configuration:** XGB / D / rolling / h5, Sharpe 0.909, DSR 0.783.

**Headline cell, EN / D / rolling / h5**

| Book | Sharpe | Final $ (from $1,000) | DSR | Periods flat |
|---|---|---|---|---|
| Stocks only | **0.403** | **1,437** | 0.252 | **78.2%** (all predictions tied) |
| All names (for comparison with BIR) | 0.368 | 1,381 | — | 78.2% |

**PBO** (CSCV, S = 8, ω = rank/(N+1))

| Family | PBO |
|---|---|
| All 162 (calendar-month returns, 82 common months) | **0.600** |
| Expanding h1 / h3 / h5 | 0.600 / 0.400 / 0.214 |
| Rolling h1 / h3 / h5 (headline's group) | 0.829 / 0.400 / **0.429** |

**Caution:** 88% of the 162 configurations have a positive Sharpe (median 0.256). The book is dollar-neutral but **not beta- or sector-neutral**, so a common tilt across configurations is likely; these Sharpe ratios are not evidence of skill.

## 13. Robustness: 50-contract filter vs unfiltered

The pipeline is identical; the only difference is that the unfiltered panel keeps the 3,724 thin ticker-days (133,576 rows). The comparison covers all 162 return cells plus volatility on Sets A and D (36 cells). Source: `robustness/unfiltered_vs_primary.csv`, `robustness/summary.txt`; unfiltered outputs in `benchmark_unfiltered/` and `predictions_unfiltered/`.

**Returns**
- Cell ICs correlate **0.845** between the two panels; mean |ΔIC| 0.0028, max 0.0110.
- **7 cells clear t > 3 in each panel; 5 of them are common:**
  - EN / candidate_6 / rolling h3 and h5
  - XGB / B / rolling h1
  - XGB / candidate_6 / rolling h3 and h5
- Only with the filter: XGB / C / rolling h1 and XGB / repr_grid_raw / rolling h3.
- Only without the filter: OLS / candidate_6 / rolling h1 and OLS / repr_grid_raw / expanding h1.

**Volatility:** correlation **1.000**, mean |ΔIC| 0.0008, and all 36 cells clear t > 3 in both panels.

**Headline cells, return, rolling h5**

| Cell | Primary (filter) | Unfiltered |
|---|---|---|
| EN / A | +0.0042 (t 1.32) | −0.0008 (t −0.25) |
| EN / D | +0.0058 (t 1.24) | +0.0066 (t 1.39) |
| OLS / A | +0.0052 (t 0.64) | +0.0094 (t 1.16) |
| OLS / D | +0.0062 (t 0.83) | +0.0100 (t 1.39) |
| XGB / A | +0.0099 (t 2.03) | +0.0064 (t 0.87) |
| XGB / D | +0.0057 (t 0.63) | +0.0076 (t 1.18) |

None of the conclusions depends on the filter.

## 14. Constant-prediction statistics

Source: `benchmark/constant_predictions.csv`. These are test dates whose predictions are constant across names; each scores IC = 0, and on rebalance dates the book is flat.

| Target | Model | Share of test dates | Windows constant throughout |
|---|---|---|---|
| return | ElasticNet | **48.5%** | **773 / 1,593** |
| return | OLS | 0.0% | 0 / 1,593 |
| return | XGBoost | 0.1% | 0 / 1,593 |
| volatility | all three | 0.0% | 0 |

**All 773 constant-throughout ElasticNet windows are intercept-only fits**, with every coefficient zero. The selected α is 0.1 in 695 of them, 0.01 in 27 and 0.001 in 51: the demeaned return target has so little variance that even α = 0.001 can zero every coefficient. Another 6 windows are only partly constant (α = 0.01, one nonzero coefficient).

## 15. ElasticNet convergence audit

Source: `revision/T0b/en_convergence_audit.py` → `en_convergence/en_fits.csv`, `en_convergence/summary.txt`.

**Method:** every ElasticNet window of the canonical grid was re-fitted identically (3,186 fits, 108 cells). The re-fits reproduce the saved predictions exactly (max |diff| 0).

**Results**
- **Convergence warnings:** 34, in 22 of the 3,186 window fits (0.7%); 32 for returns (20 fits) and 2 for volatility (2 fits).
- **Final fits at the selected α:** **0 of 3,186 failed to converge.** They took a median of 13 iterations, at most 725, against `max_iter` = 1,000.
- **Where the warnings come from:** all 34 come from the cross-validation path, i.e. the α grid fitted on the CV folds, not from a selected model.

**Conclusion:** no numerical convergence fix is required, and the BIR specification (max_iter 1000, tol 1e-3) is unchanged. The only possible effect is that, in those 22 windows, a CV score for a non-selected α rests on an approximate fit.

**Selected α**

| Target | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|---|---|---|---|
| return | 37 | 826 | 35 | 695 |
| volatility | 862 | 722 | 9 | 0 |

## 16. Saved predictions and final counts

- **Predictions** (every test observation; 34,846,686 rows over 324 cells; 409 MB; zstd parquet):
  - `revision/out/T0b/predictions/<target>/h<h>/<set>.parquet`, with `target` ∈ {return, volatility}
  - columns: target, model, feature_set, scheme, horizon, window, date, ticker, pred, y (demeaned target), y_raw
- **Robustness predictions:** `revision/out/T0b/predictions_unfiltered/`.
- **Diagnostics (not canonical):** `revision/out/T0b/diagnostic_with_future_event_features/`.
- **Final data:**
  - primary panel: 129,852 ticker-days, **59 tickers = 56 stocks + SPY/QQQ/IWM** (META excluded), 2,261 trading dates (2016-01-04 .. 2024-12-31), 49–59 names per date
  - each cell is tested on 1,701 dates / 98,769 ticker-days (expanding, 2018-01-03 .. 2024-10-10) or 2,016 dates / 116,334 ticker-days (rolling, 2016-10-03 .. 2024-10-10)
  - unfiltered robustness panel: 133,576 ticker-days

## 17. Remaining warnings and limitations

**Run history**
- The canonical grid was interrupted once: the Mac's battery fell to 1%, which throttled the CPU and caused sleep.
- It was resumed on mains power under `caffeinate`, running only the 32 unfinished jobs.
- The 22 jobs completed before the interruption are byte-identical to their pre-resume checksums (`logs/completed_before_resume.sha256`, 44/44 OK).
- Jobs are independent and deterministic: candidate_6 predictions are bit-identical across two separate runs (3.2 million predictions, max |diff| 0).

**Data**
- **META** is excluded. A WRDS re-pull by PERMNO 13407 across the ticker change would allow it back (WRDS could not be reached non-interactively).
- **The universe** is the BIR 60 minus META. Survivorship and end-of-sample selection still bias results upward, as stated in BIR.
- **The other 44 Dubach symbols are not on this machine** (`revision/out/data_recovery/`).
- **feat_42 (sector-ETF return)** keeps the BIR mapping (META/GOOG/NFLX → XLK). XLC exists only from 2018-06, so the communication names' feat_42 is missing before then; it is imputed per window, with an indicator.
- **IVs and greeks are vendor-supplied**, with an unknown dividend/rate/American-exercise treatment.
- **BKM moments** use OTM quotes inside the cleaning filter's 0.80–1.20 moneyness band (truncation), with spot-based log-moneyness and no dividends.

**Model behaviour**
- ElasticNet's return forecasts are intercept-only in 48.5% of test dates (773 of 1,593 windows). Its return numbers therefore mostly reflect "no forecast", scored IC = 0.
- XGBoost's return fits stop very early: a median of 10 trees, and 20 or fewer in 62% of windows (volatility: median 185 trees).

**Economics**
- Spread costs are assumed (5 bp × 15%), not measured, and borrow is a flat 50 bp.
- The book is dollar-neutral but **not beta- or sector-neutral**: 88% of configurations have a positive Sharpe.
- The 5 never-trading configurations enter SR0 at Sharpe 0. This is a convention; excluding them gives an SR0 of 0.624 instead of 0.631.
- The family PBO uses the 82 calendar months common to all 162 configurations, i.e. the expanding test period.
- FFNN and TFT were not rerun, so the BIR trial count N = 198 no longer applies.

**Scope and wording**
- The last partial test block (2024-10-11 .. 2024-12-31) is not evaluated, the same window geometry as BIR.
- The earnings-proximity conditioner can now only be an ex-post descriptive diagnostic (rule R9).
- The volatility target is the simple sum-of-squares realised volatility.

**Log warnings, none affecting results**
- numpy "All-NaN slice" RuntimeWarnings: a feature entirely missing in a training window, which is dropped for that window by design.
- sklearn ConvergenceWarnings from the CV path (Section 15).
- One multiprocessing "leaked semaphore" warning, from the interrupted run's shutdown.

## 18. Number → script map

| Numbers | Script | Output |
|---|---|---|
| data rules, row counts, validation | `build_panel.py` | `data_audit.csv`, `data_checks/*`, `feature_sets_canonical.yaml` |
| Yahoo earnings snapshot | `fetch_yahoo_earnings.py` | `raw/yahoo_earnings_dates.csv` |
| per-window metrics, predictions | `run_grid.py` | `benchmark/parts/`, `predictions/`, `benchmark_unfiltered/`, `predictions_unfiltered/` |
| cell statistics, Table 3, lifts, old vs new, IC definitions, constant predictions, headline table | `summarize.py` | `benchmark/{cells,table3_corrected,lifts_over_A,old_vs_new_cells,old_vs_new_headline_cells,ic_definition_comparison,constant_predictions,headline_numbers}.csv`, `benchmark/summary.txt` |
| Sharpe, SR0, DSR, PBO | `economics.py` | `economics/{strategy_summary.csv,pbo.csv,economics.txt,period_returns_stocks.parquet}` |
| 50-contract robustness | `robustness.py` | `robustness/{unfiltered_vs_primary.csv,summary.txt}` |
| ElasticNet convergence | `en_convergence_audit.py` | `en_convergence/{en_fits.csv,summary.txt}` |
| bootstrap definition | `bootstrap.py` | (library; self-test printed) |
| checksums of large outputs | `run_all.sh`, step 5 | `MANIFEST_large_files.sha256` |

The full rerun order is in `revision/T0b/run_all.sh`.
