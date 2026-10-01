# Task 0 — Audit and definitions (BIR-era code and results)

**Status:** complete and frozen, 2026-09-30. Nothing in the manuscript, `results/`, `data/`, `src/` or `scripts/` was modified. Task 1 has not started.

**What this covers:** the code, data and saved outputs behind the manuscript submitted to Borsa İstanbul Review (`paper/manuscript.tex`). It answers the Task 0 items in `paper/steps.md`. Section 9 lists every finding, favourable or not.

**Main finding:** every number printed in the paper's tables reproduces from the saved outputs. But the audit found blocking correctness problems in the pipeline that produced those outputs. The saved predictions therefore cannot be the canonical basis for Tasks 1–5; Task 0b has to rebuild them first. Two blocking findings came out of the ElasticNet re-fit done for this write-up and were not in the earlier review: **P4** (ElasticNet constant forecasts are silently dropped) and **P5** (the long/short book is built from tie-broken predictions).

---

## 0. Frozen BIR-era state

| What | Where |
|---|---|
| Local git snapshot | commit `b47f351` on `main`, annotated tag `bir-snapshot-2026-09-30`. Revision work continues on branch `revision`. Includes `src/data/`, `results/logs/` and the two OOS parquets, all force-added because the existing `.gitignore` rule `data/` also matches `src/data/`. `data/` (9.6 GB) is not in git. |
| Full immutable copy | `~/Desktop/options_research_BIR_snapshot_2026-09-30/`: an APFS clone of the whole project including `data/` and `.git`. Made read-only with `chmod -R a-w`. Contains `MANIFEST.sha256` (1,041 files) and `SNAPSHOT_README.txt`. Verify with `shasum -a 256 -c MANIFEST.sha256 --quiet` from inside the folder. |
| Not in git, only in the clone | `data/`, `results/model_checkpoints/`, and three personal root files kept out of git by a local-only `.git/info/exclude` (the poster HTML, `pinar-guidedpaper.pdf`, `background.jpg`). |

Anonymity: the snapshot commit's author is your git identity (`0xpinara`), and `paper/steps.md` names the authors' university. Do not push this local history to the anonymized GitHub remote.

## 1. How this audit was produced

| Script (new, `revision/T0/`) | Output (`revision/out/T0/`) | What it does |
|---|---|---|
| `t0_refits.py` (17 s) | `t0_refits.txt`, `en_D_rolling_h5_predictions.parquet`, `en_D_rolling_h5_window_ic_check.csv`, `ols_rank_check.csv` | Re-fits ElasticNet/D/rolling/5d with the unchanged BIR code path and saves its predictions. Re-fits OLS in float32 and float64 for 6 sets × 3 windows. |
| `t0_checks.py` (14 s; run after `t0_refits.py`) | `t0_checks.txt`, `S1_…` to `S12_…` CSVs | Recomputes every number cited below from `results/tables/` and `data/`. Sections S1–S13 are referenced throughout. |

**Seeds:**
- The published i.i.d. regime bootstrap uses seed 1 and 5,000 draws, as in `return_extras.py`.
- The moving-block regime bootstrap uses seed 1 and 4,000 draws, reset for each cell × conditioner × block length.
- The meta-analysis quarter bootstrap uses seed 7, as in `return_extras.py`.

**Environment** (S13): Python 3.9.6 on macOS 15.6.1 arm64; numpy 2.0.2, pandas 2.2.3, scipy 1.13.1, scikit-learn 1.6.1, xgboost 2.1.4, pyarrow 21.0.0, matplotlib 3.9.4, torch 2.8.0. With these versions the float32 OLS re-fits reproduce the saved OLS ICs, so they are, or behave like, the versions the BIR grid used. The BIR runs recorded no versions.

Items marked *(sub-audit)* come from a code read by a delegated review and were not independently recomputed. Every other number here was recomputed by the scripts above.

---

## 2. Saved predictions (Task 0 item 1)

| File | Content | Keys | ETFs |
|---|---|---|---|
| `results/tables/full_matrix_oos_xgb.parquet` | **XGBoost only**, return target, all 9 sets × 2 schemes × 3 horizons; 5,961,033 rows; columns `pred`, `y_demeaned`, `y_raw` | model, feature_set, scheme, horizon, date, ticker. There is **no window column**: windows are the consecutive 63-date test blocks. | included (SPY, QQQ, IWM) |
| `results/tables/vol_oos_xgb.parquet` | XGBoost only, volatility target; 5,961,033 rows; `pred`, `y_demeaned` | same | included |
| `results/tables/full_matrix_ls_daily.parquet` | per-period long/short P&L at the 15% cost scheme for OLS, EN, XGB and FFNN | model, feature_set, scheme, horizon, date | the book trades all 60 names |

- **No OLS, ElasticNet, FFNN or TFT predictions were saved.** The planned check "EN Set D rolling 5d from saved predictions" is therefore impossible as written; Section 3 re-fits it instead.
- There is no TFT long/short series: the file is written only when a run finishes, and the TFT run did not finish (P9).
- New in Task 0: `revision/out/T0/en_D_rolling_h5_predictions.parquet` holds the re-fitted EN headline cell (119,682 rows, includes `window`).

## 3. Per-window IC (Task 0 item 2)

**Definition used for every benchmark number** (`src/evaluation/metrics.py:61-64,234`; `scripts/run_vol_matrix.py:170`): one Spearman correlation between the raw model prediction and the date-demeaned target, **pooled over all ticker-days of the 63-day test window** (about 60 × 63 rows). The cell statistic is the mean of these per-window values. It is **not** the mean of daily cross-sectional Spearman ICs.

Other sections use other definitions:
- **Regime test** (`return_extras.py:39-55`) and the SVI-vs-raw comparison in `referee_fixes.py`: mean of daily cross-sectional Spearman ICs.
- **Sector-neutral check** (`sector_neutral_ic.py`): pooled per 63-day block, same as the benchmark.

The paper therefore mixes two IC definitions (§4.1, §4.5).

**Reproducing EN / D / rolling 9m/3m / 5d** (`t0_refits.txt`):
- The BIR code path re-fits identically: per-window IC max |fresh − saved| = 9.7e-17.
- Mean IC 0.033099, NW t 4.8074, equal to the saved values (paper: 0.0331, 4.81).
- Long/short on the re-fitted predictions: $5,227.71 and coded Sharpe 1.1504, both identical to the saved values.
- **But only 10 of the 32 windows have a defined IC.** In the other 22, ElasticNet chose the intercept-only model, the prediction is constant, its Spearman is NaN, and the HAC function drops it silently (P4).

**XGBoost cells** reproduce from the saved predictions; for example the raw column of `sector_neutral_ic.txt` gives D/rolling/5d = −0.0025, matching Table 3.

## 4. HAC inference (Task 0 item 3)

`scripts/posthoc_stats.py:37-54` (`_nw_tstat`, duplicated in `run_vol_matrix.py`, `signal_injection.py` and `sector_neutral_ic.py`):
- Newey–West with a Bartlett kernel and lag max(1, ⌊n^{1/3}⌋), which is 3 for both 27 and 32 windows.
- Applied to the per-window IC series after **dropping non-finite values**.
- "Survives t>3" uses |t| > 3. All 7 published survivors are positive.

## 5. Regime variables (Task 0 item 4)

| Conditioner | Definition | Level of the split |
|---|---|---|
| VRP (`feat_12`) | ATM IV minus trailing 30-day realised vol. The ATM IV (`feat_06`, `aggregated_features.py:56-75`) is the mean vendor IV of contracts with K/S in [0.97, 1.03] at the expiry nearest 30 DTE. The realised vol is the SD of CRSP daily returns over t−29..t, × √252. Units are volatility, not variance. | **per ticker-day.** Within-date SD 0.066 vs SD of daily means 0.062 (S8). Threshold = expanding median of the pooled VRP over all earlier dates. |
| VIX (`feat_38`) | VIX close (Yahoo) | per date (within-date SD 0) |
| Earnings (`feat_43`) | days to next earnings ≤ 7 (Compustat dates) | per ticker-day |

- `fig_regime_return.pdf` comes from `scripts/make_paper_figures.py`, which plots **hard-coded constants** copied from `return_extras.txt`. Rerunning `return_extras.py` would not update the figure.
- Because the VRP split is per ticker-day, it is a cross-sectional conditioning, not a market regime.

## 6. Implied volatilities and option data fields (Task 0 item 5)

- **Source:** IVs and greeks (`impl_volatility`, `delta`, `gamma`, `theta`, `vega`, `rho`) are **supplied by the data vendor**; our code never inverts option prices. The vendor's dividend, interest-rate and American-exercise treatment is unknown, and the upstream repository is now private.
- **Our own moment calculation:** the BKM moments use rf from the T-bill series and a forward of F = S·e^{rτ} with no dividends.
- **Per-contract fields available:** `best_bid`, `best_offer`, `mid_price`, bid/ask sizes, `last`, volume, open interest, `underlying_price` (same date), expiration, strike, type. Raw `*_underlying.parquet` files hold `close` and `adjusted_close`.
- **Implications for Task 3:** Method A (put–call parity on mid prices) is feasible. Method B (IV spread) cannot be validated against a known IV methodology.

## 7. Surface features in Set D (Task 0 item 6)

The 34 columns:
- **SVI parameters (5):** `svi_a`, `svi_b`, `svi_rho`, `svi_m`, `svi_sigma`, from the expiry nearest 30 days.
- **SVI dynamics (5):** `d_svi_*`, the day-over-day change. These jump when the reference expiry rolls *(sub-audit)*.
- **(Δ, τ) grid (20):** `iv_{10dp,25dp,atm,25dc,10dc}_{1m,2m,3m,6m}`, evaluated from SVI at 30/60/90/180 days. Each tenor uses the nearest expiry within ±50%, with no interpolation across expiries *(sub-audit)*.
- **Derived (4)**, from `surface_model_features.py:326-351`:
  - `rr_25d_1m` = IV(25Δ call) − IV(25Δ put). The put-minus-call skew used in Task 5 is −rr.
  - `bf_25d_1m` = mean(IV 25Δp, IV 25Δc) − IV ATM.
  - `term_slope_atm` = IV ATM 3m − IV ATM 1m.
  - `wing_spread_1m` = IV(10Δ put) − IV(10Δ call).

Other skew and term-structure measures: `svi_rho` (parametric skew); and, in Sets B and C, `feat_08` (iv_skew) and `feat_09` (term-structure slope).

**`repr_grid_raw` is a different grid** (`iv_surface_features.py:18-78`):
- cubic `griddata` over 5 moneyness levels (K/S 0.90–1.10) × 3 maturities (30/60/90 days), plus 5 derived features
- nearest-neighbour fill where the cubic interpolation fails
- no 6-month tenor, and fixed ±10% wings rather than 10Δ

## 8. Long/short backtest and the existing injection code (Task 0 items 7–8)

**Backtest** (`src/trading/backtest.py:114-236`, called at `run_full_matrix.py:573-600`):
- **Inputs:** the concatenated out-of-sample predictions of all windows.
- **Rebalancing:** on every h-th date of that sequence, so holds are non-overlapping.
- **Positions:**
  - rank by prediction; long the top 6 and short the bottom 6, each at ±1/6
  - dollar-neutral, gross exposure 2
  - **all 60 names, ETFs included**
- **Period P&L:** Σ w·r(h-day raw return) − spread cost − borrow.
- **Spread cost:** Σ|Δw| over both legs × 5 bp × 0.15, i.e. 0.75 bp per unit of turnover. The 5 bp is an assumed spread applied to every name; `quoted_spread=None` everywhere.
- **Borrow:** 50 bp/yr × short gross exposure (1.0) ÷ (252/h), charged each period on the short leg only, at the same flat rate for all names.
- **Leg detail:** long-leg and short-leg returns are stored per period.
- **Sharpe:** ((1+μ)^{252/h} − 1)/(σ√(252/h)), a geometric annualised return over annualised vol (E1).
- **Ties:** predictions are sorted with pandas' default, non-stable sort, so tied predictions produce an arbitrary basket (P5).

**Signal injection** (`scripts/signal_injection.py`):
- Base target: y_base = demean_t(r^{(3)}). σ = pooled SD of y_base.
- Injection: c = δσ/√(1−δ²), z ~ N(0,1) i.i.d. per row (seed 1000+s, s ∈ {0,1,2}), y_inj = demean_t(y_base + c·z).
- z is appended to Set A plus the ticker dummies.
- Design: expanding scheme, h = 3; OLS, EN and XGB.
- Oracle IC: the pooled-per-window Spearman(z, y_inj), averaged over windows. Recovered IC: the same definition applied to the prediction.
- The realised oracle IC is **≈1.22–1.23 × δ** (S12; e.g. δ = 0.05 → 0.0611), not δ as the paper states.

---

## 9. Findings register

Severity: **B** blocking (changes a reported result or invalidates an inference) · **M** major (method differs from the paper; numbers may change) · **W** wording only (numbers unaffected) · **I** infrastructure / reproducibility. The same list is in `findings.csv`.

### Data layer

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| D1 | B | **Stock data are CRSP/Compustat via WRDS, not Yahoo.** Returns, prices, volume and shares (market cap) come from CRSP; earnings dates from Compustat; FF factors from WRDS. Yahoo supplies only VIX, VIX3M, ^IRX, sector ETFs and dividends. The paper (§3, Data availability) and `README.md` say Yahoo; `settings.yaml:105-107` says WRDS is unused. CRSP licensing also limits "panel available on request". | `src/data/clean_stocks.py:47,128-136,179` |
| D2 | B | **META identifier error.** CRSP "META" is two securities. PERMNO 21413 covers 2021-06-30..2022-01-28 (148 rows, median price $14.91): not Meta Platforms. PERMNO 13407 (Meta) covers only 2022-06-09..2024-12-31 (644 rows). The option snapshot's META starts 2021-07-08 with underlying ≈ $343, i.e. Facebook/Meta. So 148 rows pair Meta's options with another security's returns, and Meta is missing from 2022-01-28 to 2022-06-09. No other ticker maps to more than one PERMNO, and no PERMNO maps to more than one ticker. | S11, `S11_ticker_permno_collisions.csv` |
| D3 | M | **Price-based features are not split-adjusted.** `feat_28/29/33-36` are computed from raw CRSP `prc`. There are 16 corporate-action events: AAPL, AMZN, AVGO, CMCSA, GOOG, NEE, NVDA ×2, TSLA ×2 and WMT splits; GE's 1:8 reverse split; spin-offs (GE 2023 and 2024, T 2022); and the META identifier change. Example: NVDA on 2024-06-10 has `feat_28` −0.894 raw (winsorised to −0.244 in the panel), `feat_29` −0.886, `feat_36` −0.868. The targets use CRSP total returns and are unaffected, except for D2. | S11, `S11_split_or_identifier_events.csv` |
| D4 | M | **BKM moments.** (i) The quartic integrand has +4x³ where BKM has −4x³, with x = ln(K/S) (`aggregated_features.py:333`). (ii) e^{rτ} is applied inside `_bkm_contracts` and again in the skew/kurtosis formulas. (iii) The forward has no dividends. (iv) Expiries within ±5 days of the 30-day expiry are pooled. The textbook formula correlates 0.98–0.996 with the coded one for stocks and 0.91–0.92 for SPY *(sub-audit)*. | `aggregated_features.py:285-392` |
| D5 | M | **`feat_50` (opening-trade put/call ratio) is a copy of `feat_01`.** Contracts are keyed on (strike, dte, type). dte changes daily, so the previous open interest is never found (it defaults to 0) and every contract counts as "opening". Equal to `feat_01` on 99.5% of stock rows (59% of ETF rows). | `aggregated_features.py:213-222`, S11 |
| D6 | M | **GOOG has no earnings dates.** The Compustat file has no GOOG or GOOGL rows; `feat_43` for GOOG is the constant imputed value 46. | S11 |
| D7 | M | **The 50-contract minimum is not enforced.** `merge_sources.py:53` only sets a `thin_options_day` flag that nothing reads. 3,872 flagged ticker-days (3,870 of them in the modelling panel) keep their option features. | S11, `S11_thin_option_days.csv` |
| D8 | W | **The sample end is set by CRSP, not the option data.** The option snapshot ends 2025-12-12..22 for every ticker and starts by 2013-01-02 for 59 of 60 tickers (META 2021-07-08). CRSP ends 2024-12-31; the config has start 2016-01-01 and end 2025-12-31. The paper's "snapshot ends [Dec 2024]; extending would require a paid vendor" is wrong. A 2013–2025 sample needs only a CRSP re-pull. | S11, `S11_option_snapshot_coverage.csv` |
| D9 | W | **`repr_grid_raw` is not "the same grid".** It is a moneyness × DTE interpolation with nearest-neighbour fill (Section 7); "model-free / no fit" overstates it. | `iv_surface_features.py:18-78` |
| D10 | W | **IVs are vendor-supplied** (Section 6). | `clean_options.py:42-51` |
| D11 | W | **Models see 133,773 rows, not 134,368.** The inner join with the legacy split file drops its 10 buffer dates. | S11 |
| D12 | M | **Volatility target joins non-adjacent days.** RV is built after the split join, so 480 five-day RV windows sum across the buffer gaps. | S11, `run_vol_matrix.py:71-81` |
| D13 | W | **Sector-ETF mapping is not current GICS.** `feat_42` uses the `settings.yaml` map, which gives META, GOOG and NFLX the XLK return. Only `sector_neutral_ic.py` relabels to current GICS (Appendix C). | `clean_stocks.py:165-174` |
| D14 | I | **No end-to-end data command.** Downloads need a WRDS login. No code produces the IBES parquets. The option download depends on `static.philippdubach.com`; the local raw snapshot is the only copy *(sub-audit)*. | `src/data/*`, `scripts/download_new_tickers.py` |

### Preprocessing and models

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| P1 | B | **Preprocessing look-ahead.** Winsorising (0.1/99.9% quantiles) and imputation (ticker median → global median → 0) are fitted **once** on the legacy split's train block (2016-01-04..2021-05-18) and saved into `panel_unnormalized.parquet`, which every walk-forward window reuses. Windows whose training ends before May 2021 therefore use statistics from their own test periods. Only ElasticNet's `StandardScaler` is fitted per window. The paper says all preprocessing is fitted on the training block. | `run_pipeline.py:131-146`, `src/preprocessing/{winsorize,impute}.py` |
| P2 | M | **Surface columns are never preprocessed.** The 34 Set-D columns and the 20 raw-grid columns are neither winsorised nor imputed, because those steps only touch `feat_*`. Their NaNs (median 1.9%, up to 3.7%) become 0 in `run_full_matrix.py:422`, i.e. an IV or SVI parameter of zero. | S11 |
| P3 | B | **OLS is numerically rank-truncated.** `LinearRegression` is fitted on float32 X, and sklearn's lstsq cutoff max(n,p)·eps₃₂ keeps 6–8 of 80–114 directions (A, repr_svi, D, repr_grid) and **1** of 93/108 (B, C). The float32 re-fit reproduces the saved OLS IC in 17/18 window-cells (the remaining one differs by about 1e-6). In float64 the fit has full rank and B ≠ C (window 15: 0.061 vs 0.075). Affects the OLS column of Table 3 (both targets), all of Appendix A's explanation, the §4 OLS narrative ("collinearity effect", "overfits"), the dissociation caption, and the injection text about OLS. | `run_full_matrix.py:194-198,405`; `ols_rank_check.csv` |
| P4 | B | **ElasticNet constant forecasts are silently dropped.** In **578 of 1,593** EN return windows (40 of 54 cells) EN picks the intercept-only model; the window IC is NaN and is dropped from the mean and t. Five of the seven return cells that clear t>3 are EN cells resting on 6–17 of 32 windows. Counting a constant forecast as IC = 0, only **3 of 162** cells clear t>3: EN/B/rolling/3d (t 3.10), XGB/A/expanding/1d (3.64) and XGB/B/rolling/1d (3.06). EN/D/rolling/5d becomes 0.0103 (t 2.31); EN/A and EN/repr_bkm become t 1.42. The volatility grid has no NaN windows. FFNN has 26 NaN windows in 11 cells. Also affects the meta-analysis (it drops those windows), the power SE, and probably the EN injection curve, whose per-window values were not saved. | S3, `S3_constant_prediction_windows.csv` |
| P5 | B | **The long/short book is built from ties.** In constant-forecast windows the backtest sorts tied predictions, which yields an arbitrary basket (e.g. 2016-10-03: long AAPL, ORCL, LLY, LMT, LOW, MA; short GE, GOOG, GS, HD, HON, XOM). In the headline cell **278 of 404 periods** are such periods. They compound ×2.00 (arithmetic Sharpe 1.12); the 126 informative periods compound ×2.62 (Sharpe 1.24). Together ×5.23, i.e. $5,228. | S3, `S3_headline_ls_by_window_type.csv` |
| P6 | M | **Linear models use only 80% of each training window.** The purged training block is split 80/20 by date; OLS and EN are fitted on the first 80%, and the last 20% serves only as XGBoost's early-stopping fold. Not disclosed. This is not a leak. | `run_full_matrix.py:158-166,184-214` |
| P7 | W | **Early-stopping fold is not purged.** XGBoost early-stops on the last 20% of the training block, which is not purged from the 80% fitting part (labels overlap by h). This touches only train/validation, never test. | same |
| P8 | I | **Neural baselines are unseeded.** `set_seed` is called only in `run_pipeline.main()`, so the walk-forward FFNN and TFT runs are unseeded and ran on MPS. FFNN receives unstandardised features; TFT uses z-scores fitted on the legacy split. Their 36 Sharpe ratios enter SR0 and cannot be reproduced. | `run_pipeline.py:761`; `run_full_matrix.py:232-264,414,431` |
| P9 | W | **The TFT run crashed.** It was launched for 6 sets × 2 schemes × 3 horizons and finished only sets A–C at h = 1; the log ends mid-run. The paper presents "sets A–C at the one-day horizon (6)" as the design. | `results/logs/full_matrix_run.log:10112-11420` |

### Statistics

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| S1 | B | **IC definition.** The benchmark IC is a pooled per-window Spearman (Section 3), not "the cross-sectional rank correlation"; other sections use daily cross-sectional ICs. | Section 3 |
| S2 | W | **HAC** is as described (Section 4), except that NaN windows are dropped (see P4). | `posthoc_stats.py:37-54` |
| S3 | W | **The 0.0064 SE includes the neural cells.** It is the median over 198 cells including the 36 FFNN/TFT cells. On the 162-cell OLS/EN/XGB grid the median is 0.0068, giving a threshold of 0.0204 (not 0.0192) and names needed of 1,567 / 2,786 / 6,269 for lifts 0.004 / 0.003 / 0.002 (paper: 1,382 / 2,457 / 5,529). | S2, `power_analysis.py:23` |
| S4 | B | **The VRP split is per ticker-day** (Section 5), yet the paper calls it "high-VRP days" and compares it with a per-date VIX split. | S8 |
| S5 | B | **The regime bootstrap is not a block bootstrap.** The "date-block bootstrap" resamples single daily ICs i.i.d., and resamples the high and low buckets independently (`return_extras.py:58-73`), even though 3- and 5-day outcomes overlap. With a moving-block bootstrap joint over calendar days (L = 10 / 21 / 63):<br>• the high-VRP CI excludes 0 in **3 / 3 / 2 of 4** cells (published: 4/4)<br>• the high−low difference beats Bonferroni in **2 / 2 / 2 of 4** (published: 3/4)<br>• the VIX difference p-values are 0.995, 0.916, 0.812 and **0.128** (paper: all > 0.8) | S8, `S8_regime_bootstrap.csv` |
| S6 | W | **The unconditional baseline isn't "near zero".** The unconditional IC of the four regime cells averages about 0.008. | S8 |
| S7 | W | **The meta-analysis includes 28 FFNN/TFT cells** (172 cells in total). With OLS/EN/XGB only: +0.0032, CI [+0.0005, +0.0062], p = 0.023 (published: +0.0029, [−0.0000, +0.0059], p = 0.051). Per-window lifts with a NaN IC are dropped. | S6 |
| S8 | W | **"More than ten SEs" for volatility.** 0.0362 / 0.0035 = 10.2 uses ddof 0 across 12 overlapping cells; with ddof 1 it is 9.8. The per-cell paired HAC t of per-window D−A is 5.7–12.4 (median 7.7), so the result holds but the wording does not. | S5 |
| S9 | W | **Grid vs raw-grid medians.** The paper's "median signed difference ≈ 0.0008" is from `referee_fixes.txt`: XGBoost only, daily-IC definition, 6 cells. Under the benchmark definition across all 18 cells the median is +0.0001 (XGBoost only: +0.0022); the max \|diff\| is 0.0251 (EN rolling 5d). | S7 |
| S10 | W | **Injection calibration.** The oracle IC is ≈ 1.22δ (Section 8). The recovered ICs also drop NaN windows. | S12 |
| S11 | W | **Sector-neutral check.** It reproduces (XGB only, pooled per 63-day block, current-GICS relabel at analysis time). The ETFs form their own three-name "sector". The max \|dIC\| of 0.0104 slightly exceeds "about one HAC SE" (0.006–0.008). | `sector_neutral_ic.txt` |

### Economics

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| E1 | B | **The Sharpe isn't the stated formula.** It is geometric (Section 8). The arithmetic μ/σ·√(252/h) for the headline is **1.0252**, not 1.1504. Under arithmetic Sharpes, SR0 over 198 configurations is 1.1010 and **0** configurations clear it (published: 1.1529, 2 clear). | S10, `backtest.py:90-101` |
| E2 | B | **The PBO comes from the wrong group.** 0.571 is the expanding/3d group (32 trials: OLS/EN/XGB × 9 sets + 5 FFNN). The headline cell's own group, rolling/5d, gives **0.429**. All groups: 0.243 / 0.571 / 0.286 (expanding h1/h3/h5) and 0.414 / 0.743 / 0.429 (rolling). The standard rank/(N+1) convention lowers each, e.g. to 0.514 and 0.400. §4.7 rests its conclusion on "0.571 > ½". | S9, `posthoc_stats.py:119-153` |
| E3 | M | **ETF handling is inconsistent.** The grid's long/short trades all 60 names, so the headline 1.15, $5,228, SR0 and PBO describe the all-names book. The ETF-free book exists only in `etf_free_econ.txt` (1.07, $4,585, geometric). §4.7 says the ETFs are removed and then reports the all-names figures. "IC moves by at most 0.001" is false for EN/A/rolling/5d (+0.0021). | `etf_free_econ.txt` |
| E4 | M | **The SR0 trial family and the DSR label.** N = 198 = 162 + 30 FFNN + 6 TFT (the subset that survived the crash, P9). The family leaves out other long/short variants in the repository: the 25%-spread variants, the long-flat hurdle rows, and the single-split `tft_svi` runs (1-day long/short Sharpe 1.13). SR0 without TFT is 1.0983, which the headline would clear (+0.052); with OLS/EN/XGB only it is 1.0250 (+0.126). "Deflated Sharpe" is only the SR0 benchmark: the only DSR probability anywhere is for the headline (≈ 0.497, `referee_fixes.txt`), computed from a hard-coded Sharpe. | S10; `referee_fixes.txt:190-198`; `tft_svi_strategy_summary.csv` |
| E5 | W | **Cost assumptions.** The cost model is as described in Section 8; the spread is assumed, not measured. | `frictions.py` |
| E6 | W | **The dealer-gamma result is computed outside the README map.** +0.0022, CI [−0.021, +0.024], p = 0.84 comes from `referee_fixes.py` (XGB/D/expanding/3d, i.i.d. date bootstrap), and that script is not in the README's number→script map. | `referee_fixes.txt:148-160` |

### Figures, appendices, reproducibility

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| R1 | I | **Figures are written outside the paper folder.** `make_paper_figures.py` writes the six PDFs to `../EasyChair3.5/figures` if that folder exists, otherwise to `results/figures/`; LaTeX looks in `paper/figures/`. None of the six PDFs is in the repository. The script runs cleanly in a scratch copy (12 PDFs) *(sub-audit)*. | `make_paper_figures.py:17-21` |
| R2 | I | **Some figures plot hard-coded values.** `fig_regime_return` (VRP cells), `fig_meta` (per-representation lifts, pooled CI) and `fig_regime_ci` (dealer gamma) use constants copied from text outputs. | `make_paper_figures.py:112-127,262-270,302-312` |
| R3 | M | **TreeSHAP is not run on the benchmark models.**<br>• One XGBoost per target, trained on dates before 2021-05-18 (no purge) and attributed on later dates.<br>• 400 trees, lr 0.02, no early stopping, vs the benchmark's 500 / 0.01 / early stopping 50.<br>• **No ticker dummies**: `treeshap.py:42` looks for the prefix `ticker_`, but the encoder writes `tkr_`.<br>• The paper's volatility ordering is wrong. The actual order is VIX 16.4%, earnings 9.1%, iv_25dc_6m 8.7%, iv_25dc_3m 7.9%, risk-free 6.3%, …, svi_a 2.4% (10th).<br>• The return list omits hvol_30d (4.7%). | `treeshap.py:20-60`, `treeshap.txt` |
| R4 | I | **The reproducibility check is narrower than the paper claims.**<br>• `reproducibility_check.py` tests 3 return cells, all on Set D (one of them XGBoost), with no OLS or volatility cells.<br>• It compares fresh vs saved runs only through cell means printed to 5 decimals, while the paper claims per-window equality.<br>• The saved `reproducibility.txt` comes from an older version of the script, run from `/Users/pia/Desktop/guided-research/...`, and the current script has never been run.<br>• This audit's EN re-fit does reproduce per-window equality (Section 3). | `reproducibility_check.py:35-39,112-137` |
| R5 | I | **The public repository has no data code.** The `.gitignore` rule `data/` also ignores `src/data/`, so the anonymized GitHub repository has no `src/data/` (checked via the GitHub API listing). | `.gitignore:2` |
| R6 | I | **No run commands are recorded.** The `run_full_matrix.py` defaults (6 models including TabNet, 5 sets) do not reproduce the grid. The log shows three chained `--resume` runs (Section 10). | `full_matrix_run.log:7,4413,10112` |
| R7 | I | **No version pins.** `requirements.txt` has only `>=` bounds and there is no lockfile. Bit-exactness depends on versions: sklearn's lstsq cutoff for OLS, and xgboost's default tree method (no `tree_method` is set). | `requirements.txt` |
| R8 | I | **A destructive script.** `scripts/rebuild_for_60_tickers.sh` deletes `results/tables/`, `data/features/` and the checkpoints (`rm -rf`, lines 41-59). Do not run it. | line 59 |
| R9 | I | **Logs point to a folder that no longer exists** (`~/Desktop/guided-research/...`), so the code that produced the outputs cannot be diffed against the current code. | `results/logs/*` |
| R10 | W | **Appendix A reproduces but has no script.** All six cells match at 6 decimals and B/C at 9 decimals; the per-window max \|B−C\| is 1.03e-7 for returns and 4.4e-8 for volatility. No script prints these values and the README map has no entry. | S4 |

### What checks out

- **Reported numbers:**
  - Tables 3 and 4 reproduce exactly from the saved per-window files: values and t>3 marks 54/54 and 12/12, and `posthoc_ic_tstats.csv` agrees with a recomputation to 1e-16.
  - Volatility control: 158/162, max t 58.7, D lift 0.0362, repr_svi 0.0235, headline slice 0.044.
  - Power: 0.0192; 1,382 / 2,457 / 5,529 names; DT(30) ≈ 0.027; 17 names for 0.036.
  - Every text output the paper cites reproduces: injection, meta-analysis, regime, sector-neutral, ETF-free and economics, all as coded.
- **Walk-forward protocol:**
  - Window geometry matches the paper: expanding 27 windows (504-day minimum, 63-day step) and rolling 32 windows (189/63/63).
  - The last h training dates are purged, and the HAC lag is 3.
  - The hyperparameters match the footnote.
- **Features and targets:**
  - The feature-set counts match: A 21, B 34, C 49, candidate_6 6, D 55, and the repr_* sets 31/45/41/23, plus 59 ticker dummies.
  - The targets are forward compounded CRSP returns and realised volatility over t+1..t+h, demeaned per date with the ETFs included.
  - Set A's lagged realised volatility is strictly backward-looking.
- **Data and surface construction** *(sub-audit)*:
  - The option filters are implemented and the cleaned files show zero violations: OI ≥ 10, bid ≥ $0.05, K/S ∈ [0.80, 1.20], DTE ∈ [7, 180], IV ∈ [0.01, 5], highest-volume quote per date/strike/expiry/type.
  - The SVI and (Δ, τ) grid construction is as described.

## 10. How the BIR grid was run (reconstructed from the log)

```bash
python scripts/run_full_matrix.py --models OLS ElasticNet XGBoost \
  --sets A B C candidate_6 D repr_svi repr_grid repr_grid_raw repr_bkm   # 162 cells
python scripts/run_full_matrix.py --models FFNN --resume                 # 30 cells (default 5 sets)
python scripts/run_full_matrix.py --models TFT --sets A B C candidate_6 D repr_svi --resume   # crashed after 6 cells
python scripts/run_vol_matrix.py
python scripts/posthoc_stats.py && python scripts/vol_dissociation.py && python scripts/power_analysis.py
python scripts/return_extras.py && python scripts/signal_injection.py && python scripts/sector_neutral_ic.py
python scripts/etf_free_econ.py && python scripts/referee_fixes.py && python scripts/treeshap.py
python scripts/make_paper_figures.py
```

## 11. Decisions needed before Task 0b

1. **WRDS access.** Can CRSP be re-pulled **by PERMNO** (with `cfacpr`) and the Compustat earnings by GVKEY? This fixes D2, D3 and D6, and is the only route to D8 (extending the sample).
2. **META.** Choose one: (a) PERMNO 13407 from the first option date (2021-07-08); (b) drop META before 2022-06-09; (c) drop META entirely.
3. **Sample window.** Keep 2016–2024 for comparability with the BIR version, extend to 2013–2025 for power, or report both.
4. **50-contract rule (D7).** Enforce it (option features missing on thin days), or remove the claim from the paper.
5. **Constant forecasts (P4/P5; not covered by the Task 0b spec).** Recommended: a constant forecast counts as **IC = 0** for each affected date under the daily cross-sectional definition, and the book is **flat** (no position) when predictions are tied. Also report the share of null-model windows. The alternative, dropping them, has to be disclosed and inflates EN.
6. **Surface missing values (P2).** Proposed: per-window training-block imputation, as for `feat_*`, plus a missingness indicator for the surface block. Never 0.
7. **SR0 / PBO family without FFNN/TFT.** Proposed: the 162 OLS/EN/XGB long/short configurations at the stated cost, with PBO computed per scheme × horizon group and over the full family, all with the arithmetic Sharpe.

## 12. Number → script map for this task

Every number in Sections 3–9 is printed in `t0_refits.txt` (EN reproduction, OLS ranks) or `t0_checks.txt` (sections S1–S13), with the matching CSV named alongside. To regenerate, run `python revision/T0/t0_refits.py && python revision/T0/t0_checks.py`; neither script writes outside `revision/out/T0/`.
