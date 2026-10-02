# Task 1 — Paired inference for the marginal return lift over Set A

**Status:** complete, 2026-10-02; stopped and waiting for review. Tasks 2–5 have not started, and the manuscript is unchanged.

**Inputs:**
- the corrected Task 0b predictions (`revision/out/T0b/predictions/`, git tag `task0b-corrected-baseline-2026-10-02`)
- the canonical **daily cross-sectional** rank IC: per window, the mean of daily Spearman ICs across names, with constant predictions scoring 0

**Recomputation check:** every per-window IC was recomputed from the saved predictions and matches Task 0b's `benchmark/windows.csv` to within 9.9e-17 (9,558 windows).

## Definitions

- **Estimand:** **ΔIC_w(X) = IC_w(X) − IC_w(A)**, for the same target, model, scheme, horizon and test window w.
- **Per cell** (`T1_lift_table.csv`):

  | Quantity | Definition |
  |---|---|
  | mean IC(A), mean IC(X) | with the HAC SE of each |
  | correlation | over windows, of IC_w(X) and IC_w(A) |
  | mean ΔIC | over windows |
  | SE(ΔIC) | Newey–West HAC, Bartlett kernel, lag ⌊n^(1/3)⌋, on the per-window ΔIC series (n = 27 expanding / 32 rolling windows) |
  | paired t | mean / SE |
  | DT_Δ | 3 × SE, the t > 3 detection threshold |
  | MDE80_Δ | (3 + 0.8416) × SE |
  | raw p | two-sided, Student t with n − 1 df (the normal p is in `p_raw_normal`) |
  | adjusted p | **Holm** within the pre-specified family |

- **Hurdle vs multiplicity.** These are separate criteria, reported separately:
  - the **t > 3 hurdle** (`clears_t3`, positive lift)
  - **multiplicity-adjusted significance** (`holm_sig_5pct`: Holm p < 0.05, family-wise, two-sided)
- **Degenerate cells.** Four cells (ElasticNet, expanding, h1, on Sets C, D, repr_svi and repr_bkm) have both fits intercept-only in every window, so ΔIC = 0 throughout and SE = 0. Their t is undefined and their p is set to 1, since they carry no evidence against H0. They stay in the family.

**Families**, fixed before any result was computed; Holm is applied within each:

| Family | Tests | Content |
|---|---|---|
| **primary** | 72 | nested sets C, D, repr_svi, repr_grid, repr_grid_raw, repr_bkm × {ElasticNet, XGBoost} × 2 schemes × 3 horizons (the plan's nested regularised family) |
| ols | 36 | the same nested sets with OLS |
| nonnested | 36 | B and candidate_6 vs A, all three models. **These are not marginal lifts:** B and candidate_6 do not contain Set A |
| vol_* | 72 / 36 / 36 | the same three families on the volatility target (positive control) |

## Results: return target

**Primary family** (72 tests): ΔIC (paired t); † = t > 3.

| Set | Model | exp h1 | exp h3 | exp h5 | roll h1 | roll h3 | roll h5 | min Holm p |
|---|---|---|---|---|---|---|---|---|
| C | EN | 0 (degenerate) | -0.0035 (-0.45) | +0.0058 (+0.75) | -0.0021 (-0.69) | +0.0061 (+2.08) | -0.0054 (-0.65) | 1.000 |
| C | XGB | -0.0036 (-0.78) | -0.0025 (-0.24) | +0.0085 (+1.17) | +0.0069 (+1.36) | -0.0060 (-0.77) | -0.0093 (-1.20) | 1.000 |
| D | EN | 0 (degenerate) | -0.0016 (-0.19) | +0.0066 (+0.71) | -0.0030 (-0.83) | +0.0055 (+1.50) | +0.0016 (+0.27) | 1.000 |
| D | XGB | +0.0005 (+0.08) | -0.0045 (-0.51) | +0.0023 (+0.30) | +0.0006 (+0.12) | +0.0018 (+0.41) | -0.0042 (-0.38) | 1.000 |
| repr_svi | EN | 0 (degenerate) | -0.0068 (-0.96) | +0.0045 (+0.49) | -0.0031 (-0.97) | +0.0024 (+0.67) | -0.0019 (-0.37) | 1.000 |
| repr_svi | XGB | +0.0021 (+0.53) | -0.0090 (-1.28) | -0.0067 (-0.67) | -0.0016 (-0.48) | -0.0052 (-1.09) | -0.0074 (-0.87) | 1.000 |
| repr_grid | EN | +0.0014 (+1.12) | -0.0047 (-0.47) | +0.0091 (+0.96) | +0.0004 (+0.13) | +0.0063 (+1.67) | -0.0014 (-0.20) | 1.000 |
| repr_grid | XGB | +0.0037 (+0.60) | -0.0035 (-0.37) | +0.0032 (+0.40) | +0.0022 (+0.66) | +0.0088 (+2.12) | -0.0027 (-0.35) | 1.000 |
| repr_grid_raw | EN | +0.0004 (+1.02) | -0.0038 (-0.61) | +0.0088 (+1.09) | -0.0013 (-0.55) | +0.0032 (+1.04) | +0.0042 (+1.38) | 1.000 |
| repr_grid_raw | XGB | -0.0018 (-0.39) | -0.0094 (-0.94) | +0.0044 (+0.68) | +0.0014 (+0.29) | +0.0111 (+1.71) | +0.0032 (+0.43) | 1.000 |
| repr_bkm | EN | 0 (degenerate) | +0.0001 (+1.08) | -0.0011 (-1.29) | -0.0000 (-1.08) | -0.0016 (-1.20) | -0.0067 (-1.11) | 1.000 |
| repr_bkm | XGB | -0.0022 (-0.40) | -0.0022 (-0.36) | +0.0026 (+0.41) | +0.0002 (+0.05) | +0.0005 (+0.14) | -0.0022 (-0.41) | 1.000 |

| Criterion | Primary family |
|---|---|
| t > 3 | **0 of 72** |
| Holm p < 0.05 | **0 of 72** (smallest Holm p 1.000) |
| raw p < 0.05 | 2: EN / C / rolling h3 (+0.0061, t 2.08, p 0.045) and XGB / repr_grid / rolling h3 (+0.0088, t 2.12, p 0.042); 72 × 0.05 = 3.6 are expected by chance |
| mean ΔIC | −0.0000 |
| t range | −1.29 .. +2.12 |

Per-set mean ΔIC over its 12 primary cells: D +0.0005, C −0.0004, repr_svi −0.0027, repr_grid +0.0019, repr_grid_raw +0.0017, repr_bkm −0.0011.

**OLS family** (36 tests): none clears t > 3 or Holm. The two cells with raw p < 0.05 are *negative* lifts (C / expanding h1 −0.0049; repr_bkm / expanding h3 −0.0013).

| Set | Model | exp h1 | exp h3 | exp h5 | roll h1 | roll h3 | roll h5 | min Holm p |
|---|---|---|---|---|---|---|---|---|
| C | OLS | -0.0049 (-2.07) | -0.0011 (-0.41) | -0.0004 (-0.12) | -0.0014 (-0.65) | +0.0006 (+0.24) | +0.0005 (+0.19) | 1.000 |
| D | OLS | -0.0016 (-0.88) | -0.0014 (-0.68) | -0.0003 (-0.10) | -0.0017 (-0.86) | -0.0012 (-0.37) | +0.0010 (+0.25) | 1.000 |
| repr_svi | OLS | -0.0023 (-1.27) | -0.0022 (-1.44) | -0.0014 (-0.55) | -0.0023 (-1.16) | -0.0026 (-1.13) | -0.0014 (-0.48) | 1.000 |
| repr_grid | OLS | -0.0008 (-0.46) | -0.0001 (-0.06) | -0.0005 (-0.19) | -0.0007 (-0.35) | +0.0000 (+0.01) | +0.0012 (+0.41) | 1.000 |
| repr_grid_raw | OLS | +0.0002 (+0.21) | -0.0009 (-0.55) | -0.0004 (-0.24) | -0.0011 (-0.95) | +0.0005 (+0.26) | +0.0005 (+0.21) | 1.000 |
| repr_bkm | OLS | -0.0011 (-1.55) | -0.0013 (-2.29) | -0.0012 (-1.70) | -0.0006 (-1.30) | -0.0004 (-0.96) | -0.0009 (-1.57) | 1.000 |

**Non-nested comparisons** (36 tests; not marginal lifts):

| Set | Model | exp h1 | exp h3 | exp h5 | roll h1 | roll h3 | roll h5 | min Holm p |
|---|---|---|---|---|---|---|---|---|
| B | OLS | -0.0010 (-0.25) | +0.0052 (+1.38) | +0.0082 (+1.71) | +0.0030 (+0.91) | +0.0072 (+2.28) | +0.0085 (+2.14) | 1.000 |
| B | EN | +0.0006 (+1.09) | +0.0027 (+0.27) | +0.0109 (+1.10) | +0.0014 (+0.47) | +0.0088 (+1.20) | +0.0024 (+0.34) | 1.000 |
| B | XGB | -0.0048 (-1.02) | +0.0010 (+0.11) | +0.0100 (+0.95) | +0.0097 (+2.10) | +0.0022 (+0.31) | -0.0010 (-0.16) | 1.000 |
| candidate_6 | OLS | +0.0031 (+0.85) | +0.0069 (+1.29) | +0.0096 (+1.69) | +0.0061 (+1.81) | +0.0076 (+1.77) | +0.0096 (+1.92) | 1.000 |
| candidate_6 | EN | +0.0025 (+1.22) | +0.0003 (+0.03) | +0.0210 (+1.71) | +0.0049 (+1.39) | +0.0164 (+3.17)† | +0.0187 (+2.15) | 0.124 |
| candidate_6 | XGB | -0.0058 (-1.03) | +0.0017 (+0.20) | +0.0076 (+0.85) | +0.0032 (+0.70) | +0.0106 (+1.66) | +0.0102 (+1.84) | 1.000 |

Exactly one cell clears the t > 3 hurdle: EN / candidate_6 / rolling h3 (+0.0164, t 3.17, raw p 0.0034). It is **not** multiplicity-significant (Holm p 0.124). This is the case where the two criteria disagree.

**Standard errors** (primary family, medians):

| | SE(ΔIC) | SE(IC_X) | SE(IC_A) | corr(IC_X, IC_A) |
|---|---|---|---|---|
| all | **0.0055** | 0.0054 | 0.0048 | 0.50 |
| ElasticNet | 0.0036 | 0.0043 | 0.0031 | 0.70 |
| XGBoost | 0.0063 | 0.0060 | 0.0055 | 0.40 |

- Median DT_Δ = 3 × SE = **0.0165**; median MDE80_Δ = **0.0211**.
- **Pairing does not make the lift's SE much smaller than the level SE**, because the two ICs correlate only about 0.5 across windows. The plan expected otherwise (P1). The paper's detection threshold of about 0.019 is therefore roughly right for the lift as well.
- Lifts of the size the paper discussed (about 0.003) are a factor of five or more below what a cell can detect.

## Results: volatility target (positive control)

| Family | t > 3 | Holm p < 0.05 | mean ΔIC | t range |
|---|---|---|---|---|
| primary | 63 of 72 | 63 of 72 | +0.0346 | +0.01 .. +22.16 |
| ols | 32 of 36 | 34 of 36 | +0.0160 | +0.50 .. +10.86 |
| nonnested | 27 of 36 | 27 of 36 | +0.0251 | +0.66 .. +7.67 |

The paired method detects incremental information whenever it is present.

## Interpretation

**No option representation adds statistically detectable incremental return information over Set A.**
- Of the 72 pre-specified nested tests, none reaches t > 3 and none is Holm-significant. Point estimates scatter around zero (mean −0.0000; the largest t is 2.12). The same holds for OLS.
- The only t > 3 return cell is a non-nested comparison: the six classical option signals *instead of* Set A, ElasticNet, rolling h3. It does not survive Holm, and it is not a marginal lift.
- The same paired procedure finds strong, multiplicity-significant volatility lifts in 63 of 72 cells. So the return null reflects the data, not a lack of power in the method relative to a known signal.
- At this universe size, a cell's lift must reach about 0.017 (3 × SE) to clear the hurdle; with 80% power, about 0.021.

## Deviations from the plan, and choices

- **p-values** use Student t with n − 1 df. This is conservative relative to the normal; normal-based p-values are also in the table.
- **Degenerate cells** get p = 1.
- **Added families:** the OLS and non-nested families and the volatility control were added as separate Holm families. The primary family is exactly the planned one: 72 tests.
- **Sign:** "clears t > 3" counts positive lifts, as in the paper's hurdle; p-values are two-sided.
- **Seeds:** none needed, since nothing is random. Run time about 50 s.

## Number → script map

| Numbers | Script | Output |
|---|---|---|
| all of the above | `revision/T1/t1_paired_lift.py` | `T1_lift_table.csv` (288 cells: return and volatility), `T1_window_ics.csv` (9,558 window ICs), `T1_summary.txt` |
