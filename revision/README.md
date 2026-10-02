# Revision workspace (JFM resubmission)

The plan is `paper/steps.md`; the authors' amendments of 2026-09-30 are copied verbatim below. All revision code lives in `revision/`, and outputs go to `revision/out/T<k>/`. BIR-era files under `results/`, `data/`, `src/`, `scripts/` and `paper/` are never overwritten.

## Status

| Task | Status | Outputs |
|---|---|---|
| Freeze of the BIR-era state | done 2026-09-30 | git tag `bir-snapshot-2026-09-30` (commit `b47f351`, branch `main`); read-only APFS clone `~/Desktop/options_research_BIR_snapshot_2026-09-30/` with `MANIFEST.sha256` |
| T0 audit and definitions | **done, frozen** | `revision/out/T0/README.md`, `findings.csv`, `t0_*.txt`, `S*_*.csv`; code `revision/T0/` |
| T0b correctness rebuild | **done 2026-10-01 (23:36); stopped at the decision gate**: canonical grid 324 cells, unfiltered robustness, economics, EN convergence audit; includes the additional blocking correction R9 (future-event features removed) | `revision/out/T0b/README.md`; code `revision/T0b/` |
| T1 paired marginal-lift inference | **done 2026-10-02; stopped for review** | `revision/out/T1/README.md`; code `revision/T1/` |
| T2–T5 | not started (gate: T1 review) | — |
| Manuscript rewrite (Phase 2) | not started (gate: T1–T5 review) | — |

Work happens on git branch `revision`; `main` stays at the BIR snapshot.

## Standing rules added during Task 0b (authors, 2026-10-01)

- No predictor may use a future event date unless a point-in-time record shows it was known at t. `feat_43`, `feat_44`, `feat_45` and `feat_51` are therefore excluded from every canonical feature set, with no replacements (`revision/out/T0b/feature_sets_canonical.yaml`).
- **Task 4:** the earnings-proximity conditional test is not part of the primary predictive analysis. If shown at all, it is labelled an ex-post descriptive diagnostic, unless a point-in-time earnings calendar is obtained. The VRP and VIX analyses stay.
- META is excluded until PERMNO 13407 can be verified across the ticker change (WRDS).
- Primary sample 2016–2024; 50-contract rule enforced, with an unfiltered robustness run; daily cross-sectional IC; constant predictions score zero; flat book when all predictions tie; training-window-only preprocessing; OLS in float64.

## Regenerating Task 0

```bash
python revision/T0/t0_refits.py         # EN headline re-fit + OLS rank check (~20 s)
python revision/T0/t0_checks.py         # all other T0 numbers (~15 s)
python revision/T0/t0_findings_csv.py   # findings.csv from the README register
```

## Authors' instructions, 2026-09-30 (verbatim)

> Task 0 audit is complete. Do NOT start Task 1 yet.
>
> The audit uncovered blocking correctness issues that invalidate the assumption that the existing saved predictions can be used as the canonical basis for Tasks 1–5.
>
> ## Step A — Finish and freeze Task 0
>
> 1. Write the complete audit to: `revision/out/T0/README.md`. Include every finding, including unfavorable ones.
> 2. Do not modify the manuscript yet.
> 3. Do not overwrite any existing BIR-era results.
> 4. Create an immutable backup/version of the current project state before making fixes. If practical, initialize local version control or make a full timestamped backup so the rejected BIR version can always be reconstructed.
> 5. Stop after writing the Task 0 report and confirm the files created.
>
> # New Task 0b — Correctness rebuild
>
> After Task 0 is documented, create a new Task 0b before Task 1. The purpose of Task 0b is to establish a clean canonical dataset, feature pipeline, walk-forward procedure, predictions, and evaluation definition.
>
> ## 0b.1 Fix the data layer
>
> Audit and correct, at minimum: META security mapping; split-adjusted stock prices/features and returns; BKM kurtosis formula/sign; `feat_50` duplication/problem; GOOG earnings-data issue; the claimed minimum-50-contract ticker-day rule (either actually enforce it, or remove the claim from the paper); verify all other ticker mappings for obvious identifier collisions; correct the documented data sources (CRSP/Compustat/WRDS where actually used, Yahoo only where actually used).
>
> Do not silently correct data. Record before/after row counts and every rule in `revision/out/T0b/data_audit.csv` and `revision/out/T0b/README.md`.
>
> ## 0b.2 Fix walk-forward preprocessing leakage
>
> All preprocessing that learns parameters must be fitted separately inside each training window. This includes, where applicable: winsorization; imputation; scaling/standardization; any learned preprocessing transformation. No future observations may influence an earlier window.
>
> For each test window: 1. fit preprocessing using only the permitted training block; 2. transform train/validation/test with those fitted parameters; 3. train the model; 4. predict the test block.
>
> Surface-feature missing-value treatment must also be documented explicitly. Do not silently convert all missing values to zero unless this is a deliberate, justified design decision.
>
> ## 0b.3 Fix numeric/model issues
>
> Run OLS in float64, not float32. Preserve the same model specifications otherwise. Pin package versions and seeds. Ensure EN and XGB predictions are saved; save OLS predictions as well if inexpensive. Save predictions with: model, set, scheme, horizon, window, date, ticker, prediction, target.
>
> ## 0b.4 Choose one canonical IC definition
>
> The manuscript claims cross-sectional rank IC, but the old benchmark computes one pooled Spearman correlation over all ticker-days in a test window. For the revised paper, use the finance-standard interpretation: 1. on every test date, compute cross-sectional Spearman correlation across stocks; 2. average those daily ICs within the test window; 3. use the resulting per-window IC series for HAC inference.
>
> Use this same definition consistently in: benchmark tables; paired D−A lift; power analysis; VRP conditioning; volatility control; meta-analysis; signal injection.
>
> Before committing to this definition, produce a small comparison for the two headline cells showing old pooled IC versus daily-cross-sectional IC. Document the difference. The revised paper should not mix the two definitions.
>
> ## 0b.5 Re-run the canonical benchmark
>
> After all corrections, rerun the main return and volatility grid for OLS, ElasticNet, XGBoost using the corrected pipeline. Do not initially rerun FFNN/TFT. They are not necessary for the central empirical paper unless later needed. Save all new results under a new revision path. Never overwrite the old BIR result files.
>
> For each old headline number, produce a before/after comparison table. At minimum include: Set A; Set D; repr_svi; repr_grid; repr_grid_raw; repr_bkm; both schemes; h = 1, 3, 5; return and volatility targets.
>
> ## 0b.6 Rebuild economics correctly
>
> After corrected predictions exist: use the standard arithmetic Sharpe: mean / sd × sqrt(252/h); recompute the entire searched strategy set consistently; recompute the expected-maximum/deflated-Sharpe benchmark correctly; compute PBO for the appropriate search family and clearly define that family; make ETF inclusion/exclusion consistent with the manuscript; report the tradeable headline using stocks only if that is the stated design. Do not rely on the old 1.15 Sharpe, SR0, or PBO values.
>
> ## 0b.7 Fix bootstrap terminology and implementation
>
> The old regime procedure resampled individual dates, so do not call it a block bootstrap. For overlapping 3-day and 5-day outcomes, use an actual time-block/bootstrap procedure with a pre-specified block length at least as large as the return horizon. Document the exact bootstrap and use it consistently in subsequent Task 4 analysis.
>
> ## 0b.8 Fix reproducibility infrastructure
>
> Also correct: `.gitignore` so source data-processing code is not accidentally excluded; figure output paths; scripts that use hard-coded figure numbers; explicit reproducible run commands; package/version pins; number→script mapping; repository documentation. Do NOT run any script that deletes the old `results/tables/`.
>
> # Decision gate after Task 0b
>
> When Task 0b finishes, STOP. Send back: 1. Task 0 README; 2. Task 0b README; 3. data before/after table; 4. old vs corrected headline results; 5. pooled-IC vs daily-cross-sectional-IC comparison; 6. corrected return benchmark; 7. corrected volatility benchmark; 8. corrected economics summary; 9. paths to saved OLS/EN/XGB predictions. Do not start Task 1 until these corrected results have been reviewed.
>
> # After Task 0b is approved
>
> Then continue with the original revision plan, but always use the corrected Task-0b outputs:
> Task 1: paired marginal D−A inference.
> Task 2: evaluation-side N scaling using corrected predictions and re-demeaning/re-ranking inside each sampled cross-section.
> Task 3: borrow-cost / borrow-pressure analysis and economics sensitivity.
> Task 4: VRP analysis, explicitly treated as cross-sectional conditioning if the split remains ticker-day level. Also construct a separate date-level VRP state if economically useful.
> Task 5: realistic surface-carried injection, with the Set-A-residualized carrier as the main incremental-power experiment.
>
> Only after Tasks 1–5 are reviewed should the manuscript be rewritten. Paper 1 remains an empirical options/finance paper. The general ML/power framework, theory, large-N Monte Carlo, and broader methodological contribution remain reserved for Paper 2.

## Authors' earlier amendments to the plan, 2026-09-30 (verbatim, original language)

These amend Tasks 2–5, Phase 2 and the journal choice in `paper/steps.md`. They will be merged into `steps.md` when the authors confirm.

> **Planda yapılacak değişiklikler**
>
> 1. **Task 2 – N scaling**
>    - Bunu “full pipeline power scaling” olarak değil, **evaluation-side cross-sectional SE scaling** olarak tanımla, çünkü saved predictions kullanıyoruz.
>    - \(V(N)=A+B/N\) modelini **A ≥ 0 ve B ≥ 0 constraint** ile fit et.
>    - N > 57 sonuçlarını açıkça “model-based extrapolation” olarak etiketle.
>    - Mümkünse sadece primary iki cell için N={20,40,57} ile optional retraining validation yap; zorunlu değil.
> 2. **Task 3 – borrow cost**
>    - “Our universe is almost always general collateral/easy-to-borrow” sonucunu önceden varsayma. Bunu **test edilecek hipotez** olarak yaz.
>    - Main borrow-pressure measure’da gelecekte gerçekleşmiş dividend bilgisi kullanma. Sadece date-t’de bilinebilen/trailing dividend bilgisi kullan.
>    - Put-call parity measure’a başlangıçta **option-implied borrow/carry wedge** veya **borrow-pressure proxy** de.
>    - Method B’deki IV-spread → fee formülünü yalnızca hakemli bir kaynakta exact olarak doğrulayabiliyorsan kullan. Doğrulanamıyorsa Method B’yi annualized fee estimate olarak kaldır; yalnız descriptive IV-spread diagnostic olabilir.
>    - “easy-to-borrow” sonucu ancak Task 3 bunu gerçekten gösterirse manuscript’e girsin.
> 3. **Task 4 – high VRP**
>    - En önemli estimand: \(\Delta IC = IC_D-IC_A\) ve özellikle \(\Delta IC_{highVRP}-\Delta IC_{lowVRP}\).
>    - Yeni dört primary D−A high-vs-low test için ayrı bir multiplicity family tanımla ve Holm/Bonferroni correction uygula.
>    - Task 0’da VRP split ticker-day level çıkarsa buna “market regime” deme; “cross-sectional VRP conditioning” olarak adlandır.
>    - 2020 robustness için hem NBER/COVID recession window’u hem de **tüm 2020’yi çıkararak** sonucu raporla.
> 4. **Task 5 – realistic injection**
>    - Real surface feature ile yapılan injection’a ek olarak bir **A-residualized carrier** oluştur: \(z^\perp=z-\widehat E[z|A]\).
>    - Residualization modelini sadece training data’da fit et, test data’ya uygula; leakage olmasın.
>    - Bu residualized carrier, gerçek incremental D−A detection testinin primary versiyonu olsun.
>    - 45-degree “perfect recovery” line’ını kaldır. Onun yerine injected signal, recovered paired lift ve t=3 crossing göster.
> 5. **Phase 2 wording**
>    - Task 3 sonucu gelmeden “easy-to-borrow”, “short-sale frictions are weakest” gibi ifadeler manuscript’e ekleme.
>    - Şimdilik yalnızca “large, liquid stocks” de.
>    - “This null reflects limited power” gibi kesin ifadeleri “is consistent with limited power” olarak yumuşat.
>    - Paper 1 yalnız empirical finance/options paper olarak kalsın. General ML theory, Monte Carlo ve genel power-aware methodology ayrı Paper 2’ye saklansın.
> 6. **Journal**
>    - Primary target şimdilik **Journal of Futures Markets**.
>    - Secondary journal Phase 1 sonuçlarından sonra seçilecek.
>    - Zorunlu submission fee/APC gerektiren venue’ları önceliklendirme.
>
> **Çalışma sırası** *(summary of the authors' work order, not verbatim)*: Task 0 → dur ve raporla; Task 1 (paired lift; Holm ile t>3’ü karıştırma) → dur; Task 2 (önce iki primary cell; constrained A+B/N ve log-log) → dur; Task 3 (flat borrow 25–300bp, break-even fee, legs ayrı; no-look-ahead dividend; 3a/3b) → dur; Task 4 (IC_A, IC_D, IC_D−IC_A, high−low; 2020 ve tüm-2020 exclusion; VRP-vs-VIX horse race; dispersion control) → dur; Task 5 (raw carrier ve primary olarak Set-A-residualized carrier; D ve A retrain) → dur; Phase 2 ancak Task 0–5 birlikte değerlendirildikten sonra. Her yeni rakam number→script tablosuna; hiçbir olumsuz sonuç gizlenmeyecek, threshold sonuçlara bakıldıktan sonra değiştirilmeyecek.
