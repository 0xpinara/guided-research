# Diagnostic only — not canonical

Aborted Task 0b grid run (2026-10-01, 18:35–19:0x): 39 of 54 jobs finished before it was stopped.
Its feature sets still contained the future-event features feat_43 (days to next earnings),
feat_44 (earnings within 7 days), feat_45 (days to next ex-dividend date) and feat_51 (implied
earnings move), which use realised future event dates. These outputs must not be used as
canonical Task 0b results; they are kept only to show the effect of removing those features.
The canonical grid is in ../predictions and ../benchmark.
