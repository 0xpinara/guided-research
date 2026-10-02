"""Task 0b — robustness comparison: 50-contract rule enforced (primary) vs not enforced.

Both grids use the identical corrected pipeline; the only difference is the panel
(panel_primary drops ticker-days with fewer than 50 cleaned contracts; panel_unfiltered keeps
them). Reads benchmark/parts and benchmark_unfiltered/parts; writes
revision/out/T0b/robustness/unfiltered_vs_primary.csv and robustness/summary.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from common import OUT  # noqa: E402
from summarize import cell_stats, CELL  # noqa: E402

ROB = OUT / "robustness"
ROB.mkdir(parents=True, exist_ok=True)


def load(tag: str) -> pd.DataFrame:
    parts = sorted((OUT / f"benchmark{tag}" / "parts").glob("*.csv"))
    return cell_stats(pd.concat([pd.read_csv(p) for p in parts], ignore_index=True))


def main() -> None:
    prim, unf = load(""), load("_unfiltered")
    m = prim.merge(unf, on=CELL, suffixes=("_primary", "_unfiltered"))
    m["d_ic"] = m.ic_unfiltered - m.ic_primary
    m.to_csv(ROB / "unfiltered_vs_primary.csv", index=False)
    lines = [f"cells compared: {len(m)} (return grid; volatility sets A and D)"]
    for tgt, g in m.groupby("target"):
        lines.append(f"{tgt}: mean |dIC| {g.d_ic.abs().mean():.4f}, max |dIC| {g.d_ic.abs().max():.4f}, "
                     f"corr {g[['ic_primary', 'ic_unfiltered']].corr().iloc[0, 1]:.3f}; t>3 cells primary "
                     f"{int((g.t_primary > 3).sum())}, unfiltered {int((g.t_unfiltered > 3).sum())}")
    hl = m[(m.target == "return") & (m.feature_set.isin(["A", "D"])) & (m.scheme == "rolling_9m_3m") & (m.horizon == 5)]
    for r in hl.itertuples():
        lines.append(f"  return {r.model}/{r.feature_set}/rolling/h5: primary IC {r.ic_primary:+.4f} (t {r.t_primary:+.2f}) "
                     f"| unfiltered {r.ic_unfiltered:+.4f} (t {r.t_unfiltered:+.2f})")
    (ROB / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
