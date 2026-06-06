"""Generate summary figures from result tables.

Outputs:
  results/figures/feature_set_comparison.png
  results/figures/model_comparison.png
  results/figures/expanding_window.png
  results/figures/regime_performance.png
  results/figures/cumulative_equity.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.utils.io_helpers import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "XGBoost": "#1f77b4",
    "FFNN": "#ff7f0e",
    "TabNet": "#2ca02c",
    "TFT": "#d62728",
    "SetTrans": "#9467bd",
}


def _model_family(name: str) -> str:
    for key in COLORS:
        if name.startswith(key):
            return key
    return "Other"


# ---------------------------------------------------------------------------
# 1. Feature-set comparison (OOS R² across feature sets for each model)
# ---------------------------------------------------------------------------

def plot_feature_set_comparison():
    df = pd.read_csv(RESULTS_DIR / "tables" / "model_comparison.csv")
    panel = df[df["feature_set"].isin(["A", "B", "C", "candidate_6", "D"])].copy()
    panel["family"] = panel["model"].map(_model_family)
    panel = panel[panel["family"].isin(["XGBoost", "FFNN", "TabNet", "TFT"])]

    order = ["A", "B", "C", "candidate_6", "D"]
    families = ["XGBoost", "FFNN", "TabNet", "TFT"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for metric, ax, title, ybase in [
        ("oos_r2", ax1, r"Out-of-sample $R^2$", 0.0),
        ("ic", ax2, "Information coefficient", 0.0),
    ]:
        width = 0.2
        x = np.arange(len(order))
        for i, fam in enumerate(families):
            sub = panel[panel["family"] == fam].set_index("feature_set").reindex(order)
            vals = sub[metric].values
            ax.bar(x + (i - 1.5) * width, vals, width,
                   label=fam, color=COLORS[fam], alpha=0.85)
        ax.axhline(ybase, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["A\n(stock)", "B\n(options)", "C\n(combined)",
                            "cand-6\n(top 6)", "D\n(SVI)"])
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle("Feature-set ablation across model families (3-day horizon)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "feature_set_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# 2. Full model comparison (bar chart of top metrics)
# ---------------------------------------------------------------------------

def plot_model_comparison():
    df = pd.read_csv(RESULTS_DIR / "tables" / "model_comparison.csv")
    panel = df[df["feature_set"].isin(["A", "B", "C", "candidate_6", "D"])].copy()
    panel["family"] = panel["model"].map(_model_family)
    # Keep only the four neural/tree families — drop benchmarks and the
    # single-feature OLS baselines to keep the chart readable.
    panel = panel[panel["family"].isin(["XGBoost", "FFNN", "TabNet", "TFT"])]

    panel["label"] = panel.apply(
        lambda r: f"{r['family']}\n{r['feature_set']}", axis=1,
    )
    panel = panel.sort_values("oos_r2")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, metric, title, fmt in [
        (axes[0], "oos_r2", r"OOS $R^2$", "{:+.4f}"),
        (axes[1], "ic", "Information Coefficient", "{:+.3f}"),
        (axes[2], "sharpe_ratio", "Sharpe ratio (ann.)", "{:.2f}"),
    ]:
        colors = panel["family"].map(COLORS)
        bars = ax.barh(panel["label"], panel[metric], color=colors, alpha=0.85)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(title)
        for bar, v in zip(bars, panel[metric]):
            if not pd.isna(v):
                ax.text(v, bar.get_y() + bar.get_height() / 2,
                        " " + fmt.format(v), va="center", fontsize=8)

    fig.suptitle("Model × feature-set results (3-day horizon, XGBoost/FFNN/TabNet/TFT)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# 3. Expanding-window backtest
# ---------------------------------------------------------------------------

def plot_expanding_window():
    path = RESULTS_DIR / "tables" / "expanding_window.csv"
    if not path.exists():
        print(f"  skip (no {path.name})")
        return
    df = pd.read_csv(path)
    df["test_start"] = pd.to_datetime(df["test_start"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    for ax, col, title, zero in [
        (axes[0, 0], "oos_r2", r"OOS $R^2$ per window", 0),
        (axes[0, 1], "ic", "IC per window", 0),
        (axes[1, 0], "accuracy", "Directional accuracy", 0.5),
        (axes[1, 1], "sharpe_ratio", "Sharpe ratio (annualized)", 0),
    ]:
        ax.plot(df["test_start"], df[col], "-o", ms=4, lw=1.2,
                color="#1f77b4", alpha=0.85)
        ax.axhline(zero, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.axhline(df[col].median(), color="red", lw=1.0, ls=":",
                   label=f"median={df[col].median():+.3f}")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"Expanding-window XGBoost (set C) — {len(df)} quarterly retrains, 2018–2024",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "expanding_window.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# 4. Regime performance
# ---------------------------------------------------------------------------

def plot_regime_performance():
    path = RESULTS_DIR / "tables" / "regime_analysis.csv"
    if not path.exists():
        print(f"  skip (no {path.name})")
        return
    df = pd.read_csv(path)
    df["base_model"] = df["model"].str.split("|").str[0]

    regime_dims = df["regime_name"].unique()

    fig, axes = plt.subplots(1, len(regime_dims), figsize=(5 * len(regime_dims), 5))
    if len(regime_dims) == 1:
        axes = [axes]

    for ax, rd in zip(axes, regime_dims):
        sub = df[df["regime_name"] == rd].copy()
        sub["regime_value"] = sub["regime_value"].astype(str)
        pivot = sub.pivot_table(
            index="regime_value", columns="base_model", values="ic",
        )
        pivot.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"], alpha=0.85)
        ax.set_title(rd.replace("_", " ").title())
        ax.set_ylabel("IC")
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Conditional performance by regime (IC, XGBoost)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "regime_performance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# 5. Summary numbers (text output for the paper)
# ---------------------------------------------------------------------------

def write_summary_numbers():
    df = pd.read_csv(RESULTS_DIR / "tables" / "model_comparison.csv")
    out_path = RESULTS_DIR / "tables" / "summary_numbers.txt"

    with open(out_path, "w") as f:
        f.write("Key numbers used in the report\n")
        f.write("=" * 60 + "\n\n")

        for fs in ["A", "B", "C", "candidate_6", "D"]:
            sub = df[df["feature_set"] == fs]
            best = sub.loc[sub["oos_r2"].idxmax()] if not sub["oos_r2"].isna().all() else None
            if best is not None:
                f.write(
                    f"set {fs:<12} best R²: {best['model']:<30} "
                    f"R²={best['oos_r2']:+.4f}  IC={best['ic']:+.4f}  "
                    f"acc={best['accuracy']:.3f}\n"
                )

        f.write("\nStatistical tests (A vs C):\n")
        st = pd.read_csv(RESULTS_DIR / "tables" / "statistical_tests.csv")
        for _, r in st.iterrows():
            f.write(f"  {r['test']:<28} p={r['p_value']:.4f}\n")

        ew = pd.read_csv(RESULTS_DIR / "tables" / "expanding_window.csv")
        f.write("\nExpanding window (XGBoost set C, quarterly):\n")
        for col in ["oos_r2", "ic", "accuracy", "sharpe_ratio"]:
            f.write(f"  {col:<18} mean={ew[col].mean():+.4f}  "
                    f"median={ew[col].median():+.4f}  std={ew[col].std():.4f}\n")
        f.write(f"  windows            n={len(ew)}\n")

    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# 6. Walk-forward: expanding vs rolling, pooled metrics per feature set
# ---------------------------------------------------------------------------

def plot_walkforward_pooled():
    path = RESULTS_DIR / "tables" / "walk_forward_pooled.csv"
    if not path.exists():
        print(f"  skip (no {path.name})")
        return
    df = pd.read_csv(path)

    order = ["A", "B", "C", "candidate_6", "D"]
    df = df[df["feature_set"].isin(order)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, metric, title, zero in [
        (axes[0], "pooled_r2",       r"Pooled OOS $R^2$", 0.0),
        (axes[1], "pooled_ic",       r"Pooled IC",        0.0),
        (axes[2], "pooled_accuracy", r"Pooled accuracy",  0.5),
    ]:
        width = 0.38
        x = np.arange(len(order))
        for i, scheme in enumerate(["expanding", "rolling_9m_3m"]):
            sub = (df[df["scheme"] == scheme]
                   .set_index("feature_set")
                   .reindex(order))
            vals = sub[metric].values
            label = "expanding" if scheme == "expanding" else "rolling 9m/3m"
            color = "#1f77b4" if scheme == "expanding" else "#d62728"
            ax.bar(x + (i - 0.5) * width, vals, width,
                   label=label, color=color, alpha=0.85)
        ax.axhline(zero, color="k", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(["A", "B", "C", "cand-6", "D"])
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Walk-forward evaluation: expanding vs rolling 9-month/3-month "
        "(XGBoost, 2018--2024)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "walkforward_pooled.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# 7. Per-year dispersion across windows (Bali et al. 2022 Figure 3 style)
# ---------------------------------------------------------------------------

def plot_per_year_dispersion():
    exp_path = RESULTS_DIR / "tables" / "expanding_window.csv"
    rol_path = RESULTS_DIR / "tables" / "rolling_window.csv"
    if not exp_path.exists() or not rol_path.exists():
        print("  skip (missing expanding/rolling CSVs)")
        return

    exp = pd.read_csv(exp_path); exp["scheme"] = "expanding"
    rol = pd.read_csv(rol_path); rol["scheme"] = "rolling_9m_3m"
    combined = pd.concat([exp, rol], ignore_index=True)
    combined["test_start"] = pd.to_datetime(combined["test_start"])
    combined["year"] = combined["test_start"].dt.year

    order = ["A", "B", "C", "candidate_6", "D"]
    combined = combined[combined["feature_set"].isin(order)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for ax, scheme, title in [
        (axes[0], "expanding",
         "Expanding window: per-window $R^2$ by feature set"),
        (axes[1], "rolling_9m_3m",
         "Rolling 9m/3m window: per-window $R^2$ by feature set"),
    ]:
        sub = combined[combined["scheme"] == scheme]
        positions = []
        x = 0
        for fs in order:
            sub_fs = sub[sub["feature_set"] == fs]
            if len(sub_fs) == 0:
                continue
            bp = ax.boxplot(
                sub_fs["oos_r2"].values, positions=[x], widths=0.65,
                patch_artist=True, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 6},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
            )
            for p in bp["boxes"]:
                p.set_facecolor("#1f77b4" if scheme == "expanding" else "#d62728")
                p.set_alpha(0.5)
            positions.append(x)
            x += 1
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(["A", "B", "C", "cand-6", "D"])
        ax.set_ylabel(r"OOS $R^2$")
        ax.set_title(title)

    fig.suptitle(
        "Per-window $R^2$ dispersion (2018--2024) --- follows Bali et al. 2022 Fig.~3",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "per_window_dispersion.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# 8. Annual OOS R² trajectory (also Bali et al. Fig. 3 inspired)
# ---------------------------------------------------------------------------

def plot_annual_trajectory():
    exp_path = RESULTS_DIR / "tables" / "expanding_window.csv"
    rol_path = RESULTS_DIR / "tables" / "rolling_window.csv"
    if not exp_path.exists() or not rol_path.exists():
        print("  skip (missing expanding/rolling CSVs)")
        return

    exp = pd.read_csv(exp_path); exp["scheme"] = "expanding"
    rol = pd.read_csv(rol_path); rol["scheme"] = "rolling_9m_3m"
    combined = pd.concat([exp, rol], ignore_index=True)
    combined["test_start"] = pd.to_datetime(combined["test_start"])

    focus_sets = ["A", "B", "C", "D"]
    combined = combined[combined["feature_set"].isin(focus_sets)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    color_map = {"A": "#1f77b4", "B": "#2ca02c", "C": "#ff7f0e", "D": "#9467bd"}
    for ax, scheme, title in [
        (axes[0], "expanding",  "Expanding window"),
        (axes[1], "rolling_9m_3m", "Rolling 9m/3m window"),
    ]:
        sub = combined[combined["scheme"] == scheme]
        for fs in focus_sets:
            s = sub[sub["feature_set"] == fs].sort_values("test_start")
            if len(s) == 0:
                continue
            ax.plot(s["test_start"], s["ic"], "-o", ms=3, lw=1.2,
                    label=f"set {fs}", color=color_map[fs], alpha=0.85)
        ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel("IC per window")
        ax.legend(fontsize=9, ncol=4, loc="best")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "IC trajectory across 2018--2024: expanding vs rolling walk-forward",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "walkforward_ic_trajectory.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    print("Generating figures and summary numbers...")
    plot_feature_set_comparison()
    plot_model_comparison()
    plot_expanding_window()
    plot_regime_performance()
    plot_walkforward_pooled()
    plot_per_year_dispersion()
    plot_annual_trajectory()
    write_summary_numbers()
    print("Done.")
