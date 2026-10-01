#!/usr/bin/env python3
"""Cross-sectional IC power calculator (a shippable deliverable).

Maps (effect size, target t, reference universe) -> the cross-section size N
needed to declare a cross-sectional rank-IC effect a "discovery", given that the
HAC standard error of a mean walk-forward IC scales as 1/sqrt(N).

Model
-----
A daily cross-sectional rank IC over N names has sampling sd ~ 1/sqrt(N-1).
Pooled into per-window ICs and across W windows, the realised HAC SE of the mean
IC in *this* study is se_ref at N_ref names. Because that SE scales as
1/sqrt(N), the minimum detectable effect at significance t* is
    MDE(N) = t* * se_ref * sqrt(N_ref / N),
and the universe needed to resolve a true lift `effect` at t* is
    N_req   = N_ref * (t* * se_ref / effect)^2.

Defaults are calibrated to this paper (se_ref=0.0064 at N_ref=60, t*=3, the
Harvey-Liu-Zhu hurdle), but every input is overridable so other studies can reuse
the tool.

CLI
---
    python power_calculator.py --effect 0.004
    python power_calculator.py --effect 0.01 --t 1.96 --se_ref 0.0064 --n_ref 60
"""
from __future__ import annotations
import argparse

def required_N(effect: float, t: float = 3.0, se_ref: float = 0.0064,
               n_ref: int = 60) -> float:
    """Cross-section size needed to detect `effect` (rank IC) at significance t."""
    if effect <= 0:
        return float("inf")
    return n_ref * (t * se_ref / effect) ** 2

def mde(n: float, t: float = 3.0, se_ref: float = 0.0064, n_ref: int = 60) -> float:
    """Minimum detectable IC at universe size n and significance t."""
    return t * se_ref * (n_ref / n) ** 0.5

def main():
    ap = argparse.ArgumentParser(description="Cross-sectional IC power calculator")
    ap.add_argument("--effect", type=float, default=0.004,
                    help="true marginal rank-IC lift to detect (default 0.004)")
    ap.add_argument("--t", type=float, default=3.0,
                    help="target |t| (default 3.0 = Harvey-Liu-Zhu hurdle; use 1.96 for 5%%)")
    ap.add_argument("--se_ref", type=float, default=0.0064,
                    help="reference HAC SE of the mean IC (default 0.0064, this study)")
    ap.add_argument("--n_ref", type=int, default=60,
                    help="reference cross-section size (default 60, this study)")
    a = ap.parse_args()
    n = required_N(a.effect, a.t, a.se_ref, a.n_ref)
    print(f"effect (IC lift)      : {a.effect:.4f}")
    print(f"target |t|            : {a.t}")
    print(f"reference SE @ N={a.n_ref:<4}: {a.se_ref:.4f}")
    print(f"MDE at N={a.n_ref:<4}        : {mde(a.n_ref, a.t, a.se_ref, a.n_ref):.4f}")
    print(f"=> names needed       : {n:,.0f}  ({n/a.n_ref:.1f}x the reference universe)")

if __name__ == "__main__":
    main()
