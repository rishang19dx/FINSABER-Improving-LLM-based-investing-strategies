"""
Regime-Conditional Sharpe (RCS) Analysis
========================================

Compute and display RCS rankings from pre-existing ``sharpe_records.json``
and ``SPX_Classification.csv``.

Usage::

    PYTHONPATH=. python backtest/run_rcs_analysis.py

The script prints:
1. Regime weights derived from the SPX classification (2000-2023).
2. A ranked table of all strategies by RCS.
3. Per-strategy breakdown (Bull / Sideways / Bear Sharpe + weighted
   contribution).
"""

import os
import json
import sys

# Ensure the project root is on the path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.toolkit.rcs import RegimeClassifier, compute_rcs


def main():
    results_dir = os.path.join("backtest", "output")
    spx_path    = os.path.join(results_dir, "SPX_Classification.csv")
    sharpe_path = os.path.join(results_dir, "sharpe_records.json")

    # ── Load data ────────────────────────────────────────────────────────
    if not os.path.exists(sharpe_path):
        print(f"ERROR: {sharpe_path} not found.  Run the full experiment first.")
        sys.exit(1)

    with open(sharpe_path) as f:
        sharpe_records = json.load(f)

    classifier = RegimeClassifier(spx_path)
    weights    = classifier.get_regime_weights()

    # ── Regime weights ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Regime Weights  (from SPX_Classification.csv)")
    print("=" * 65)
    for regime, w in weights.items():
        print(f"    {regime:<10}  w = {w:.4f}  ({w*100:.1f}%)")
    print()

    # ── Compute RCS ──────────────────────────────────────────────────────
    rcs_scores = compute_rcs(sharpe_records, weights)

    # ── Ranked table ─────────────────────────────────────────────────────
    ranked = sorted(rcs_scores.items(), key=lambda x: x[1], reverse=True)

    print("=" * 65)
    print("  Regime-Conditional Sharpe (RCS) Rankings")
    print("=" * 65)
    print(f"  {'#':<4} {'Strategy':<25} {'RCS':>8}")
    print("  " + "-" * 39)
    for i, (strategy, score) in enumerate(ranked, 1):
        print(f"  {i:<4} {strategy:<25} {score:>+8.4f}")

    # ── Detailed breakdown ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Per-Strategy Regime Breakdown")
    print("=" * 65)
    header = f"  {'Strategy':<25} {'Bull':>7} {'Side':>7} {'Bear':>7}  │ {'wBull':>7} {'wSide':>7} {'wBear':>7}  │ {'RCS':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for record in sharpe_records:
        s  = record["Strategy"]
        b  = record["Bull"]
        sw = record["Sideways"]
        be = record["Bear"]
        wb = weights["Bull"]   * b
        ws = weights["Sideways"] * sw
        wbe = weights["Bear"]  * be
        rcs = rcs_scores[s]
        print(f"  {s:<25} {b:>+7.3f} {sw:>+7.3f} {be:>+7.3f}  │ {wb:>+7.4f} {ws:>+7.4f} {wbe:>+7.4f}  │ {rcs:>+8.4f}")

    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
