"""
Rolling Sharpe Risk Overlay Analysis
=====================================

Load existing backtest pickle results, apply the post-hoc rolling-Sharpe
risk overlay, and produce before/after comparison tables.

Usage::

    PYTHONPATH=. python backtest/run_risk_overlay_analysis.py
    PYTHONPATH=. python backtest/run_risk_overlay_analysis.py --setup lowvol_sp500_5
    PYTHONPATH=. python backtest/run_risk_overlay_analysis.py --setup cherry_pick_both_finmem --window 60

The overlay never re-runs backtests — it operates entirely on saved equity
curves, making it fast and safe to experiment with parameters.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.toolkit.risk_overlay import apply_risk_overlay


def load_pickle_results(setup_dir: str) -> dict:
    """Load all strategy pickle files under *setup_dir*."""
    results = {}
    for strategy_name in sorted(os.listdir(setup_dir)):
        strategy_dir = os.path.join(setup_dir, strategy_name)
        if not os.path.isdir(strategy_dir) or strategy_name.startswith("."):
            continue
        pkls = [f for f in os.listdir(strategy_dir) if f.endswith(".pkl")]
        if not pkls:
            continue
        pkl_path = os.path.join(strategy_dir, pkls[0])
        try:
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)
            results[strategy_name] = data
        except Exception as e:
            print(f"  Warning: could not load {pkl_path}: {e}")
    return results


def extract_equity_curves(strategy_data: dict) -> dict:
    """Extract equity curves per ticker (concatenated across windows if needed).

    For each ticker, we pick the longest single equity curve available
    (to avoid splicing artifacts from concatenating rolling windows).
    """
    ticker_equities = {}
    for window_key, tickers in strategy_data.items():
        for ticker, result in tickers.items():
            eq = result.get("equity_with_time")
            if eq is not None and len(eq) > 10:
                equity = eq["equity"].values
                # Keep the longest curve per ticker
                if ticker not in ticker_equities or len(equity) > len(ticker_equities[ticker]):
                    ticker_equities[ticker] = equity
    return ticker_equities


def main():
    parser = argparse.ArgumentParser(description="Risk Overlay Analysis")
    parser.add_argument("--setup", default="cherry_pick_both_finmem",
                        help="Setup name (default: cherry_pick_both_finmem)")
    parser.add_argument("--window", type=int, default=30,
                        help="Rolling Sharpe window in days (default: 30)")
    parser.add_argument("--upper", type=float, default=0.0,
                        help="Full-position Sharpe threshold (default: 0.0)")
    parser.add_argument("--lower", type=float, default=-1.0,
                        help="Zero-position Sharpe threshold (default: -1.0)")
    parser.add_argument("--max-dd", type=float, default=30.0,
                        help="Circuit breaker drawdown %% (default: 30.0)")
    args = parser.parse_args()

    setup_dir = os.path.join("backtest", "output", args.setup)
    if not os.path.isdir(setup_dir):
        print(f"ERROR: {setup_dir} not found")
        sys.exit(1)

    print(f"\nLoading results from: {setup_dir}")
    all_results = load_pickle_results(setup_dir)

    if not all_results:
        print("No pickle results found.")
        sys.exit(1)

    # ── Header ────────────────────────────────────────────────────────────
    print("\n" + "=" * 105)
    print(f"  Rolling Sharpe Risk Overlay Analysis  —  Setup: {args.setup}")
    print(f"  Window: {args.window}d  |  Thresholds: [{args.lower}, {args.upper}]  |  Max DD: {args.max_dd}%")
    print("=" * 105)

    # ── Per-strategy summary ──────────────────────────────────────────────
    all_summaries = []

    for strategy_name in sorted(all_results.keys()):
        ticker_equities = extract_equity_curves(all_results[strategy_name])
        if not ticker_equities:
            continue

        raw_sharpes = []
        adj_sharpes = []
        raw_dds = []
        adj_dds = []
        raw_returns = []
        adj_returns = []

        for ticker, equity in ticker_equities.items():
            if len(equity) < args.window + 5:
                continue

            result = apply_risk_overlay(
                equity,
                sharpe_window=args.window,
                sharpe_upper=args.upper,
                sharpe_lower=args.lower,
                max_drawdown_pct=args.max_dd,
            )

            raw_sharpes.append(result["raw_metrics"]["sharpe"])
            adj_sharpes.append(result["adjusted_metrics"]["sharpe"])
            raw_dds.append(result["raw_metrics"]["max_drawdown_pct"])
            adj_dds.append(result["adjusted_metrics"]["max_drawdown_pct"])
            raw_returns.append(result["raw_metrics"]["total_return"])
            adj_returns.append(result["adjusted_metrics"]["total_return"])

        if not raw_sharpes:
            continue

        summary = {
            "strategy": strategy_name,
            "n_tickers": len(raw_sharpes),
            "raw_sharpe": np.mean(raw_sharpes),
            "adj_sharpe": np.mean(adj_sharpes),
            "raw_dd": np.mean(raw_dds),
            "adj_dd": np.mean(adj_dds),
            "raw_return": np.mean(raw_returns),
            "adj_return": np.mean(adj_returns),
        }
        summary["sharpe_delta"] = summary["adj_sharpe"] - summary["raw_sharpe"]
        summary["dd_delta"] = summary["adj_dd"] - summary["raw_dd"]
        all_summaries.append(summary)

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n  {'Strategy':<28} {'N':>3} │ {'Raw SR':>7} {'Adj SR':>7} {'ΔSR':>7} │ {'Raw DD':>7} {'Adj DD':>7} {'ΔDD':>7} │ {'Raw Ret':>8} {'Adj Ret':>8}")
    print("  " + "-" * 101)

    for s in sorted(all_summaries, key=lambda x: x["sharpe_delta"], reverse=True):
        print(
            f"  {s['strategy']:<28} {s['n_tickers']:>3} │ "
            f"{s['raw_sharpe']:>+7.3f} {s['adj_sharpe']:>+7.3f} {s['sharpe_delta']:>+7.3f} │ "
            f"{s['raw_dd']:>6.1f}% {s['adj_dd']:>6.1f}% {s['dd_delta']:>+6.1f}% │ "
            f"{s['raw_return']*100:>+7.1f}% {s['adj_return']*100:>+7.1f}%"
        )

    # ── Detailed per-ticker for key strategies ────────────────────────────
    detail_strategies = [
        s for s in ["FinMemStrategy", "FinAgentStrategy", "BuyAndHoldStrategy"]
        if s in all_results
    ]

    for strategy_name in detail_strategies:
        ticker_equities = extract_equity_curves(all_results[strategy_name])
        if not ticker_equities:
            continue

        print(f"\n  ── {strategy_name} (per-ticker detail) ──")
        print(f"    {'Ticker':<10} {'Raw SR':>7} {'Adj SR':>7} {'ΔSR':>7} │ {'Raw DD':>7} {'Adj DD':>7} │ {'Raw Ret':>8} {'Adj Ret':>8} │ {'Avg Scale':>9}")
        print("    " + "-" * 82)

        for ticker in sorted(ticker_equities.keys()):
            equity = ticker_equities[ticker]
            if len(equity) < args.window + 5:
                continue

            r = apply_risk_overlay(
                equity,
                sharpe_window=args.window,
                sharpe_upper=args.upper,
                sharpe_lower=args.lower,
                max_drawdown_pct=args.max_dd,
            )

            avg_scale = np.mean(r["scales"])
            sr_delta = r["adjusted_metrics"]["sharpe"] - r["raw_metrics"]["sharpe"]

            print(
                f"    {ticker:<10} "
                f"{r['raw_metrics']['sharpe']:>+7.3f} {r['adjusted_metrics']['sharpe']:>+7.3f} {sr_delta:>+7.3f} │ "
                f"{r['raw_metrics']['max_drawdown_pct']:>6.1f}% {r['adjusted_metrics']['max_drawdown_pct']:>6.1f}% │ "
                f"{r['raw_metrics']['total_return']*100:>+7.1f}% {r['adjusted_metrics']['total_return']*100:>+7.1f}% │ "
                f"{avg_scale:>8.1%}"
            )

    print("\n" + "=" * 105)
    print("  ΔSR > 0 → overlay improved risk-adjusted returns")
    print("  ΔDD < 0 → overlay reduced maximum drawdown")
    print("  Avg Scale < 100% → overlay was actively reducing exposure")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
