"""
Probabilistic Sharpe Ratio (PSR) Analysis
==========================================

Load existing backtest pickle results, extract equity curves, compute
daily returns, and evaluate PSR + MinTRL for every strategy × ticker.

Usage::

    PYTHONPATH=. python backtest/run_psr_analysis.py
    PYTHONPATH=. python backtest/run_psr_analysis.py --setup lowvol_sp500_5
    PYTHONPATH=. python backtest/run_psr_analysis.py --setup cherry_pick_both_finmem

Produces a table showing, for each strategy:
- Observed Sharpe, PSR (vs Sharpe=0), MinTRL in years
- Skewness, excess kurtosis of daily returns
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.toolkit.psr import compute_psr_from_returns


def load_pickle_results(setup_dir: str) -> dict:
    """Load all strategy pickle files under *setup_dir*.

    Returns
    -------
    dict
        ``{strategy_name: {window: {ticker: result_dict}}}``
    """
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


def extract_all_daily_returns(strategy_data: dict) -> np.ndarray:
    """Concatenate daily returns from all windows × tickers for a strategy."""
    all_returns = []
    for window_key, tickers in strategy_data.items():
        for ticker, result in tickers.items():
            eq = result.get("equity_with_time")
            if eq is not None and len(eq) > 1:
                equity = eq["equity"].values
                daily_ret = np.diff(equity) / equity[:-1]
                daily_ret = daily_ret[np.isfinite(daily_ret)]
                all_returns.append(daily_ret)
    if all_returns:
        return np.concatenate(all_returns)
    return np.array([])


def extract_per_ticker_returns(strategy_data: dict) -> dict:
    """Extract daily returns per ticker (pooled across windows)."""
    ticker_returns = {}
    for window_key, tickers in strategy_data.items():
        for ticker, result in tickers.items():
            eq = result.get("equity_with_time")
            if eq is not None and len(eq) > 1:
                equity = eq["equity"].values
                daily_ret = np.diff(equity) / equity[:-1]
                daily_ret = daily_ret[np.isfinite(daily_ret)]
                if ticker not in ticker_returns:
                    ticker_returns[ticker] = []
                ticker_returns[ticker].append(daily_ret)
    # concatenate per ticker
    return {t: np.concatenate(rets) for t, rets in ticker_returns.items() if rets}


def main():
    parser = argparse.ArgumentParser(description="PSR & MinTRL analysis")
    parser.add_argument(
        "--setup",
        default="cherry_pick_both_finmem",
        help="Setup name under backtest/output/ (default: cherry_pick_both_finmem)",
    )
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

    # ── Strategy-level summary ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  PSR & MinTRL Analysis  —  Setup: {args.setup}")
    print("=" * 90)
    print(
        f"  {'Strategy':<28} {'Obs SR':>7} {'Skew':>7} {'Kurt':>7} "
        f"{'T (days)':>9} {'PSR':>7} {'MinTRL':>9} {'Signif?':>8}"
    )
    print("  " + "-" * 86)

    summary_rows = []

    for strategy_name in sorted(all_results.keys()):
        daily_returns = extract_all_daily_returns(all_results[strategy_name])
        if len(daily_returns) < 10:
            continue

        psr_result = compute_psr_from_returns(daily_returns, benchmark_sr=0.0)
        sig = "YES" if psr_result["psr"] >= 0.95 else "no"

        min_trl_str = (
            f"{psr_result['min_trl_years']:.1f}y"
            if np.isfinite(psr_result["min_trl_years"])
            else "∞"
        )

        print(
            f"  {strategy_name:<28} {psr_result['observed_sr']:>+7.3f} "
            f"{psr_result['skewness']:>+7.3f} {psr_result['kurtosis']:>+7.2f} "
            f"{psr_result['n_observations']:>9d} {psr_result['psr']:>7.4f} "
            f"{min_trl_str:>9} {sig:>8}"
        )
        summary_rows.append(
            {
                "strategy": strategy_name,
                **psr_result,
            }
        )

    # ── Per-ticker detail for top strategies ──────────────────────────────
    print("\n" + "=" * 90)
    print("  Per-Ticker Detail (selected strategies)")
    print("=" * 90)

    detail_strategies = [
        s for s in ["BuyAndHoldStrategy", "FinMemStrategy", "FinAgentStrategy",
                     "ARIMAPredictorStrategy"]
        if s in all_results
    ]

    for strategy_name in detail_strategies:
        print(f"\n  ── {strategy_name} ──")
        ticker_rets = extract_per_ticker_returns(all_results[strategy_name])

        print(
            f"    {'Ticker':<10} {'Obs SR':>7} {'Skew':>7} {'Kurt':>7} "
            f"{'T':>6} {'PSR':>7} {'MinTRL':>9}"
        )
        print("    " + "-" * 56)

        for ticker in sorted(ticker_rets.keys()):
            rets = ticker_rets[ticker]
            if len(rets) < 10:
                continue
            r = compute_psr_from_returns(rets, benchmark_sr=0.0)
            min_str = (
                f"{r['min_trl_years']:.1f}y"
                if np.isfinite(r["min_trl_years"])
                else "∞"
            )
            print(
                f"    {ticker:<10} {r['observed_sr']:>+7.3f} "
                f"{r['skewness']:>+7.3f} {r['kurtosis']:>+7.2f} "
                f"{r['n_observations']:>6d} {r['psr']:>7.4f} {min_str:>9}"
            )

    print("\n" + "=" * 90)
    print("  PSR ≥ 0.95 → statistically significant at 95% confidence")
    print("  MinTRL = minimum track record needed (in years) for significance")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
