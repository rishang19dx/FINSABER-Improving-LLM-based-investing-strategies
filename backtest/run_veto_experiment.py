"""
ARIMA–LLM Veto Layer Experiment
================================

Post-hoc experiment that combines ARIMA directional forecasts with
LLM position signals (inferred from equity curves) to test whether
ARIMA can filter out false LLM buy signals in adverse regimes.

Usage::
    PYTHONPATH=. python backtest/run_veto_experiment.py
    PYTHONPATH=. python backtest/run_veto_experiment.py --setup lowvol_sp500_5
    PYTHONPATH=. python backtest/run_veto_experiment.py --setup cherry_pick_both_finmem

Produces comparison tables and plots for four variants:
  1. LLM-only (baseline)
  2. ARIMA-only (benchmark)
  3. Veto Hybrid (ARIMA vetoes LLM buys)
  4. Reverse Veto (LLM vetoes ARIMA buys — control)
"""

import argparse
import json
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.toolkit.veto_layer import (
    infer_position_from_equity,
    positions_to_signals,
    apply_veto_position,
    simulate_equity,
    BUY, HOLD, SELL,
)
from backtest.toolkit.psr import compute_psr_from_returns

warnings.filterwarnings("ignore")

# ── Plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.dpi": 150,
})

PRICE_CSV = "data/price/all_sp500_prices_2000_2024_delisted_include.csv"


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_price_data():
    """Load the master price CSV into a ticker-indexed dict of Series."""
    print("  Loading price data...")
    df = pd.read_csv(PRICE_CSV, usecols=["date", "close", "symbol"],
                     parse_dates=["date"])
    df = df.sort_values(["symbol", "date"])
    price_dict = {}
    for sym, grp in df.groupby("symbol"):
        s = grp.set_index("date")["close"].sort_index()
        s = s[~s.index.duplicated(keep="first")]
        price_dict[sym] = s
    print(f"    Loaded prices for {len(price_dict)} symbols")
    return price_dict


def load_strategy_pickle(setup_dir, strategy_name):
    """Load a single strategy's pickle result."""
    sdir = os.path.join(setup_dir, strategy_name)
    if not os.path.isdir(sdir):
        return None
    pkls = [f for f in os.listdir(sdir) if f.endswith(".pkl")]
    if not pkls:
        return None
    with open(os.path.join(sdir, pkls[0]), "rb") as f:
        return pickle.load(f)


def compute_metrics(equity):
    """Compute Sharpe, Sortino, Max DD, Total Return from an equity curve."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 10:
        return {"sharpe": 0, "sortino": 0, "max_dd": 0, "total_return": 0}

    daily_ret = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1e-12)
    daily_ret = daily_ret[np.isfinite(daily_ret)]

    if len(daily_ret) < 5 or np.std(daily_ret) < 1e-12:
        return {"sharpe": 0, "sortino": 0, "max_dd": 0, "total_return": 0}

    sharpe = np.mean(daily_ret) / np.std(daily_ret, ddof=1) * np.sqrt(252)

    down = daily_ret[daily_ret < 0]
    down_std = np.std(down, ddof=1) * np.sqrt(252) if len(down) > 1 else 0.0
    sortino = np.mean(daily_ret) * 252 / down_std if down_std > 1e-8 else 0.0
    sortino = np.clip(sortino, -100, 100)  # cap to avoid overflow

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak != 0, peak, 1e-12)
    max_dd = abs(dd.min()) * 100

    total_return = (eq[-1] / eq[0]) - 1 if eq[0] > 0 else 0

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd": round(max_dd, 2),
        "total_return": round(total_return, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-ticker Veto Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_veto_for_ticker(
    llm_equity, llm_dates, arima_equity, prices_series, ticker, window_str
):
    """Run the full veto experiment for one ticker in one window.

    Uses the *actual* ARIMA equity curve from pickles (not reconstructed)
    to infer ARIMA position signals — matched to the real Backtrader run.

    Returns a dict with metrics for all four variants, or None on failure.
    """
    dates = pd.to_datetime(llm_dates)

    # Get prices aligned to equity dates
    px_aligned = prices_series.reindex(dates)
    if px_aligned.isna().sum() > len(px_aligned) * 0.3:
        return None
    px_aligned = px_aligned.ffill().bfill()
    px = px_aligned.values

    eq_llm = np.asarray(llm_equity, dtype=float)
    eq_arima = np.asarray(arima_equity, dtype=float)
    n = min(len(eq_llm), len(px), len(eq_arima))
    eq_llm, eq_arima, px = eq_llm[:n], eq_arima[:n], px[:n]

    if n < 30:
        return None

    # ── 1. Infer positions from BOTH equity curves ───────────────────────
    llm_positions = infer_position_from_equity(eq_llm, px, window=5, corr_threshold=0.6)
    llm_signals = positions_to_signals(llm_positions)

    arima_positions = infer_position_from_equity(eq_arima, px, window=5, corr_threshold=0.6)

    # ── 2. Apply position-level veto ─────────────────────────────────────
    # Check if ARIMA is currently IN position when LLM fires BUY
    veto_signals = apply_veto_position(llm_signals, arima_positions, mode="arima_veto")
    reverse_signals = apply_veto_position(llm_signals, arima_positions, mode="reverse_veto")

    # ── 3. Simulate hybrid equity curves ─────────────────────────────────
    veto_equity = simulate_equity(px, veto_signals)
    reverse_equity = simulate_equity(px, reverse_signals)

    # ── 4. Signal diagnostics ────────────────────────────────────────────
    llm_buys = np.sum(llm_signals == BUY)
    arima_buys = np.sum(positions_to_signals(arima_positions) == BUY)
    veto_buys = np.sum(veto_signals == BUY)
    vetoed_count = llm_buys - veto_buys

    return {
        "ticker": ticker,
        "window": window_str,
        "n_days": n,
        "llm_buys": int(llm_buys),
        "arima_buys": int(arima_buys),
        "veto_buys": int(veto_buys),
        "vetoed": int(vetoed_count),
        "llm_only": compute_metrics(eq_llm),
        "arima_only": compute_metrics(eq_arima),
        "veto_hybrid": compute_metrics(veto_equity),
        "reverse_veto": compute_metrics(reverse_equity),
        "_equity_curves": {
            "llm": eq_llm, "arima": eq_arima,
            "veto": veto_equity, "reverse": reverse_equity,
            "prices": px,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ARIMA-LLM Veto Experiment")
    parser.add_argument("--setup", default="lowvol_sp500_5",
                        help="Setup name (default: lowvol_sp500_5)")
    parser.add_argument("--llm", default="FinMemStrategy",
                        choices=["FinMemStrategy", "FinAgentStrategy"],
                        help="LLM strategy to use")
    args = parser.parse_args()

    setup_dir = os.path.join("backtest", "output", args.setup)
    out_dir = os.path.join("backtest", "output", "veto_experiment")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    price_dict = load_price_data()

    print(f"\n  Loading LLM results: {args.llm} from {args.setup}")
    llm_data = load_strategy_pickle(setup_dir, args.llm)
    if llm_data is None:
        print(f"  ERROR: {args.llm} not found in {setup_dir}")
        sys.exit(1)

    print(f"  Loading ARIMA results from {args.setup}")
    arima_data = load_strategy_pickle(setup_dir, "ARIMAPredictorStrategy")
    if arima_data is None:
        print(f"  ERROR: ARIMAPredictorStrategy not found in {setup_dir}")
        sys.exit(1)

    # ── Run experiment per window × ticker ────────────────────────────────
    all_results = []
    total_windows = len(llm_data)

    for wi, (window_key, tickers) in enumerate(llm_data.items()):
        print(f"\n  Window {wi+1}/{total_windows}: {window_key}")

        # Get ARIMA results for this window (may have different tickers)
        arima_window = arima_data.get(window_key, {})

        for ticker, result in tickers.items():
            eq_df = result.get("equity_with_time")
            if eq_df is None or len(eq_df) < 30:
                continue

            if ticker not in price_dict:
                print(f"    ⚠ {ticker}: no price data, skipping")
                continue

            # Get matching ARIMA equity curve
            arima_result = arima_window.get(ticker)
            if arima_result is None:
                print(f"    ⚠ {ticker}: no ARIMA result for this window, skipping")
                continue
            arima_eq_df = arima_result.get("equity_with_time")
            if arima_eq_df is None or len(arima_eq_df) < 30:
                continue

            equity = eq_df["equity"].values
            dates = eq_df["datetime"].values
            arima_equity = arima_eq_df["equity"].values

            try:
                r = run_veto_for_ticker(
                    equity, dates, arima_equity,
                    price_dict[ticker], ticker, window_key
                )
                if r is not None:
                    all_results.append(r)
                    sr_llm = r["llm_only"]["sharpe"]
                    sr_veto = r["veto_hybrid"]["sharpe"]
                    sr_arima = r["arima_only"]["sharpe"]
                    print(f"    {ticker:6s}  LLM={sr_llm:+.3f}  "
                          f"ARIMA={sr_arima:+.3f}  "
                          f"Veto={sr_veto:+.3f}  "
                          f"(vetoed {r['vetoed']}/{r['llm_buys']} buys)")
            except Exception as e:
                print(f"    {ticker:6s}  ERROR: {e}")

    if not all_results:
        print("\n  No results produced.")
        sys.exit(1)

    # ── Aggregate results ─────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  VETO EXPERIMENT RESULTS — {args.llm} × {args.setup}")
    print(f"  {len(all_results)} ticker-windows analysed")
    print("=" * 100)

    variants = ["llm_only", "arima_only", "veto_hybrid", "reverse_veto"]
    labels = ["LLM Only", "ARIMA Only", "Veto Hybrid", "Reverse Veto"]
    metrics_keys = ["sharpe", "sortino", "max_dd", "total_return"]

    # Compute averages
    print(f"\n  {'Variant':<16} {'Sharpe':>8} {'Sortino':>8} "
          f"{'Max DD':>8} {'Tot Ret':>8}")
    print("  " + "-" * 52)

    summary_rows = []
    for var, label in zip(variants, labels):
        vals = {k: [] for k in metrics_keys}
        for r in all_results:
            for k in metrics_keys:
                vals[k].append(r[var][k])

        avg = {k: np.mean(vals[k]) for k in metrics_keys}
        print(f"  {label:<16} {avg['sharpe']:>+8.4f} {avg['sortino']:>+8.4f} "
              f"{avg['max_dd']:>7.2f}% {avg['total_return']*100:>+7.2f}%")
        summary_rows.append({"Variant": label, **{k: round(avg[k], 4) for k in metrics_keys}})

    # ── Signal diagnostics ────────────────────────────────────────────────
    total_llm_buys = sum(r["llm_buys"] for r in all_results)
    total_vetoed = sum(r["vetoed"] for r in all_results)
    veto_rate = total_vetoed / max(total_llm_buys, 1) * 100
    print(f"\n  Signal Diagnostics:")
    print(f"    Total LLM buy signals:  {total_llm_buys}")
    print(f"    Vetoed by ARIMA:        {total_vetoed} ({veto_rate:.1f}%)")
    print(f"    Passed through:         {total_llm_buys - total_vetoed}")

    # ── PSR on aggregated returns ─────────────────────────────────────────
    print(f"\n  {'Variant':<16} {'PSR':>8} {'MinTRL(y)':>10}")
    print("  " + "-" * 36)

    for var, label in zip(variants, labels):
        all_ret = []
        for r in all_results:
            curves = r.get("_equity_curves", {})
            key_map = {"llm_only": "llm", "arima_only": "arima",
                       "veto_hybrid": "veto", "reverse_veto": "reverse"}
            eq = curves.get(key_map.get(var))
            if eq is not None and len(eq) > 5:
                dr = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1e-12)
                dr = dr[np.isfinite(dr)]
                all_ret.append(dr)

        if all_ret:
            concat = np.concatenate(all_ret)
            psr = compute_psr_from_returns(concat, benchmark_sr=0.0)
            mtrl = f"{psr['min_trl_years']:.1f}y" if np.isfinite(psr["min_trl_years"]) else "∞"
            print(f"  {label:<16} {psr['psr']:>8.4f} {mtrl:>10}")

    # ── Save results ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, f"veto_{args.llm}_{args.setup}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Summary → {csv_path}")

    # ── Plot: best/worst veto examples ────────────────────────────────────
    # Sort by improvement (veto Sharpe - LLM Sharpe)
    improvements = [(r["veto_hybrid"]["sharpe"] - r["llm_only"]["sharpe"], r)
                    for r in all_results if "_equity_curves" in r]
    improvements.sort(key=lambda x: x[0], reverse=True)

    n_plot = min(4, len(improvements))
    if n_plot > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Plot top 2 improvements + bottom 2
        examples = improvements[:2] + improvements[-2:]
        for idx, (delta, r) in enumerate(examples[:n_plot]):
            ax = axes[idx]
            curves = r["_equity_curves"]
            n = len(curves["llm"])
            x = range(n)

            ax.plot(x, curves["llm"], color="#f85149", alpha=0.8,
                    linewidth=1.5, label=f'LLM (SR={r["llm_only"]["sharpe"]:+.3f})')
            ax.plot(x, curves["arima"], color="#3fb950", alpha=0.8,
                    linewidth=1.5, label=f'ARIMA (SR={r["arima_only"]["sharpe"]:+.3f})')
            ax.plot(x, curves["veto"], color="#58a6ff", alpha=0.9,
                    linewidth=2, label=f'Veto (SR={r["veto_hybrid"]["sharpe"]:+.3f})')

            title = f'{r["ticker"]} ({r["window"][:10]}…)  Δ={delta:+.3f}'
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Equity ($)")

        for idx in range(n_plot, 4):
            axes[idx].set_visible(False)

        fig.suptitle(f"Veto Layer: Best & Worst Cases — {args.llm}",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        plot_path = os.path.join(out_dir, f"veto_curves_{args.llm}_{args.setup}.png")
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Equity curves → {plot_path}")

    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
