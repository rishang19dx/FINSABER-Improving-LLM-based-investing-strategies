"""
FINSABER Unified Analysis Dashboard
====================================

Single script that loads ALL backtest results and produces:
1. Master comparison table (Sharpe, Sortino, Calmar, Omega, PSR, MinTRL, RCS)
2. Regime-conditional performance heatmap (enhanced Figure 2)
3. Risk overlay before/after comparison
4. PSR significance scatter plot (Sharpe vs MinTRL)
5. Bootstrap confidence intervals for RCS

Usage::
    PYTHONPATH=. python backtest/run_unified_dashboard.py
    PYTHONPATH=. python backtest/run_unified_dashboard.py --setups lowvol_sp500_5 cherry_pick_both_finmem
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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.toolkit.psr import compute_psr_from_returns
from backtest.toolkit.risk_overlay import apply_risk_overlay
from backtest.toolkit import metrics

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────
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

STRATEGY_DISPLAY = {
    "BuyAndHoldStrategy": "Buy & Hold",
    "SMACrossStrategy": "SMA Cross",
    "WMAStrategy": "WMA Cross",
    "ATRBandStrategy": "ATR Band",
    "BollingerBandsStrategy": "Bollinger Bands",
    "TrendFollowingStrategy": "Trend Following",
    "TurnOfTheMonthStrategy": "Turn of Month",
    "ARIMAPredictorStrategy": "ARIMA",
    "XGBoostPredictorStrategy": "XGBoost",
    "FinMemStrategy": "FinMem (LLM)",
    "FinAgentStrategy": "FinAgent (LLM)",
}

STRATEGY_COLORS = {
    "Buy & Hold": "#58a6ff",
    "ARIMA": "#3fb950",
    "XGBoost": "#56d364",
    "SMA Cross": "#8b949e",
    "WMA Cross": "#8b949e",
    "ATR Band": "#d2a8ff",
    "Bollinger Bands": "#d2a8ff",
    "Trend Following": "#d2a8ff",
    "Turn of Month": "#d2a8ff",
    "FinMem (LLM)": "#f85149",
    "FinAgent (LLM)": "#ff7b72",
}

OUTPUT_DIR = os.path.join("backtest", "output", "dashboard")


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_setup_results(setup_dir):
    """Load all strategy pickles from a setup directory."""
    results = {}
    for name in sorted(os.listdir(setup_dir)):
        sdir = os.path.join(setup_dir, name)
        if not os.path.isdir(sdir) or name.startswith(".") or name == "plots":
            continue

        # Handle FinRL sub-directories (PPO, A2C, etc.)
        if name == "FinRLStrategy":
            for algo in sorted(os.listdir(sdir)):
                algo_dir = os.path.join(sdir, algo)
                if not os.path.isdir(algo_dir):
                    continue
                pkls = [f for f in os.listdir(algo_dir) if f.endswith(".pkl")]
                if pkls:
                    try:
                        with open(os.path.join(algo_dir, pkls[0]), "rb") as f:
                            results[f"RL-{algo}"] = pickle.load(f)
                    except Exception:
                        pass
            continue

        pkls = [f for f in os.listdir(sdir) if f.endswith(".pkl")]
        if not pkls:
            continue
        try:
            with open(os.path.join(sdir, pkls[0]), "rb") as f:
                results[name] = pickle.load(f)
        except Exception:
            pass
    return results


def extract_all_returns(strategy_data):
    """Concatenate daily returns from all windows × tickers."""
    all_ret = []
    for window, tickers in strategy_data.items():
        for ticker, result in tickers.items():
            eq = result.get("equity_with_time")
            if eq is not None and len(eq) > 1:
                equity = eq["equity"].values
                dr = np.diff(equity) / equity[:-1]
                dr = dr[np.isfinite(dr)]
                all_ret.append(dr)
    return np.concatenate(all_ret) if all_ret else np.array([])


def extract_equity_curves(strategy_data):
    """Extract longest equity curve per ticker."""
    curves = {}
    for window, tickers in strategy_data.items():
        for ticker, result in tickers.items():
            eq = result.get("equity_with_time")
            if eq is not None and len(eq) > 10:
                equity = eq["equity"].values
                if ticker not in curves or len(equity) > len(curves[ticker]):
                    curves[ticker] = equity
    return curves


def get_display_name(raw_name):
    """Map internal strategy name to display name."""
    if raw_name.startswith("RL-"):
        return raw_name
    return STRATEGY_DISPLAY.get(raw_name, raw_name)


def get_color(display_name):
    if display_name.startswith("RL-"):
        return "#e3b341"
    return STRATEGY_COLORS.get(display_name, "#8b949e")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Master Metrics Table
# ═══════════════════════════════════════════════════════════════════════════

def compute_master_table(all_results, setup_name):
    """Compute all metrics for every strategy in a setup."""
    rows = []
    for strat_name, strat_data in sorted(all_results.items()):
        daily_returns = extract_all_returns(strat_data)
        if len(daily_returns) < 10:
            continue

        returns_series = pd.Series(daily_returns)
        n = len(daily_returns)

        # Basic metrics
        mean_r = np.mean(daily_returns)
        std_r = np.std(daily_returns, ddof=1)
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0
        sortino = metrics.calculate_sortino_ratio(returns_series, 0.03)

        # Cumulative for Calmar
        cum_ret = np.cumprod(1 + daily_returns)
        total_ret = cum_ret[-1] - 1
        ann_ret = (1 + total_ret) ** (252 / n) - 1
        peak = np.maximum.accumulate(cum_ret)
        dd = (cum_ret - peak) / peak
        max_dd_pct = abs(dd.min()) * 100

        calmar = metrics.calculate_calmar_ratio(ann_ret, max_dd_pct)
        omega = metrics.calculate_omega_ratio(returns_series, threshold=0.0)

        # PSR
        psr_res = compute_psr_from_returns(daily_returns, benchmark_sr=0.0)

        display = get_display_name(strat_name)
        rows.append({
            "Strategy": display,
            "raw_name": strat_name,
            "Setup": setup_name,
            "Sharpe": round(sharpe, 3),
            "Sortino": round(sortino, 3),
            "Calmar": round(calmar, 3),
            "Omega": round(omega, 3),
            "PSR": round(psr_res["psr"], 4),
            "MinTRL_y": round(psr_res["min_trl_years"], 1) if np.isfinite(psr_res["min_trl_years"]) else float("inf"),
            "Skew": round(psr_res["skewness"], 3),
            "Kurt": round(psr_res["kurtosis"], 2),
            "N_obs": psr_res["n_observations"],
            "Ann_Ret": round(ann_ret, 4),
            "Max_DD": round(max_dd_pct, 2),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Regime Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_regime_heatmap(sharpe_records_path, output_path):
    """Enhanced version of the paper's Figure 2."""
    with open(sharpe_records_path) as f:
        records = json.load(f)

    strategies = [r["Strategy"] for r in records]
    regimes = ["Bull", "Sideways", "Bear"]
    data = np.array([[r[reg] for reg in regimes] for r in records])

    fig, ax = plt.subplots(figsize=(8, max(6, len(strategies) * 0.45)))

    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=0.7)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=10)

    for i in range(len(strategies)):
        for j in range(len(regimes)):
            val = data[i, j]
            color = "white" if abs(val) > 0.3 else "#c9d1d9"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_title("Average Sharpe Ratio by Market Regime", fontsize=14,
                 fontweight="bold", pad=15)
    fig.colorbar(im, ax=ax, label="Sharpe Ratio", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Regime heatmap → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PSR Scatter Plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_psr_scatter(master_df, output_path):
    """Sharpe vs MinTRL scatter with significance threshold."""
    df = master_df.copy()
    df = df[df["MinTRL_y"] < 500]  # exclude inf

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        color = get_color(row["Strategy"])
        marker = "D" if "LLM" in row["Strategy"] else ("s" if "RL-" in row["Strategy"] else "o")
        size = 120 if "LLM" in row["Strategy"] else 80
        ax.scatter(row["Sharpe"], row["MinTRL_y"], c=color, s=size,
                   marker=marker, edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(row["Strategy"], (row["Sharpe"], row["MinTRL_y"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")

    # Reference lines
    ax.axhline(y=20, color="#3fb950", linestyle="--", alpha=0.5, label="20y data available")
    ax.axvline(x=0, color="#f85149", linestyle=":", alpha=0.4)

    ax.set_xlabel("Observed Sharpe Ratio", fontsize=12)
    ax.set_ylabel("Min Track Record Length (years)", fontsize=12)
    ax.set_title("PSR Significance: How Long to Trust the Sharpe?", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ PSR scatter → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Risk Overlay Comparison
# ═══════════════════════════════════════════════════════════════════════════

def compute_overlay_table(all_results):
    """Risk overlay before/after for all strategies."""
    rows = []
    for strat_name, strat_data in sorted(all_results.items()):
        curves = extract_equity_curves(strat_data)
        if not curves:
            continue

        raw_srs, adj_srs, raw_dds, adj_dds = [], [], [], []

        for ticker, equity in curves.items():
            if len(equity) < 35:
                continue
            try:
                r = apply_risk_overlay(equity, sharpe_window=30,
                                       sharpe_upper=0.0, sharpe_lower=-1.0,
                                       max_drawdown_pct=30.0)
                raw_srs.append(r["raw_metrics"]["sharpe"])
                adj_srs.append(r["adjusted_metrics"]["sharpe"])
                raw_dds.append(r["raw_metrics"]["max_drawdown_pct"])
                adj_dds.append(r["adjusted_metrics"]["max_drawdown_pct"])
            except Exception:
                continue

        if raw_srs:
            rows.append({
                "Strategy": get_display_name(strat_name),
                "Raw_SR": round(np.mean(raw_srs), 3),
                "Adj_SR": round(np.mean(adj_srs), 3),
                "Delta_SR": round(np.mean(adj_srs) - np.mean(raw_srs), 3),
                "Raw_DD": round(np.mean(raw_dds), 1),
                "Adj_DD": round(np.mean(adj_dds), 1),
                "Delta_DD": round(np.mean(adj_dds) - np.mean(raw_dds), 1),
            })
    return pd.DataFrame(rows)


def plot_overlay_bar(overlay_df, output_path):
    """Bar chart showing Sharpe improvement from risk overlay."""
    df = overlay_df.sort_values("Delta_SR", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.4)))
    colors = [get_color(s) for s in df["Strategy"]]
    bars = ax.barh(df["Strategy"], df["Delta_SR"], color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6)

    for bar, val in zip(bars, df["Delta_SR"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"+{val:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("ΔSharpe (Adjusted − Raw)", fontsize=12)
    ax.set_title("Risk Overlay Impact: Sharpe Ratio Improvement", fontsize=14,
                 fontweight="bold")
    ax.axvline(x=0, color="#f85149", linewidth=1, alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Overlay bar chart → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Bootstrap RCS Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_rcs(sharpe_records_path, spx_path, n_boot=5000, ci=0.95):
    """Block-bootstrap RCS scores with confidence intervals."""
    with open(sharpe_records_path) as f:
        records = json.load(f)

    spx = pd.read_csv(spx_path)
    regimes_list = spx["Market"].values  # array of regime labels per year

    strategies = [r["Strategy"] for r in records]
    regimes = ["Bull", "Sideways", "Bear"]

    # Observed RCS
    total = len(regimes_list)
    obs_weights = {reg: np.sum(regimes_list == reg) / total for reg in regimes}
    obs_rcs = {}
    for r in records:
        obs_rcs[r["Strategy"]] = sum(obs_weights[reg] * r[reg] for reg in regimes)

    # Bootstrap: resample years with replacement, recompute weights → RCS
    rng = np.random.default_rng(42)
    boot_rcs = {s: [] for s in strategies}

    for _ in range(n_boot):
        idx = rng.choice(total, size=total, replace=True)
        boot_regimes = regimes_list[idx]
        boot_weights = {reg: np.sum(boot_regimes == reg) / total for reg in regimes}
        for r in records:
            val = sum(boot_weights[reg] * r[reg] for reg in regimes)
            boot_rcs[r["Strategy"]].append(val)

    alpha = (1 - ci) / 2
    rows = []
    for s in strategies:
        arr = np.array(boot_rcs[s])
        lo = np.percentile(arr, alpha * 100)
        hi = np.percentile(arr, (1 - alpha) * 100)
        p_neg = np.mean(arr < 0)
        p_pos = np.mean(arr > 0)
        rows.append({
            "Strategy": s,
            "RCS": round(obs_rcs[s], 4),
            "CI_lo": round(lo, 4),
            "CI_hi": round(hi, 4),
            "CI_width": round(hi - lo, 4),
            "P(RCS>0)": round(p_pos, 4),
            "P(RCS<0)": round(p_neg, 4),
        })

    return pd.DataFrame(rows).sort_values("RCS", ascending=False)


def plot_rcs_forest(rcs_df, output_path):
    """Forest plot of RCS with bootstrap CIs."""
    df = rcs_df.sort_values("RCS", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.4)))

    for i, row in df.iterrows():
        color = "#f85149" if row["RCS"] < 0 else "#3fb950"
        ax.plot([row["CI_lo"], row["CI_hi"]], [i, i], color=color,
                linewidth=2, solid_capstyle="round")
        ax.scatter(row["RCS"], i, color=color, s=80, zorder=5,
                   edgecolors="white", linewidth=0.5)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Strategy"], fontsize=10)
    ax.axvline(x=0, color="#f85149", linestyle="--", alpha=0.6)
    ax.set_xlabel("Regime-Conditional Sharpe (RCS)", fontsize=12)
    ax.set_title("RCS with 95% Bootstrap Confidence Intervals", fontsize=14,
                 fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ RCS forest plot → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cost-Adjusted Sharpe
# ═══════════════════════════════════════════════════════════════════════════

# Estimated API costs per-ticker for full composite run (from paper Appendix G)
API_COSTS = {
    "FinMem (LLM)": 300.0,      # ~$300/ticker with GPT-4o-mini
    "FinAgent (LLM)": 250.0,    # ~$250/ticker with GPT-4o-mini
}
INITIAL_PORTFOLIO = 100000.0


def add_cost_adjusted_sharpe(master_df):
    """Add CAS column: Sharpe / (1 + C_api / V0)."""
    cas_values = []
    for _, row in master_df.iterrows():
        cost = API_COSTS.get(row["Strategy"], 0.0)
        denominator = 1 + cost / INITIAL_PORTFOLIO
        cas_values.append(round(row["Sharpe"] / denominator, 4))
    master_df["CAS"] = cas_values
    return master_df


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FINSABER Unified Dashboard")
    parser.add_argument("--setups", nargs="+",
                        default=["lowvol_sp500_5"],
                        help="Setup names to analyse")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_dir = os.path.join("backtest", "output")
    sharpe_path = os.path.join(base_dir, "sharpe_records.json")
    spx_path = os.path.join(base_dir, "SPX_Classification.csv")

    print("\n" + "=" * 80)
    print("  FINSABER Unified Analysis Dashboard")
    print("=" * 80)

    # ── 1. Master Metrics Table ───────────────────────────────────────────
    all_master = []
    all_overlay = []

    for setup in args.setups:
        setup_dir = os.path.join(base_dir, setup)
        if not os.path.isdir(setup_dir):
            print(f"  ⚠ Skipping {setup} (not found)")
            continue

        print(f"\n  Loading: {setup}")
        results = load_setup_results(setup_dir)
        print(f"    Found {len(results)} strategies")

        master = compute_master_table(results, setup)
        all_master.append(master)

        overlay = compute_overlay_table(results)
        overlay["Setup"] = setup
        all_overlay.append(overlay)

    if not all_master:
        print("  ERROR: No results loaded.")
        sys.exit(1)

    master_df = pd.concat(all_master, ignore_index=True)
    master_df = add_cost_adjusted_sharpe(master_df)

    # Print master table
    print("\n" + "─" * 120)
    print("  MASTER METRICS TABLE")
    print("─" * 120)
    cols = ["Strategy", "Setup", "Sharpe", "Sortino", "Calmar", "Omega",
            "PSR", "MinTRL_y", "CAS", "Ann_Ret", "Max_DD", "N_obs"]
    print(master_df[cols].to_string(index=False))

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "master_metrics.csv")
    master_df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Master table → {csv_path}")

    # ── 2. Regime Heatmap ─────────────────────────────────────────────────
    if os.path.exists(sharpe_path):
        plot_regime_heatmap(sharpe_path,
                           os.path.join(OUTPUT_DIR, "regime_heatmap.png"))

    # ── 3. PSR Scatter ────────────────────────────────────────────────────
    plot_psr_scatter(master_df, os.path.join(OUTPUT_DIR, "psr_scatter.png"))

    # ── 4. Risk Overlay ───────────────────────────────────────────────────
    if all_overlay:
        overlay_df = pd.concat(all_overlay, ignore_index=True)
        print("\n" + "─" * 90)
        print("  RISK OVERLAY: BEFORE vs AFTER")
        print("─" * 90)
        print(overlay_df.sort_values("Delta_SR", ascending=False).to_string(index=False))

        overlay_csv = os.path.join(OUTPUT_DIR, "risk_overlay.csv")
        overlay_df.to_csv(overlay_csv, index=False)
        print(f"\n  ✓ Overlay table → {overlay_csv}")

        plot_overlay_bar(overlay_df,
                         os.path.join(OUTPUT_DIR, "overlay_improvement.png"))

    # ── 5. Bootstrap RCS ──────────────────────────────────────────────────
    if os.path.exists(sharpe_path) and os.path.exists(spx_path):
        print("\n  Running RCS bootstrap (5000 iterations)...")
        rcs_df = bootstrap_rcs(sharpe_path, spx_path, n_boot=5000)

        print("\n" + "─" * 90)
        print("  RCS WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
        print("─" * 90)
        print(rcs_df.to_string(index=False))

        rcs_csv = os.path.join(OUTPUT_DIR, "rcs_bootstrap.csv")
        rcs_df.to_csv(rcs_csv, index=False)
        print(f"\n  ✓ RCS bootstrap → {rcs_csv}")

        plot_rcs_forest(rcs_df, os.path.join(OUTPUT_DIR, "rcs_forest.png"))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print("  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:40s} {size_kb:6.1f} KB")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
