# FINSABER — Change History

A virtual diary of all modifications made to this project since cloning.

---

## 2026-05-05 22:00 IST — Initial Setup & Dependency Installation

**Context**: Cloned repo, no conda available. System has Python 3.13.4 (repo recommends 3.10).

### Actions:
- Created `.env` from `.env.example` (API keys left blank)
- Installed core pip dependencies since conda was unavailable:
  - `backtrader`, `python-dotenv`, `tqdm`, `tabulate`, `huggingface_hub`
  - `datasets`, `pandas_datareader`, `statsmodels`, `xgboost`
  - `openai`, `colorlog`
  - (numpy, pandas, matplotlib, requests were already present)

### Note:
- `pip install finsaber` failed due to NumPy compilation issues (needs GCC on Python 3.13)
- `pwb_toolbox` not installed (optional, shows harmless warning)
- RL deps (`stable_baselines3`) and full LLM deps (`sentence_transformers`, `toml`, `chromadb`) **not** installed

---

## 2026-05-05 22:20 IST — Import Guard Fixes (4 files modified)

**Problem**: The codebase has hard imports of optional heavy dependencies (RL, LLM packages) at the module level. This prevents non-LLM baseline experiments from running when those packages aren't installed.

**Fix**: Wrapped optional imports in `try/except ImportError` blocks so the rest of the framework loads cleanly.

### File: `backtest/strategy/timing/__init__.py`

```diff
-from .finrl import FinRLStrategy
+try:
+    from .finrl import FinRLStrategy
+except ImportError:
+    pass
```

**Reason**: `FinRLStrategy` imports `stable_baselines3` and `rl_traders`, which are heavy RL dependencies not needed for baseline strategies.

### File: `backtest/strategy/selection/__init__.py`

```diff
-from .fincon_agent_selector import FinConSP500Selector
+try:
+    from .fincon_agent_selector import FinConSP500Selector
+except ImportError:
+    pass
```

**Reason**: `FinConSP500Selector` transitively imports `sentence_transformers` and other LLM packages via `llm_traders.fincon_selector`.

### File: `backtest/strategy/timing_llm/__init__.py`

```diff
 from .base_strategy_iso import BaseStrategyIso
-from .finmem import FinMemStrategy
-from .finagent import FinAgentStrategy
+try:
+    from .finmem import FinMemStrategy
+except ImportError:
+    pass
+try:
+    from .finagent import FinAgentStrategy
+except ImportError:
+    pass
```

**Reason**: `FinMemStrategy` requires `toml`; `FinAgentStrategy` requires additional LLM deps. Both are unnecessary for non-LLM baseline experiments.

### File: `backtest/experiment_runner.py`

```diff
-from backtest.strategy.selection import RandomSP500Selector, MomentumSP500Selector, LowVolatilitySP500Selector, FinConSP500Selector
+from backtest.strategy.selection import RandomSP500Selector, MomentumSP500Selector, LowVolatilitySP500Selector
+try:
+    from backtest.strategy.selection import FinConSP500Selector
+except ImportError:
+    FinConSP500Selector = None
```

**Reason**: `experiment_runner.py` explicitly imports `FinConSP500Selector`, which fails when LLM deps are missing. Setting it to `None` allows non-FinCon setups to work.

---

## 2026-05-05 22:35 IST — Dataset Downloads

### Downloaded from HuggingFace (`waylonli/FINSABER-data`):

1. **Cherry-pick dataset** (~86MB)
   - Source: `data/finmem_data/stock_data_cherrypick_2000_2024_v2.pkl`
   - Saved as: `data/finmem_data/stock_data_cherrypick_2000_2024_v2.pkl`
   - Copied to: `data/finmem_data/stock_data_cherrypick_2000_2024.pkl`
   - (Copy needed because code references the filename without `_v2`)

2. **S&P500 price CSV** (~253MB)
   - Source: `data/price/all_sp500_prices_2000_2024_delisted_include.csv`
   - Saved as: `data/price/all_sp500_prices_2000_2024_delisted_include.csv`

### NOT downloaded:
- `stock_data_sp500_2000_2024.pkl` (~13GB) — needed for composite setups

---

## 2026-05-05 22:45 IST — First Successful Run

**Command**:
```bash
$env:PYTHONPATH = "."; python backtest/run_baselines_exp.py --setup cherry_pick_both_finmem --include BuyAndHoldStrategy --date_from 2022-10-06 --date_to 2023-04-10
```

**Result**: ✅ Success — BuyAndHoldStrategy ran on TSLA, NFLX, AMZN, MSFT (COIN skipped due to insufficient prior data).

### Output files overwritten:
- `backtest/output/cherry_pick_both_finmem/BuyAndHoldStrategy/results.csv`
- `backtest/output/cherry_pick_both_finmem/BuyAndHoldStrategy/2022-10-06_2023-04-10.pkl`

### Results:
| Ticker | Total Return | Annual Return | Sharpe | Max Drawdown |
|--------|-------------|---------------|--------|-------------|
| TSLA   | -20.48%     | -27.19%       | -0.34  | 52.73%      |
| NFLX   | +43.08%     | +64.21%       | 1.33   | 20.18%      |
| AMZN   | -13.25%     | -17.87%       | -0.46  | 31.55%      |
| MSFT   | +21.13%     | +30.40%       | 0.97   | 14.16%      |
| **Avg**| **+7.62%**  | **+12.39%**   |**0.37**| **29.66%**  |

---

## Summary of All Modified Files

| File | Type of Change |
|------|---------------|
| `backtest/strategy/timing/__init__.py` | Import guard for FinRLStrategy |
| `backtest/strategy/selection/__init__.py` | Import guard for FinConSP500Selector |
| `backtest/strategy/timing_llm/__init__.py` | Import guards for FinMemStrategy, FinAgentStrategy |
| `backtest/experiment_runner.py` | Import guard for FinConSP500Selector |
| `backtest/output/cherry_pick_both_finmem/BuyAndHoldStrategy/results.csv` | Overwritten by fresh run |
| `backtest/output/cherry_pick_both_finmem/BuyAndHoldStrategy/2022-10-06_2023-04-10.pkl` | Overwritten by fresh run |
| `.env` | Created from `.env.example` (untracked, in .gitignore) |
| `data/finmem_data/stock_data_cherrypick_2000_2024.pkl` | Downloaded (untracked) |
| `data/finmem_data/stock_data_cherrypick_2000_2024_v2.pkl` | Downloaded (untracked) |
| `data/price/all_sp500_prices_2000_2024_delisted_include.csv` | Downloaded (untracked) |

---

## 2026-05-06 00:35 IST — Implementation Plans Created

**Context**: Analysed the paper's experimental gaps and created 5 detailed implementation plans for high-ROI extensions, biased toward mathematical contributions.

### Files Created:
All under `plans/` directory (new):

| File | Title | Category | Effort |
|------|-------|----------|--------|
| `01_calmar_omega_ratios.md` | Calmar & Omega Ratios as Primary Metrics | Quant Math | ~20h |
| `02_regime_conditional_sharpe.md` | Regime-Conditional Sharpe (RCS) Metric | Quant Math | ~30h |
| `03_probabilistic_sharpe_ratio.md` | Probabilistic Sharpe Ratio & Min Track Record | Quant Math | ~45h |
| `04_rolling_sharpe_stop_loss.md` | Rolling Sharpe Drawdown-Triggered Stop Loss | Pipeline | ~25h |
| `05_regime_conditioned_prompting.md` | Regime-Conditioned Prompting with Volatility Signal | LLM Design | ~30h |

### Key integration points identified per plan:
- **Metrics**: `backtest/toolkit/metrics.py` (currently 38 lines with just Sortino + annual vol)
- **Metric wiring**: `backtest/finsaber_bt.py` → `_calculate_annualized_metrics()` (L330) and `_analyze_results()` (L269)
- **Results aggregation**: `backtest/toolkit/operation_utils.py` → `aggregate_results_one_strategy()` (L495)
- **LLM prompts**: `llm_traders/finmem/` and `llm_traders/finagent/`
- **Regime data**: `backtest/output/SPX_Classification.csv` and `sharpe_records.json` (pre-existing)

---

## 2026-05-06 01:20 IST — Plan 01: Calmar & Omega Ratios Implemented

**Context**: First planned extension. Added Calmar Ratio (AR / MDD) and Omega Ratio (probability-weighted gains vs losses) as primary evaluation metrics alongside existing Sharpe and Sortino.

### Mathematical Definitions

- **Calmar Ratio** = Annualised Return / Maximum Drawdown — penalises strategies with high drawdowns
- **Omega Ratio** = Σ(gains above threshold) / Σ(losses below threshold) — evaluates the entire return distribution, superior for non-normal returns. Omega > 1.0 means gains outweigh losses.

### Files Modified

| File | Change |
|------|--------|
| `backtest/toolkit/metrics.py` | Added `calculate_calmar_ratio()` and `calculate_omega_ratio()` functions |
| `backtest/finsaber_bt.py` | Wired new metrics into `_calculate_annualized_metrics()` (computation), `_analyze_results()` (print + return dict) |
| `backtest/toolkit/operation_utils.py` | Added `calmar_ratio` and `omega_ratio` columns to both `aggregate_results()` and `aggregate_results_one_strategy()`, with `.get()` fallbacks for backward compatibility with old pickle files |

### Verification Run

```bash
PYTHONPATH=. python backtest/run_baselines_exp.py \
    --setup cherry_pick_both_finmem \
    --include BuyAndHoldStrategy \
    --date_from 2022-10-06 --date_to 2023-04-10
```

### Results Comparison: Baseline vs Plan 01

**Baseline (before — only Sharpe/Sortino):**

| Ticker | Total Return | Annual Return | Sharpe | Sortino | Max Drawdown |
|--------|-------------|---------------|--------|---------|-------------|
| TSLA   | -20.48%     | -27.19%       | -0.34  | -0.47   | 52.73%      |
| NFLX   | +43.08%     | +64.21%       | 1.33   | 2.32    | 20.18%      |
| AMZN   | -13.25%     | -17.87%       | -0.46  | -0.68   | 31.55%      |
| MSFT   | +21.13%     | +30.40%       | 0.97   | 1.50    | 14.16%      |
| **Avg**| **+7.62%**  | **+12.39%**   |**0.37**|**0.67** | **29.66%**  |

**Plan 01 (after — with Calmar & Omega):**

| Ticker | Total Return | Annual Return | Sharpe | Sortino | **Calmar** | **Omega** | Max Drawdown |
|--------|-------------|---------------|--------|---------|-----------|----------|-------------|
| TSLA   | -20.48%     | -27.19%       | -0.34  | -0.47   | **-0.516**| **0.934**| 52.73%      |
| NFLX   | +43.08%     | +64.21%       | 1.33   | 2.32    | **3.181** | **1.326**| 20.18%      |
| AMZN   | -13.25%     | -17.87%       | -0.46  | -0.68   | **-0.566**| **0.911**| 31.55%      |
| MSFT   | +21.13%     | +30.40%       | 0.97   | 1.50    | **2.146** | **1.217**| 14.16%      |
| **Avg**| **+7.62%**  | **+12.39%**   |**0.37**|**0.67** |**1.061**  |**1.097** | **29.66%**  |

### Key Observations

1. **Calmar exposes drawdown severity**: TSLA's Calmar (-0.516) is worse than its Sharpe (-0.34) because Calmar directly penalises the 52.73% drawdown. NFLX's Calmar (3.181) is much better than its Sharpe (1.33) because its drawdown was moderate relative to its return.
2. **Omega reveals asymmetry**: TSLA's Omega (0.934 < 1.0) confirms losses outweigh gains at the risk-free threshold. NFLX's Omega (1.326 > 1.0) confirms gains dominate.
3. **Backward compatible**: Old pickle files without `calmar_ratio`/`omega_ratio` keys default to 0 via `.get()` fallbacks — no breakage.

