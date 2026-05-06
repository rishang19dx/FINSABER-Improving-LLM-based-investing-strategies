# MA546 Course Project: Evaluating LLM-based Financial Investing Strategies (FINSABER)

> **Course:** MA546 - Introduction to Quantitative Finance, IIT Mandi  
> **Based on Paper:** "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?" (KDD 2026)  

This repository contains our course project implementation based on the official **FINSABER** framework. We are reproducing the baseline experiments and extending the codebase to introduce advanced quantitative metrics, risk management mechanisms, and regime-conditioned LLM prompting.

---

## 📈 Project Roadmap & Implementation Plans

Five extensions to the baseline framework, biased toward mathematical contributions in quantitative finance. Detailed plans are in `plans/`:

| # | Extension | Status | Key Result |
|---|-----------|--------|------------|
| 1 | [Calmar & Omega Ratios](plans/01_calmar_omega_ratios.md) | ✅ Done | Calmar exposes TSLA's -52% drawdown; Omega confirms asymmetric losses |
| 2 | [Regime-Conditional Sharpe (RCS)](plans/02_regime_conditional_sharpe.md) | ✅ Done | FinMem RCS = **-0.1906** — negative across all regimes |
| 3 | [Probabilistic Sharpe Ratio & MinTRL](plans/03_probabilistic_sharpe_ratio.md) | ✅ Done | FinMem needs **328.7 years** to validate its Sharpe |
| 4 | [Rolling Sharpe Stop-Loss](plans/04_rolling_sharpe_stop_loss.md) | 📋 Planned | Dynamic risk overlay for LLM strategies |
| 5 | [Regime-Conditioned Prompting](plans/05_regime_conditioned_prompting.md) | 📋 Planned | Inject volatility signals into LLM prompts |

---

## 🧮 Extended Metrics & Analysis Tools

### Metrics integrated into the backtest engine

Every backtest run now automatically computes and reports these metrics alongside the original Sharpe/Sortino:

| Metric | Formula | What it reveals |
|--------|---------|-----------------|
| **Calmar Ratio** | Annual Return / Max Drawdown | Penalises high-drawdown strategies (LLMs in bear markets) |
| **Omega Ratio** | Σ(gains > τ) / Σ(losses ≤ τ) | Full return distribution — superior for non-normal returns |
| **PSR** | P(true SR > 0 \| data) | Statistical significance of observed Sharpe |
| **MinTRL** | Min observations for 95% confidence | How long you need to trust the Sharpe |

### Standalone analysis scripts

```bash
# Regime-Conditional Sharpe — ranks all 16 strategies by frequency-weighted regime Sharpe
PYTHONPATH=. python backtest/run_rcs_analysis.py

# Probabilistic Sharpe Ratio — loads pickle results and computes PSR/MinTRL
PYTHONPATH=. python backtest/run_psr_analysis.py --setup lowvol_sp500_5
PYTHONPATH=. python backtest/run_psr_analysis.py --setup cherry_pick_both_finmem
```

### Headline results

> **FinMem needs 328.7 years** of track record to validate its Sharpe of +0.097 at 95% confidence (excess kurtosis = +1289). The paper evaluated it over 6 months.

| Strategy | RCS | PSR | MinTRL | Verdict |
|----------|-----|-----|--------|---------|
| Buy & Hold | +0.4429 | 1.0000 | 9.2y | ✅ Statistically validated |
| ARIMA | +0.3776 | 1.0000 | 11.7y | ✅ Statistically validated |
| FinAgent | +0.1287 | 0.9939 | 34.9y | ⚠️ Barely significant |
| **FinMem** | **-0.1906** | **0.7927** | **328.7y** | ❌ Not trustworthy |

---

## 🛠️ Current Progress & Modifications

Detailed logs are available in [change_history.md](change_history.md).

**Completed:**
- **Environment Setup**: Initialized on Python 3.13 with `pip` (bypassing conda); configured `.env`
- **Framework Decoupling**: Import guards across `backtest/strategy/` to isolate heavy RL/LLM dependencies
- **Data Acquisition**: S&P 500 pricing (~253MB) + cherry-picked stocks (~86MB) from HuggingFace
- **Plan 01 — Calmar & Omega**: Added `calculate_calmar_ratio()` and `calculate_omega_ratio()` to `backtest/toolkit/metrics.py`, wired into engine + aggregation
- **Plan 02 — RCS**: Created `backtest/toolkit/rcs.py` (RegimeClassifier + compute_rcs) and `backtest/run_rcs_analysis.py`
- **Plan 03 — PSR & MinTRL**: Created `backtest/toolkit/psr.py` (Bailey & de Prado 2014) and `backtest/run_psr_analysis.py`, wired into engine + aggregation

---

## 📖 Original FINSABER Overview (Baseline)

FINSABER is a comprehensive framework for evaluating trading strategies with a specific focus on comparing traditional technical analysis approaches with modern machine learning and large language model (LLM) based strategies.

<img src="https://github.com/waylonli/FINSABER/blob/main/figs/framework.png" width="900">

### 1. Environment Setup (Original Instructions)

To set up the full environment for running LLM experiments:

```bash
git clone https://github.com/waylonli/FINSABER
cd FINSABER
conda create -n finsaber python=3.10
conda activate finsaber
pip install -r requirements-complete.txt --no-deps
```

*Note: For our project, we are using a lightweight pip environment. See `change_history.md` for our specific dependency setup.*

Rename `.env.example` to `.env` and set the environment variables.
- `OPENAI_API_KEY` is required to run LLM-based strategies.
- `HF_ACCESS_TOKEN` is optional.

### 2. Data

We provide aggregated datasets on [HuggingFace](https://huggingface.co/datasets/waylonli/FINSABER-data). Datasets are auto-downloaded when running experiments.

| Dataset | Content | Size |
| :--- | :--- | :--- |
| **S&P500 Full** | Aggregated data (Price + News + Filings) | ~13 GB |
| **Price Only** | CSV format price-only data | ~253 MB |
| **Selected Symbols** | Aggregated data for TSLA, AMZN, MSFT, NFLX, COIN | ~60 MB |

### 3. Reproducing Results

The paper contains three experimental setups: *selective (cherry picking) setup*, *selected-4 setup*, and *composite setup*.

**Baselines (non-LLM):**
```bash
python backtest/run_baselines_exp.py \
    --setup <setup_name> \
    --include <strategy_name> \
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --training_years 2 \
    --rolling_window_size 1 \
    --rolling_window_step 1
```

**LLM Strategies:**
```bash
python backtest/run_llm_traders_exp.py \
    --setup <setup_name> \
    --strategy <strategy_name> \
    --strat_config_path <config_path> \
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --rolling_window_size 1 \
    --rolling_window_step 1
```

### 4. Extending the Framework

You can plug in your own datasets and strategies by subclassing the helpers under `backtest/strategy` and `backtest/data_util`. 
- **Custom Timing**: Subclass `backtest.strategy.timing.base_strategy.BaseStrategy`.
- **Custom LLM Timing**: Inherit from `backtest.strategy.timing_llm.base_strategy_iso.BaseStrategyIso`.
- **Custom Selection**: Subclass `backtest.strategy.selection.base_selector.BaseSelector`.
- **Custom Dataset**: Derive from `backtest.data_util.backtest_dataset.BacktestDataset`.

### Citation

```
@misc{li2025llmbasedfinancialinvestingstrategies,
      title={Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?}, 
      author={Weixian Waylon Li and Hyeonjun Kim and Mihai Cucuringu and Tiejun Ma},
      year={2025},
      eprint={2505.07078},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2505.07078}, 
}
```
