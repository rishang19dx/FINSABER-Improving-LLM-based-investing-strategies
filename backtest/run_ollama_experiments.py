"""
Ollama LLM Experiment Runner
=============================

Runs FINSABER experiments with locally-hosted Ollama models.
Executes FinMem and/or FinAgent strategies on the cherry-pick setup.

Usage::

    # FinMem with qwen2.5:3b (full cherry-pick, 5 stocks)
    PYTHONPATH=. python backtest/run_ollama_experiments.py --strategy finmem --model qwen

    # FinMem smoke test (1 stock, TSLA only)
    PYTHONPATH=. python backtest/run_ollama_experiments.py --strategy finmem --model qwen --smoke

    # FinAgent with qwen2.5:3b
    PYTHONPATH=. python backtest/run_ollama_experiments.py --strategy finagent --model qwen

    # Both strategies
    PYTHONPATH=. python backtest/run_ollama_experiments.py --strategy both --model qwen
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_util import FinMemDataset
from backtest.finsaber import FINSABER
from backtest.toolkit.operation_utils import aggregate_results_one_strategy


MODEL_CONFIGS = {
    "qwen": {
        "finmem_config_path": "strats_configs/finmem_ollama_qwen_config.toml",
        "finmem_strat_json": "strats_configs/finmem_ollama_qwen_cherry.json",
        "finagent_strat_json": "strats_configs/finagent_ollama_qwen_cherry.json",
        "display_name": "qwen2.5:3b",
    },
}


def run_finmem(model_key: str, smoke: bool = False):
    """Run FinMem strategy with the specified Ollama model."""
    from backtest.strategy.timing_llm.finmem import FinMemStrategy

    cfg = MODEL_CONFIGS[model_key]
    setup_name = f"cherry_pick_ollama_{model_key}_finmem"

    if smoke:
        tickers = ["TSLA"]
        date_from = "2023-01-01"
        date_to = "2023-02-01"
        setup_name += "_smoke"
    else:
        tickers = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]
        date_from = "2022-10-06"
        date_to = "2023-04-10"

    print("\n" + "=" * 70)
    print(f"  FINSABER × Ollama — FinMem with {cfg['display_name']}")
    print(f"  Tickers: {tickers}")
    print(f"  Period:  {date_from} → {date_to}")
    print(f"  Setup:   {setup_name}")
    print("=" * 70 + "\n")

    trade_config = {
        "tickers": tickers,
        "silence": False,
        "setup_name": setup_name,
        "date_from": date_from,
        "date_to": date_to,
        "data_loader": FinMemDataset(
            pickle_file="data/finmem_data/stock_data_cherrypick_2000_2024.pkl"
        ),
    }

    strat_params = {
        "config_path": cfg["finmem_config_path"],
        "market_data_info_path": "data/finmem_data/stock_data_cherrypick_2000_2024.pkl",
        "date_from": "$date_from",
        "date_to": "$date_to",
        "symbol": "$symbol",
        "training_period": ["2021-08-17", "2022-10-05"],
    }

    engine = FINSABER(trade_config)
    start_time = time.time()

    try:
        ticker_metrics = engine.run_iterative_tickers(
            FinMemStrategy, strat_params=strat_params
        )
        elapsed = time.time() - start_time
        print(f"\n  ✓ FinMem completed in {elapsed/60:.1f} minutes")
        print(f"  Results: {ticker_metrics}")

        # Aggregate
        try:
            aggregate_results_one_strategy(
                setup_name, "FinMemStrategy", output_dir="backtest/output"
            )
        except Exception as e:
            print(f"  Warning: aggregation failed: {e}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ✗ FinMem FAILED after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()

    return setup_name


def run_finagent(model_key: str, smoke: bool = False):
    """Run FinAgent strategy with the specified Ollama model."""
    from backtest.strategy.timing_llm.finagent import FinAgentStrategy

    cfg = MODEL_CONFIGS[model_key]
    setup_name = f"cherry_pick_ollama_{model_key}_finagent"

    if smoke:
        tickers = ["TSLA"]
        date_from = "2023-01-01"
        date_to = "2023-02-01"
        setup_name += "_smoke"
    else:
        tickers = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]
        date_from = "2022-10-06"
        date_to = "2023-04-10"

    print("\n" + "=" * 70)
    print(f"  FINSABER × Ollama — FinAgent with {cfg['display_name']}")
    print(f"  Tickers: {tickers}")
    print(f"  Period:  {date_from} → {date_to}")
    print(f"  Setup:   {setup_name}")
    print("=" * 70 + "\n")

    trade_config = {
        "tickers": tickers,
        "silence": False,
        "setup_name": setup_name,
        "date_from": date_from,
        "date_to": date_to,
        "data_loader": FinMemDataset(
            pickle_file="data/finmem_data/stock_data_cherrypick_2000_2024.pkl"
        ),
    }

    strat_params = {
        "market_data_info_path": "data/finmem_data/stock_data_cherrypick_2000_2024.pkl",
        "date_from": "$date_from",
        "date_to": "$date_to",
        "symbol": "$symbol",
        "training_period": ["2021-08-17", "2022-10-05"],
        "provider_config": "llm_traders/finagent/configs/ollama_qwen_config.json",
        "llm_model_id": "qwen2.5:3b",
    }

    engine = FINSABER(trade_config)
    start_time = time.time()

    try:
        ticker_metrics = engine.run_iterative_tickers(
            FinAgentStrategy, strat_params=strat_params
        )
        elapsed = time.time() - start_time
        print(f"\n  ✓ FinAgent completed in {elapsed/60:.1f} minutes")
        print(f"  Results: {ticker_metrics}")

        try:
            aggregate_results_one_strategy(
                setup_name, "FinAgentStrategy", output_dir="backtest/output"
            )
        except Exception as e:
            print(f"  Warning: aggregation failed: {e}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ✗ FinAgent FAILED after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()

    return setup_name


def main():
    parser = argparse.ArgumentParser(description="FINSABER Ollama Experiment Runner")
    parser.add_argument(
        "--strategy",
        choices=["finmem", "finagent", "both"],
        default="finmem",
        help="Which strategy to run (default: finmem)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="qwen",
        help="Ollama model to use (default: qwen)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick smoke test (1 stock, 1 month)",
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("█" + " FINSABER × Ollama Experiment Suite ".center(68) + "█")
    print("█" * 70)
    print(f"  Model:    {MODEL_CONFIGS[args.model]['display_name']}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Mode:     {'SMOKE TEST' if args.smoke else 'FULL EXPERIMENT'}")
    print("█" * 70 + "\n")

    setup_names = []

    if args.strategy in ("finmem", "both"):
        name = run_finmem(args.model, smoke=args.smoke)
        setup_names.append(name)

    if args.strategy in ("finagent", "both"):
        name = run_finagent(args.model, smoke=args.smoke)
        setup_names.append(name)

    # ── Post-hoc analysis on completed experiments ────────────────────────
    print("\n" + "=" * 70)
    print("  Post-Hoc Analysis")
    print("=" * 70)

    for setup_name in setup_names:
        setup_dir = os.path.join("backtest", "output", setup_name)
        if os.path.isdir(setup_dir):
            print(f"\n  Analyzing: {setup_name}")
            try:
                from backtest.run_psr_analysis import main as psr_main
                sys.argv = ["", "--setup", setup_name]
                psr_main()
            except Exception as e:
                print(f"    PSR analysis skipped: {e}")

            try:
                from backtest.run_risk_overlay_analysis import main as overlay_main
                sys.argv = ["", "--setup", setup_name]
                overlay_main()
            except Exception as e:
                print(f"    Risk overlay analysis skipped: {e}")
        else:
            print(f"  ⚠ No output found for {setup_name}")

    print("\n" + "█" * 70)
    print("█" + " All experiments complete ".center(68) + "█")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
