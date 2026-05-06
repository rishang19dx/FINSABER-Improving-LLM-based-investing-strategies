#!/bin/bash
# run_all_and_shutdown.sh
# Runs full FINSABER Ollama experiments and shuts down the machine.

echo "Killing any existing experiments (e.g. smoke test)..."
pkill -f run_ollama_experiments.py
sleep 2

echo "Starting full Ollama experiments..."

# Navigate to repo
cd /home/rishang/Desktop/FINSABER/FINSABER

# Run full experiments with qwen
export QT_QPA_PLATFORM=offscreen
PYTHONPATH=. python backtest/run_ollama_experiments.py --strategy both --model qwen > data/full_experiment_run.log 2>&1

echo "Experiments finished. Initiating shutdown..."

# Shutdown
sudo shutdown -P now || systemctl poweroff || poweroff
