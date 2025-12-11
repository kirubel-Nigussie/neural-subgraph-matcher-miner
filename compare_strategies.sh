#!/bin/bash

# Dataset and Model
DATASET="data/CA-GrQc.pkl"
MODEL="ckpt/model.pt"
TRIALS=100
# TRIALS=10

# Create directories
mkdir -p logs
mkdir -p plots/cluster/greedy plots/cluster/mcts plots/cluster/beam
mkdir -p results/greedy results/mcts results/beam

echo "============================================"
echo "Running Comparison of Search Strategies"
echo "Dataset: $DATASET"
echo "============================================"

# 1. Greedy Search
echo ""
echo "[1/3] Running GREEDY Search..."
start_time=$(date +%s)
python -m subgraph_mining.decoder --dataset=$DATASET --model_path=$MODEL --n_trials=$TRIALS --graph_type=undirected --search_strategy=greedy | tee logs/greedy.log
mv plots/cluster/*.html plots/cluster/greedy/ 2>/dev/null
mv results/out-patterns* results/greedy/ 2>/dev/null
end_time=$(date +%s)
echo "Done. Time taken: $((end_time - start_time)) seconds."
echo "Log saved to logs/greedy.log"
echo "Plots saved to plots/cluster/greedy/"
echo "Results saved to results/greedy/"

# 2. MCTS (Monte Carlo Tree Search)
echo ""
echo "[2/3] Running MCTS..."
start_time=$(date +%s)
python -m subgraph_mining.decoder --dataset=$DATASET --model_path=$MODEL --n_trials=$TRIALS --graph_type=undirected --search_strategy=mcts | tee logs/mcts.log
mv plots/cluster/*.html plots/cluster/mcts/ 2>/dev/null
mv results/out-patterns* results/mcts/ 2>/dev/null
end_time=$(date +%s)
echo "Done. Time taken: $((end_time - start_time)) seconds."
echo "Log saved to logs/mcts.log"
echo "Plots saved to plots/cluster/mcts/"
echo "Results saved to results/mcts/"



# 3. Beam Search
echo ""
echo "[3/3] Running BEAM Search..."
start_time=$(date +%s)
python -m subgraph_mining.decoder \
    --dataset=data/CA-GrQc.pkl \
    --model_path=ckpt/model.pt \
    --n_trials=1 \
    --graph_type=undirected \
    --search_strategy=beam \
    --beam_width=5 \
    --max_pattern_size=5 \
    | tee logs/beam.log
mv plots/cluster/*.html plots/cluster/beam/ 2>/dev/null
mv results/out-patterns* results/beam/ 2>/dev/null
end_time=$(date +%s)
echo "Done. Time taken: $((end_time - start_time)) seconds."
echo "Log saved to logs/beam.log"
echo "Plots saved to plots/cluster/beam/"
echo "Results saved to results/beam/"

echo ""
echo "============================================"
echo "Comparison Complete!"
echo "Check the 'logs/' directory for details."
echo "============================================"
