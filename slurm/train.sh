#!/bin/bash
#SBATCH -J rl_train              # Job name
#SBATCH -p cpu_short             # Partition (1h max)
#SBATCH --time=00:30:00          # Max time
#SBATCH --ntasks=1               # 1 task
#SBATCH --cpus-per-task=4        # 4 CPU cores
#SBATCH --mem=4G                 # Memory
#SBATCH -o logs/train_%j.out     # Stdout
#SBATCH -e logs/train_%j.err     # Stderr

# ── Setup ──────────────────────────────────────────────────────────
echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
cd /gpfs/workdir/fernandeda/rl_assignment
echo "Working dir: $(pwd)"

module purge
source activate /gpfs/workdir/fernandeda/conda_envs/rl_assignment

mkdir -p results/models logs

# ── Run training ───────────────────────────────────────────────────
# Train both agents with 3 seeds, 50K episodes
for SEED in 0 1 2; do
    echo ""
    echo "=== Seed $SEED ==="
    python experiments/run_training.py \
        --num_episodes 50000 \
        --seed $SEED \
        --agent both \
        --output_dir results
done

echo ""
echo "=== Job finished at $(date) ==="
