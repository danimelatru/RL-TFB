#!/bin/bash
#SBATCH -J rl_screen             # Job name
#SBATCH -p cpu_med               # Partition (4h limit)
#SBATCH --time=03:00:00          # 3h — screen-v0 Q-table can grow very large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                 # More memory for large screen Q-tables
#SBATCH -o logs/screen_%j.out
#SBATCH -e logs/screen_%j.err

echo "=== Screen experiment started at $(date) ==="
echo "Node: $(hostname)"
cd /gpfs/workdir/fernandeda/rl_assignment

module purge
source activate /gpfs/workdir/fernandeda/conda_envs/rl_assignment

mkdir -p results logs

python experiments/run_generalization.py \
    --num_episodes 50000 \
    --seed 42 \
    --experiment screen \
    --output_dir results

echo "=== Finished at $(date) ==="
