#!/bin/bash
#SBATCH -J rl_generalization     # Job name
#SBATCH -p cpu_med               # Partition (4h limit)
#SBATCH --time=01:30:00          # 1.5h — MC+Sarsa training (50k) + 5 configs × 100 eval (capped)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o logs/generalization_%j.out
#SBATCH -e logs/generalization_%j.err

echo "=== Generalization experiment started at $(date) ==="
echo "Node: $(hostname)"
cd /gpfs/workdir/fernandeda/rl_assignment

module purge
source activate /gpfs/workdir/fernandeda/conda_envs/rl_assignment

mkdir -p results logs

python experiments/run_generalization.py \
    --num_episodes 50000 \
    --seed 42 \
    --experiment generalization \
    --output_dir results

echo "=== Finished at $(date) ==="
