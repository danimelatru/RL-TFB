#!/bin/bash
#SBATCH -J rl_extra              # Job name
#SBATCH -p cpu_short             # Partition
#SBATCH --time=00:50:00          # Max time
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -o logs/extra_%j.out
#SBATCH -e logs/extra_%j.err

echo "=== Extra experiments started at $(date) ==="
echo "Node: $(hostname)"
cd /gpfs/workdir/fernandeda/rl_assignment

module purge
source activate /gpfs/workdir/fernandeda/conda_envs/rl_assignment

mkdir -p results logs

python experiments/run_generalization.py \
    --num_episodes 50000 \
    --seed 42 \
    --experiment all \
    --output_dir results

echo "=== Finished at $(date) ==="
