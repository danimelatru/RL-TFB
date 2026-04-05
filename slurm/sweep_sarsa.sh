#!/bin/bash
#SBATCH -J rl_sweep_sarsa        # Job name
#SBATCH -p cpu_med               # Partition (4h limit, faster nodes)
#SBATCH --time=01:30:00          # 1.5h — ~15min training + buffer for slow nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-23             # 4 alphas × 6 lambdas = 24 configs
#SBATCH -o logs/sweep_sarsa_%A_%a.out
#SBATCH -e logs/sweep_sarsa_%A_%a.err

echo "=== Sarsa sweep config $SLURM_ARRAY_TASK_ID started at $(date) ==="
echo "Node: $(hostname)"
cd /gpfs/workdir/fernandeda/rl_assignment

module purge
source activate /gpfs/workdir/fernandeda/conda_envs/rl_assignment

mkdir -p results logs

python experiments/run_sweep.py \
    --agent sarsa \
    --config_id $SLURM_ARRAY_TASK_ID \
    --num_episodes 30000 \
    --n_seeds 3 \
    --output_dir results

echo "=== Finished at $(date) ==="
