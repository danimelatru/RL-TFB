#!/bin/bash
#SBATCH -J rl_sweep_mc           # Job name
#SBATCH -p cpu_short             # Partition
#SBATCH --time=00:45:00          # Max time
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --array=0-19             # 5 alphas × 4 gammas = 20 configs
#SBATCH -o logs/sweep_mc_%A_%a.out
#SBATCH -e logs/sweep_mc_%A_%a.err

echo "=== MC sweep config $SLURM_ARRAY_TASK_ID started at $(date) ==="
echo "Node: $(hostname)"
cd /gpfs/workdir/fernandeda/rl_assignment

module purge
source activate /gpfs/workdir/fernandeda/conda_envs/rl_assignment

mkdir -p results logs

python experiments/run_sweep.py \
    --agent mc \
    --config_id $SLURM_ARRAY_TASK_ID \
    --num_episodes 30000 \
    --n_seeds 3 \
    --output_dir results

echo "=== Finished at $(date) ==="
