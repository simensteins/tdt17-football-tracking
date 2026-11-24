#!/bin/bash
# Run any command on one IDUN GPU and log the GPU's power draw.
#
#   sbatch --job-name=<name> jobs/gpu.slurm <command> [args...]
#
# Power samples go to logs/<job name>-<job id>-gpu.csv (every 30 s), so energy
# use can be computed from measurements instead of rated wattage.

#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/tdt17/bin/activate
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=timestamp,name,power.draw,utilization.gpu,memory.used \
    --format=csv -l 30 > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu.csv" &
monitor=$!
trap 'kill $monitor' EXIT

echo "job $SLURM_JOB_ID on $(hostname), started $(date -Is)"
echo "command: $*"
"$@"
echo "finished $(date -Is)"
