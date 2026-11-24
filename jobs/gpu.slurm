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
# Any GPU with at least 40 GB; the job log and power log record which one ran.
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/tdt17/bin/activate
cd "$SLURM_SUBMIT_DIR"

# Ultralytics treats this as a non-interactive console and prints progress at
# most once a minute as plain lines, instead of redrawing the bar in the log.
export GITHUB_ACTIONS=true

nvidia-smi --query-gpu=timestamp,name,power.draw,utilization.gpu,memory.used \
    --format=csv -l 30 > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu.csv" &
monitor=$!
trap 'kill $monitor' EXIT

echo "job $SLURM_JOB_ID on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)), started $(date -Is)"
echo "command: $*"
"$@"
echo "finished $(date -Is)"
