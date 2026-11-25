#!/bin/bash
# Run any command on IDUN CPUs, for work that needs no GPU (tracking, evaluation).
#
#   sbatch --job-name=<name> --time=<limit> jobs/cpu.slurm <command> [args...]

#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/tdt17/bin/activate
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

echo "job $SLURM_JOB_ID on $(hostname), started $(date -Is)"
echo "command: $*"
"$@"
echo "finished $(date -Is)"
