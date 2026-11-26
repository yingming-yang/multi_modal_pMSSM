#!/bin/bash
# =============================================================================
# Time one training step at a range of batch sizes and extrapolate the cost of a
# full run. Reads the same yaml as the training job, so the benchmarked model
# matches the model that will actually be trained.
#
# Submit:
#   sbatch slurm/benchmark_batch_size.sbatch
# Extra arguments are forwarded to the python script, e.g.
#   sbatch slurm/benchmark_batch_size.sbatch --batch_sizes 512 4096 --timed_steps 100
#   sbatch slurm/benchmark_batch_size.sbatch --data_fraction 0.05    # fast dry run
# =============================================================================
#SBATCH --job-name=nf_bs_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJ_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONFIG="${PROJ_DIR}/slurm/train_norm_flow.yaml"
SCRIPT="${PROJ_DIR}/normalizing_flow/benchmark_batch_size.py"
LOG_DIR="${PROJ_DIR}/logs"

mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/bs_bench_${SLURM_JOB_ID}.out" 2>"${LOG_DIR}/bs_bench_${SLURM_JOB_ID}.err"

echo "=========================================="
echo "[bs-bench] job_id=${SLURM_JOB_ID}"
echo "[bs-bench] CONFIG=${CONFIG}"
echo "[bs-bench] started: $(date)"
echo "=========================================="

cd "${PROJ_DIR}"
pixi run python -u "${SCRIPT}" --config "${CONFIG}" "$@"

echo "=========================================="
echo "[bs-bench] finished: $(date)"
echo "=========================================="
