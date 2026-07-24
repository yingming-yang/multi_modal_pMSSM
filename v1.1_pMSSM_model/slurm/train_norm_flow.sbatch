#!/bin/bash
# =============================================================================
# Conditional normalizing flow training on pMSSM data.
#
# Hyperparameters are read from slurm/train_norm_flow.yaml.
# Logs land under logs/<job_id>/ inside the repository.
#
# Submit:
#   sbatch slurm/train_norm_flow.sbatch
# For a test run:
#   sbatch slurm/train_norm_flow.sbatch --test_run
# To attach a note (saved into the run's hyperparams json):
#   sbatch slurm/train_norm_flow.sbatch --comment "some note"
# =============================================================================
#SBATCH --job-name=norm_flow_0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# ---- config -----------------------------------------------------------------
PROJ_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONFIG="${PROJ_DIR}/slurm/train_norm_flow.yaml"
SCRIPT="${PROJ_DIR}/normalizing_flow/v1.1_norm_flow.py"
LOG_DIR="${PROJ_DIR}/logs"

mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/train_norm_${SLURM_JOB_ID}.out" 2>"${LOG_DIR}/train_norm_${SLURM_JOB_ID}.err"

echo "=========================================="
echo "[norm-flow] job_id=${SLURM_JOB_ID}"
echo "[norm-flow] PROJ_DIR=${PROJ_DIR}"
echo "[norm-flow] CONFIG=${CONFIG}"
echo "[norm-flow] started: $(date)"
echo "=========================================="

START_TIME=$(date +%s)

# ---- run training -----------------------------------------------------------

cd "${PROJ_DIR}"
pixi run python -u "${SCRIPT}" --config "${CONFIG}" "$@"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo "=========================================="
echo "[norm-flow] finished: $(date)"
echo "[norm-flow] elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="
