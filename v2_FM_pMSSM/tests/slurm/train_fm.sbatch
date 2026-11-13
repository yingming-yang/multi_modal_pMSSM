#!/bin/bash
# =============================================================================
# Train with tests/scripts/train_fm.py on one A100.
# Hyperparameters come from tests/slurm/train_fm.yaml (passed as --config).
# Evaluate afterwards with tests/eval_fm.ipynb (and tests/slurm/sample_fm.sbatch).
#
# Submit (from any directory):
#   sbatch /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/slurm/train_fm.sbatch
# Extra arguments go to the training script and override the YAML, e.g.:
#   sbatch .../tests/slurm/train_fm.sbatch --max_files 20 --max_iters 20000
# Choose variables by modality (data_type) or step, passed to select():
#   sbatch .../tests/slurm/train_fm.sbatch --data_type theory_parameter dark_matter
# Use a different YAML file (the last --config wins):
#   sbatch .../tests/slurm/train_fm.sbatch --config /path/to/other.yaml
#
# Monitor:
#   squeue --me --start
#   tail -f /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/checkpoints/logs/fm_<jobid>.out
#
# Results in tests/checkpoints/fm_<jobid>/ (log in tests/checkpoints/logs/fm_<jobid>.out):
#   fm_<jobid>_weights.pt           dict: model_state, best_model_state, hparams, mean, std, file_names, ...
#   fm_<jobid>_loss_history.npz     step, train_loss, val_loss, lr
#   fm_<jobid>_loss.png             loss curve
# =============================================================================
#SBATCH --job-name=fm_pmssm
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/checkpoints/logs/fm_%j.out
#SBATCH --error=/u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/checkpoints/logs/fm_%j.out
# A job with 1-3 GPUs always ends up in the shared gpu1 partition, so no --partition is set.

set -euo pipefail

PROJ_DIR=/u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM
OUT_DIR="${PROJ_DIR}/tests/checkpoints"
RUN_NAME="fm_${SLURM_JOB_ID}"
CONFIG="${PROJ_DIR}/tests/slurm/train_fm.yaml"

echo "[fm] job_id=${SLURM_JOB_ID} node=$(hostname) started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
START=$(date +%s)

cd "${PROJ_DIR}"

echo "[fm] training"
pixi run python -u "${PROJ_DIR}/tests/scripts/train_fm.py" --config "${CONFIG}" --out_dir "${OUT_DIR}" --run_name "${RUN_NAME}" "$@"

ELAPSED=$(( $(date +%s) - START ))
echo "[fm] finished: $(date)  elapsed: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"
