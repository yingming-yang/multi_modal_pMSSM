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
# Choose variables by modality (data_types) or step, passed to select():
#   sbatch .../tests/slurm/train_fm.sbatch --data_types theory_parameter dark_matter
# Choose the training masks (default by_feature, see train_fm.yaml):
#   sbatch .../tests/slurm/train_fm.sbatch --mask_type by_type
# Use a different YAML file (the last --config wins):
#   sbatch .../tests/slurm/train_fm.sbatch --config /path/to/other.yaml
# Name the run (folder fm_<jobid>_<run_name>/), here or as run_name in the YAML:
#   sbatch .../tests/slurm/train_fm.sbatch --run_name by_feature
# Longer time limit (an sbatch option, so it goes before the script; 100k steps took ~1.2 h with 20
# variables, more variables take longer):
#   sbatch --time=04:00:00 .../tests/slurm/train_fm.sbatch
#
# Monitor:
#   squeue --me --start
#   tail -f /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/checkpoints/fm_<jobid>[_<run_name>]/fm_<jobid>[_<run_name>].out
#
# Results in tests/checkpoints/fm_<jobid>[_<run_name>]/ (written as <name> below):
#   <name>.out                      log. Slurm opens it as tests/checkpoints/logs/fm_<jobid>.out (the run
#                                   folder does not exist yet at submission), and train_fm.py moves it here
#                                   when it starts, about a minute in. If a job fails earlier, it stays in logs/.
#   <name>_weights.pt              dict: model_state, best_model_state, hparams, mean, std, train_files, val_files, test_files,
#                                  checkpoints ({step: model_state} at n_checkpoints log-spaced steps, e.g. 100, 1k, 10k, 100k), ...
#   <name>_loss_history.npz        step, train_loss, val_loss, lr; per-mask validation losses (at the
#                                  checkpoint steps): mask_step, mask_loss, mask_names
#   <name>_loss.png                loss curve
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
CONFIG="${PROJ_DIR}/tests/slurm/train_fm.yaml"

# train_fm.py moves this log into its run folder when it starts (same path as #SBATCH --output).
# Slurm keeps writing to the moved file (same filesystem), including its own messages such as a time-limit cancel.
export FM_SLURM_LOG="${PROJ_DIR}/tests/checkpoints/logs/fm_${SLURM_JOB_ID}.out"

echo "[fm] job_id=${SLURM_JOB_ID} node=$(hostname) started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
START=$(date +%s)

cd "${PROJ_DIR}"

echo "[fm] training"
pixi run python -u "${PROJ_DIR}/tests/scripts/train_fm.py" --config "${CONFIG}" "$@"

ELAPSED=$(( $(date +%s) - START ))
echo "[fm] finished: $(date)  elapsed: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"
