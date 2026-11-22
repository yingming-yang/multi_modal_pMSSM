#!/bin/bash
# =============================================================================
# Run all sampling and test-loss jobs of tests/checkpoints/compare/<compare_name>/sampling_inputs.pt
# with tests/scripts/sample_compare.py on one A100 (written by tests/compare_fm.ipynb).
#
# 1. In tests/compare_fm.ipynb set sample_mode = "save" and run it (login node is fine).
# 2. Submit (from any directory), with the compare_name as the first argument:
#      sbatch /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/slurm/sample_compare.sbatch benchmark_shared_per_feature
#    gpudev stops the job after 15 minutes. Finished jobs are saved, so if the log
#    does not end with "all jobs done", submit again and it continues.
#    For a longer time limit, override the partition on the command line
#    (the queue in gpu1 can be long):
#      sbatch --partition=gpu1 --time=01:00:00 .../sample_compare.sbatch <compare_name>
# 3. In tests/compare_fm.ipynb set sample_mode = "load" and run it.
#
# Monitor:
#   squeue --me --start
#   tail -f /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/checkpoints/logs/sample_compare_<jobid>.out
#
# Results:
#   tests/checkpoints/compare/<compare_name>/sampling_outputs.pt  dict: settings, results {run: {job: samples}},
#                                                                  losses {run: {mask: per-point test loss}}
#   tests/checkpoints/logs/sample_compare_<jobid>.out             log
# =============================================================================
#SBATCH --job-name=sample_compare
#SBATCH --partition=gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --output=/u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/checkpoints/logs/sample_compare_%j.out
#SBATCH --error=/u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/checkpoints/logs/sample_compare_%j.out

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch sample_compare.sbatch <compare_name> [extra arguments for sample_compare.py]"
    exit 1
fi
COMPARE_NAME="$1"
shift

PROJ_DIR=/u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM

echo "[sample_compare] job_id=${SLURM_JOB_ID} node=$(hostname) compare=${COMPARE_NAME} started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
START=$(date +%s)

cd "${PROJ_DIR}"   # sample_compare.py uses paths relative to the project directory
pixi run python -u tests/scripts/sample_compare.py --compare "${COMPARE_NAME}" "$@"

ELAPSED=$(( $(date +%s) - START ))
echo "[sample_compare] finished: $(date)  elapsed: $((ELAPSED/60))m $((ELAPSED%60))s"
