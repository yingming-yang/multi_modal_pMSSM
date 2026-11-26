#!/bin/bash -l
# Smoke-test submission for v1.1.2_bidir_norm_flow.py on the small, synthetic
# example dataset (normalizing_flow/example_data/). Adjust partition/time/mem
# once this is confirmed working, before pointing --config at a real dataset.
#SBATCH -J bidir_flow_smoke
#SBATCH -p small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000
#SBATCH --time=00:15:00
#SBATCH -o /u/yinya/workspace/multi_modal_pMSSM/v1.1_pMSSM_model/normalizing_flow/slurm/logs/%x.%j.out
#SBATCH -e /u/yinya/workspace/multi_modal_pMSSM/v1.1_pMSSM_model/normalizing_flow/slurm/logs/%x.%j.err

set -euo pipefail

PROJECT_DIR=/u/yinya/workspace/multi_modal_pMSSM/v1.1_pMSSM_model
PYTHON=${PROJECT_DIR}/.pixi/envs/default/bin/python

cd "${PROJECT_DIR}"

# CPU-only run: hide any GPUs so device selection in the script falls back to
# CPU cleanly (the script's torch.set_num_threads already respects
# SLURM_CPUS_PER_TASK, set automatically from --cpus-per-task above).
export CUDA_VISIBLE_DEVICES=""

srun "${PYTHON}" normalizing_flow/v1.1.2_bidir_norm_flow.py \
    --config normalizing_flow/slurm/example_config.yaml \
    --comment "slurm smoke test on synthetic example_data"
