#!/bin/bash
#SBATCH -t 240:00:00  # time requested in hour:minute:second
#SBATCH --mem=220G
#SBATCH --gres=gpu:volta:2
#SBATCH --partition=overflow
#SBATCH --output=/home/zguo30/ppg_ecg_proj/ecg_only_baseline/slurm_outputs/%j.out

source /labs/hulab/stark_conda/bin/activate
conda activate base_pytorch

echo "JOB START"

nvidia-smi

python main.py