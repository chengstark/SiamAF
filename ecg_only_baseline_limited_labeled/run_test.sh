#!/bin/bash
#SBATCH -t 2:00:00  # time requested in hour:minute:second
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=overflow

source /labs/hulab/stark_conda/bin/activate
conda activate base_pytorch

echo "JOB START"

nvidia-smi

python test_pipeline.py