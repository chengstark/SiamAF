#!/bin/bash
#SBATCH -t 4:00:00  # time requested in hour:minute:second
#SBATCH --mem=650G
#SBATCH --cpus-per-task=16
#SBATCH --partition=overflow

source /labs/hulab/stark_conda/bin/activate
conda activate base_pytorch

echo "JOB START"

nvidia-smi

python make_whole_data.py