#!/bin/bash

#SBATCH --job-name=cellpose
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hccassidy1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=job.out
#SBATCH --mem=32G

export py=/mnt/parscratch/users/ach22jc/anaconda/.envs/sa_env/bin/python
export SLURM_EXPORT_ENV=ALL

module load Anaconda3/2022.10
module load OpenBLAS/0.3.23-GCC-12.3.0
module load GCC/12.3.0
module load CUDA/12.4.0

source activate sa_env

nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used,memory.free --format=csv -l 2 > gpu_stats_${SLURM_JOB_ID}.log &
srun $py /users/ach22jc/cellpose-repo/script.py
