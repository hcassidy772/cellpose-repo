#!/bin/bash

#SBATCH --job-name=sd_opt_s
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hccassidy1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=sd_opt_job.out
#SBATCH --mem=32G

export py=/mnt/parscratch/users/ach22jc/anaconda/.envs/sdist_env/bin/python
export SLURM_EXPORT_ENV=ALL
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/apps/testapps/el7-znver3/software/staging/CUDA/11.8.0/"

module load OpenBLAS/0.3.23-GCC-12.3.0
module load GCC/12.3.0 
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

# nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used,memory.free --format=csv -l 2 > gpu_stats_${SLURM_JOB_ID}.log &

srun $py /users/ach22jc/cellpose-repo/opt.py
