#!/bin/bash

#SBATCH -J fudan-qds-test
#SBATCH -p fat
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

module load anaconda3
conda activate qds_venv

## 如果slurm不在项目目录需要cd到工作目录

python3 test/scripts/test_ckpt_cpu.py --lmax 100 200 300
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
else
    echo "Error: Python script failed with exit code $PYTHON_EXIT_CODE"
    exit 1
fi