#!/bin/bash

#SBATCH -J fudan-qds-test       # 作业名
#SBATCH -p xinda                  # 选择的作业队列
#SBATCH -o %j.out               # 标准输出文件(%j会被替换为作业ID)
#SBATCH -e %j.err               # 标准错误文件
#SBATCH -N 1                    # 申请节点数
#SBATCH --ntasks-per-node=1     # 每节点运行的任务数量
#SBATCH --cpus-per-task=32      # 每个任务的核心数
#SBATCH --exclude=a02r3n12

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

python3 test/scripts/test_ckpt_cpu.py --lmax 100 200 300     # 实际执行命令
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "Successfully,Python script executed successfully!"
    echo "Job completed successfully at: $(date)"
else
    echo "Error: Failed,Python script failed with exit code $PYTHON_EXIT_CODE"
    echo "Job failed at: $(date)"
    exit 1
fi