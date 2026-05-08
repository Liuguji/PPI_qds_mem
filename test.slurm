#!/bin/bash

#SBATCH -J simple-add-test        # Job name
#SBATCH -p xinda                  # Queue/partition
#SBATCH -o %j.out                 # Standard output file
#SBATCH -e %j.err                 # Standard error file
#SBATCH -N 1                      # Number of nodes
#SBATCH --ntasks-per-node=1       # Tasks per node
#SBATCH --cpus-per-task=1         # CPU cores per task

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

python3 - <<'PY'
result = 1 + 1
print(f"1 + 1 = {result}")
PY
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "Success: Python script executed successfully!"
    echo "Job completed successfully at: $(date)"
else
    echo "Error: Python script failed with exit code $PYTHON_EXIT_CODE"
    echo "Job failed at: $(date)"
    exit 1
fi
