#!/bin/bash
  
#SBATCH --job-name=main   # job name
#SBATCH --output=o.out # output log file
#SBATCH --error=e.err  # error file
#SBATCH --time=01:00:00  # 1 hour of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATH --partition=debug # GPU partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

# Load all required modules below. As an example we load cuda/9.1
# module load cuda/9.1

# Add lines here to run your GPU-based computations.
srun a
