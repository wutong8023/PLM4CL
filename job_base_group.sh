#!/bin/bash

#SBATCH -t 100800
#SBATCH -N 1
#SBATCH --gres=gpu:1

# Memory usage (MB)
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=128000

#SBATCH --mail-user=wutong8023@163.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# IMPORTANT!!! check the job name!
# replace line 21 with job_name!
#
#
#
#
#
#
#
#
#
module load python3
source /home/tongwu/envs/pseudoCL/bin/activate
module load cuda-11.2.0-gcc-10.2.0-gsjevs3



