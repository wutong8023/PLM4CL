#!/bin/bash

#
#
#
#
#
#
#

#SBATCH --account=da33
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Memory usage (MB)
#SBATCH --mem-per-cpu=64000

#SBATCH --mail-user=wutong8023@163.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load anaconda/5.0.1-Python3.6-gcc5

source activate /home/twu/da33/tong/envs/pseudoCL/
module load cuda/11.0

