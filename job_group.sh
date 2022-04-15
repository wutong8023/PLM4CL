#!/bin/bash

#SBATCH -t 100800
#SBATCH -N 1
#SBATCH --gres=gpu:1

# Memory usage (MB)
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=128000

#SBATCH --mail-user=wutong8023@163.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# IMPORTANT!!! check the job name!
#SBATCH -o log/%J-main_all.out
#SBATCH -e log/%J-main_all.err
#
#
#
#
#SBATCH -J main_all
#
#
#
module load python3
source /home/tongwu/envs/pseudoCL/bin/activate
module load cuda-11.2.0-gcc-10.2.0-gsjevs3



python3 -m analyze.time_ft_bt_acc_analysis --info main_all  --vis_type all  --vis_by ptm --setting class  --pltf gp

