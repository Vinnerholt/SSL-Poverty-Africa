#!/bin/bash
#SBATCH -A SNIC2021-3-7
#SBATCH -t 0-05:00:00
#SBATCH --gpus-per-node=A100:1

# Args: $1 = checkpoint_dir, $2 = fold, $3 = fraction, $4 = plot_title, $5 = file_name
echo Args: $1 = checkpoint_dir, $2 = fold, $3 = fraction, $4 = plot_title, $5 = file_name

module load Python/3.8.6-GCCcore-10.2.0
module load TensorFlow/2.5.0-fosscuda-2020b
module load PyTorch/1.8.1-fosscuda-2020b
module load torchvision/0.9.1-fosscuda-2020b-PyTorch-1.8.1
module load matplotlib/3.3.3-fosscuda-2020b

cd /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/Supervised
echo SCRIPT STARTED

python make_r2_plot.py --checkpoint-dir $1 --fold $2 --fraction $3 --plot-title $4 --file-name $5
