#!/bin/bash
#SBATCH -A SNIC2021-3-7
#SBATCH -t 5-00:00:00
#SBATCH --gpus-per-node=A100:4

#TRAIN ON FRACTIONS
# ARGUMENTS: $1 = augs (moco or flips or none), $2 = save-dir

module load Python/3.8.6-GCCcore-10.2.0
module load TensorFlow/2.5.0-fosscuda-2020b
module load PyTorch/1.8.1-fosscuda-2020b
module load torchvision/0.9.1-fosscuda-2020b-PyTorch-1.8.1

cd /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/geography-aware-ssl/moco_fmow/moco
echo SCRIPT STARTED

python main_moco_tp.py  --lr 0.03 --dist-url 'tcp://127.0.0.1:8893' --world-size 1 --rank 0 --mlp -j 4 --cos --multiprocessing-distributed --pretrained --augs $1 --save-dir $2