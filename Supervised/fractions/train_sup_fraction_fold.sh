#!/bin/bash
#SBATCH -A SNIC2021-3-7
#SBATCH -t 0-12:00:00
#SBATCH --gpus-per-node=A100:4

#TRAIN ON FRACTIONS
# ARGUMENTS: $1 = fraction, $2 = fold

module load Python/3.8.6-GCCcore-10.2.0
module load TensorFlow/2.5.0-fosscuda-2020b
module load PyTorch/1.8.1-fosscuda-2020b
module load torchvision/0.9.1-fosscuda-2020b-PyTorch-1.8.1

cd /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/Supervised
echo SCRIPT STARTED

python supervised_training.py  --lr 0.02 --dist-url 'tcp://127.0.0.1:8894' --world-size 1 --rank 0 -j 4 --multiprocessing-distributed  -b 128 --epochs 50 --fold $2 --save-dir fractions/$1/fold_$2 --folds-path /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/CreateFolds/dhs_incountry_folds_$1.pkl --drop-last-batch 0
