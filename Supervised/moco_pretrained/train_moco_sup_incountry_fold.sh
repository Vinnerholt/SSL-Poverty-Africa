#!/bin/bash
#SBATCH -A SNIC2021-3-7
#SBATCH -t 2-00:00:00
#SBATCH --gpus-per-node=A100:4

#TRAIN ON FRACTIONS
# ARGUMENTS: $1 = fold, $2 = save-dir, $3 = checkpoint-path, $X = MoCo epoch (specified with 4 digits, e.g. 0015 for 16 epochs of pretraining)

module load Python/3.8.6-GCCcore-10.2.0
module load TensorFlow/2.5.0-fosscuda-2020b
module load PyTorch/1.8.1-fosscuda-2020b
module load torchvision/0.9.1-fosscuda-2020b-PyTorch-1.8.1

cd /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/Supervised
echo SCRIPT STARTED
echo $1
echo $2
echo $3
echo $4
echo $5

python supervised_training.py  --lr 0.02 --dist-url 'tcp://127.0.0.1:8896' --world-size 1 --rank 0 -j 4 --multiprocessing-distributed -b 128 --epochs 30 --fold $1 --save-dir $2/fold_$1 --folds-path /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/CreateFolds/new_dhs_incountry_folds.pkl --resume $3
