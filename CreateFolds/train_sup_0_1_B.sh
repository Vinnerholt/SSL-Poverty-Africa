#!/bin/bash
#SBATCH -A SNIC2021-3-7
#SBATCH -t 0-12:00:00
#SBATCH --gpu-per-node=A100:4

cd /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/Supervised
echo SCRIPT STARTED

python supervised_training.py  --lr 0.02 --dist-url 'tcp://127.0.0.1:8894' --world-size 1 --rank 0 -j 4 --multiprocessing-distributed  -b 128 --epochs 50 --fold B --save-dir fractions/0_1/fold_B --folds-path /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/CreateFolds/dhs_incountry_folds_0_1.pkl --drop-last-batch 0
