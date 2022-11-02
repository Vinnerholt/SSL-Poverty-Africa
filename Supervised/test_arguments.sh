#!/bin/bash
#SBATCH -A SNIC2021-3-7
#SBATCH -t 0-00:02:00
#SBATCH --gpus-per-node=T4:1

cd /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/Supervised
echo SCRIPT STARTED
echo $1
echo $2
