#!/bin/bash
#SBATCH -A SNIC2021-3-7
#SBATCH -t 2-00:00:00
#SBATCH --gpu-per-node=A100:4

cd /cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/Supervised
echo SCRIPT STARTED