#!/bin/bash
#SBATCH -p publicgpu
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -n 2
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rdschwa2@asu.edu
#SBATCH --export=NONE

##SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
##SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

module load anaconda/py3

source activate mypytorch

python mnistClassifier.py 

0.01

1

"""python cifarClassifier.py

4

1

0.001

0.9"""