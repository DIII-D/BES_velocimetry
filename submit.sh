#!/bin/bash

#SBATCH -A trn007
#SBATCH -q debug
#SBATCH -t 20:00
#SBATCH -C cpu

export INPUT_FILE_PATH=/global/cfs/cdirs/trn007/DIII-D/imgs.itg.600.602.bes.s200729.h5

module load conda
mamba activate $SCRATCH/bes-velocimetry

mkdir -p $SCRATCH/hackathon/outputs

cd $SCRATCH/hackathon

bes-velocimetry --fn ${INPUT_FILE_PATH} --cores 256 --nsteps 10
