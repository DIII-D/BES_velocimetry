#!/bin/bash

#SBATCH -A trn007
#SBATCH -q debug
#SBATCH -t 10:00
#SBATCH -C gpu

export INPUT_FILE_PATH=/global/cfs/cdirs/trn007/DIII-D/imgs.itg.600.602.bes.s200729.h5

module load conda
mamba activate $SCRATCH/bes-velocimetry

mkdir -p $SCRATCH/hackathon/outputs

cd $SCRATCH/hackathon

nsys profile --trace cuda,osrt,nvtx --python-sampling=true bes-velocimetry --fn ${INPUT_FILE_PATH} --cores 128 --nsteps 5
