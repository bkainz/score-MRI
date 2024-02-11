#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --output=scoreMRI_%j.log
#SBATCH --export=ALL
#SBATCH --time=10:00:00  

cd $WORK/score-MRI

bash
source ~/.bashrc
source activate score-POCS 

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun python inference_multi-coil_SSOS.py --data '001' --N 3000 --acc_factor 32 --center_fraction 0.03

#conda install nvidia/label/cuda-12.1.0::cuda-toolkit
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#pip install -r requirements.txt

#interactive
#salloc --gres=gpu:a40:1 --partition=a40 --time=01:00:00

#nvidia-smi
#srun --jobid=1083633 --overlap --pty /bin/bash

#sbatch
#sbatch slurmjon_alex.sh

#squeue for job id
