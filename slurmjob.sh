#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --nodelist=gracehop1
#SBATCH --account=iwai
#SBATCH --output=scoreMRI_%j.log
#SBATCH --export=ALL
#SBATCH --time=08:00:00  

SCRATCH_DIRECTORY=$WORK/kelp/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
#cd ${SCRATCH_DIRECTORY}

cd $WORK/score-MRI

bash
source ~/.bashrc
source activate pytorch-build 

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun python inference_multi-coil_SSOS.py --data '001' --N 5000 --acc_factor 64 --center_fraction 0.01


#srun --nodes=1 --nodelist=genoa2 nvidia-smi

#srun --nodes=1 --nodelist=genoa2 nvcc --version
#gracehop1
#sbatch slumjob.sh
#srun -N 1 -w gracehop1 --time=01:00:00 --pty bash -i 
#export CUDA_HOME=
#export CUDA_HOME=$CONDA_PREFIX/bin
#conda install nvidia/label/cuda-12.1.0::cuda-toolkit
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#pip uninstall torch
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#conda install nvidia/label/cuda-11.6.0::cuda-toolkit
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#CUDA_VISIBLE_DEVICES=0 /home/saturn/iwai/iwai001h/miniconda3_aarch64/envs/score-POCS_gh/bin/python3 inference_multi-coil_hybrid.py --data '001' --N 2000

#CUDA_VISIBLE_DEVICES=0 python3 inference_multi-coil_hybrid.py --data '001' --N 2000


#srun -N 1 --gres=gpu:a40:1 --time=01:00:00 --pty bash -i  
#salloc --gres=gpu:a40:1 --partition=a40 --time=01:00:00

#python3 inference_multi-coil_SSOS.py --data '001' --N 2000


#conda install nvidia/label/cuda-12.1.0::cuda-toolkit
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#pip install -r requirements.txt