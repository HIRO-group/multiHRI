#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --output=run-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=naren@colorado.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=amc
#SBATCH --constraint=gpu

#module purge

#module load anaconda/2023.09
#module load cuda/12.1


#if ls -la | grep -q .venv; then echo ".venv already exists"; else conda create -y --prefix .venv python=3.7; fi
#conda activate $(realpath .venv)

#pip install -e ../overcooked_ai
#pip install -e .

export WANDB_API_KEY=$(cat wandb.token)
#export WANDB_CACHE_DIR=/scratch/alpine/nasi4978/.wandb
#export WANDB_DIR=/scratch/alpine/nasi4978/wandb
export WANDB_PROJECT=n_minus_1_sp
export WANDB_ENTITY=narendasan

python scripts/train_agents.py
