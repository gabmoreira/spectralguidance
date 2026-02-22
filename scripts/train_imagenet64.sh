#!/bin/bash
#SBATCH --job-name=train_eigen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00

mkdir -p logs
source /home/gmoreira/.env312/bin/activate

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi -i $CUDA_VISIBLE_DEVICES

python ./train.py \
    dataset=imagenet64 \
    training.batch_size=2048 \
    training.num_chunks=3 \
    model.num_eigenfunctions=700 \
    model.base_channels=256 \
    scheduler.beta_schedule="squaredcos_cap_v2" \
    scheduler.num_train_timesteps=4000 \
    scheduler.step=10