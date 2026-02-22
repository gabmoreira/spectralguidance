#!/bin/bash
#SBATCH --job-name=train_eigen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=48:00:00

mkdir -p logs
source /home/gmoreira/.env312/bin/activate

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi -i $CUDA_VISIBLE_DEVICES

echo "Using Python:"
python --version

python ./train.py \
    dataset=cifar10 \
    training.batch_size=2048 \
    training.num_chunks=2 \
    model.num_eigenfunctions=512 \
    model.base_channels=256 \
    scheduler.step=10 \
    scheduler.pretrained=google/ddpm-cifar10-32