#!/bin/bash
#SBATCH --job-name=train_eigen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00

mkdir -p logs
source /home/gmoreira/.env312/bin/activate

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi -i $CUDA_VISIBLE_DEVICES

python ./train.py \
    dataset=celeba-hq-mask \
    training.batch_size=2048 \
    training.num_chunks=16 \
    model.num_eigenfunctions=512 \
    scheduler.step=10 \
    scheduler.pretrained=google/ddpm-ema-celebahq-256