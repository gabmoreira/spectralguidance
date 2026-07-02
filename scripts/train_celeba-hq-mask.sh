#!/bin/bash

mkdir -p logs

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi -i $CUDA_VISIBLE_DEVICES

torchrun --nproc_per_node=2 ./train.py \
    dataset=celeba-hq-mask \
    training.name="eps1e-8" \
    training.batch_size=2048 \
    training.num_chunks=16 \
    training.ridge=1e-3 \
    eval.t=100 \
    eval.label_indices=[4,5,8,9,15,17,20,38] \
    model.num_eigenfunctions=512 \
    scheduler.pretrained=google/ddpm-ema-celebahq-256