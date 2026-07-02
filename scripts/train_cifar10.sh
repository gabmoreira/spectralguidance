#!/bin/bash

mkdir -p logs

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi -i $CUDA_VISIBLE_DEVICES

echo "Using Python:"
python --version

python ./train.py \
    dataset=cifar10 \
    training.name="eps1e-6" \
    training.batch_size=2048 \
    training.num_chunks=2 \
    training.ridge=1e-3 \
    training.lr=1e-4 \
    eval.t=100 \
    eval.label_indices=[0,1,2,3,4,5,6,7,8,9] \
    model.num_eigenfunctions=512 \
    model.base_channels=256 \
    scheduler.pretrained=google/ddpm-cifar10-32