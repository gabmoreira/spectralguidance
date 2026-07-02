#!/bin/bash

mkdir -p logs

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi -i $CUDA_VISIBLE_DEVICES

torchrun --nproc_per_node=4 ./train.py \
    dataset=imagenet64 \
    training.name="eps1e-6" \
    training.batch_size=4096 \
    training.num_chunks=3 \
    training.num_workers=4 \
    model.num_eigenfunctions=2000 \
    model.time_emb_dim=512 \
    model.base_channels=128 \
    model.channel_mults=[1,2,4,8,16] \
    model.max_channels=2048 \
    model.append_last=True \
    scheduler.beta_schedule="squaredcos_cap_v2" \
    scheduler.num_train_timesteps=4000 \
    scheduler.step=10