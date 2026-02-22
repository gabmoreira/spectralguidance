#!/usr/bin/env python3
import os
import hydra
import torch
import numpy as np
import logging

from omegaconf import DictConfig, OmegaConf
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from train_utils import train_courant_fischer
from model import TimeConditionedEncoder
from data import get_dataset
from utils import save_eigenvalue_plot

logger = logging.getLogger("train")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info("Starting training.")
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    writer = SummaryWriter(".")

    dataset = get_dataset(
        dataset=cfg.dataset.name,
        split="train",
        augment=True,
        cache_dir=cfg.dataset.cache_dir,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        pin_memory=cfg.training.pin_memory,
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.persistent_workers,
        shuffle=True,
        collate_fn=getattr(dataset, "collate_fn", None),
    )
    
    phi_encoder = TimeConditionedEncoder(
        image_size=cfg.dataset.image_size,
        out_dim=cfg.model.num_eigenfunctions,
        time_emb_dim=cfg.model.time_emb_dim,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        min_resolution=cfg.model.min_resolution,
        max_channels=cfg.model.max_channels,
        num_train_timesteps=cfg.scheduler.num_train_timesteps,
    )
    phi_encoder = phi_encoder.to(cfg.device)

    if cfg.scheduler.pretrained is not None:
        noise_scheduler = DDIMScheduler.from_pretrained(cfg.scheduler.pretrained)
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.scheduler.num_train_timesteps,
            beta_start=cfg.scheduler.beta_start,
            beta_end=cfg.scheduler.beta_end,
            beta_schedule=cfg.scheduler.beta_schedule,
        )

    optimizer = AdamW(
        params=phi_encoder.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = ExponentialLR(optimizer, gamma=cfg.training.gamma)

    start_epoch = 0
    eigenvalues_t = {}

    if cfg.training.checkpoint_path is not None:
        if not os.path.exists(cfg.training.checkpoint_path):
            logger.error(f"Checkpoint {cfg.training.checkpoint_path} not found")
            return  
        logger.info(f"Loading checkpoint from {cfg.training.checkpoint_path}")
        checkpoint = torch.load(cfg.training.checkpoint_path, map_location=cfg.device, weights_only=False)
        phi_encoder.load_state_dict(checkpoint["phi_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        eigenvalues_t = checkpoint.get("eigenvalues_t", {})
        eigenvalues_t = {k : v.cpu() for k,v in eigenvalues_t.items()}
        start_epoch = checkpoint.get("epoch", 0) + 1

    for epoch in range(start_epoch, cfg.training.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", colour="blue")
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(cfg.device)
            else:
                x = batch.to(cfg.device)
            batch_size = x.size(0)

            t = np.random.choice(range(
                cfg.scheduler.start,
                cfg.scheduler.num_train_timesteps,
                cfg.scheduler.step
            ))
            t_batch = torch.full((batch_size,), t, device=cfg.device) 
            
            loss, eigenvalues = train_courant_fischer(
                x,
                t_batch,
                model=phi_encoder,
                noise_scheduler=noise_scheduler,
                num_chunks=cfg.training.num_chunks,
                optimizer=optimizer,
                grad_clip=cfg.training.grad_clip,
                eps=cfg.training.eps,
                ridge=cfg.training.ridge,
            )

            if eigenvalues is not None:
                eigenvalues_t[t] = eigenvalues.cpu()              
  
            if batch_idx % 50 == 0:
                save_eigenvalue_plot(eigenvalues_t, "evals.png", epoch, writer)

            pbar.set_postfix({
                "loss" : loss,
                "t" : t,
                "lr" : scheduler.get_last_lr()[0],
            })
            
        scheduler.step()
        writer.add_scalar(f"train/lr", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar(f"train/loss", loss, epoch)
        save_eigenvalue_plot(eigenvalues_t, "evals.png", epoch, writer)
        torch.save(phi_encoder.state_dict(), "state_dict.pt")
        torch.save({
            "phi_encoder": phi_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "eigenvalues_t" : eigenvalues_t,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, "checkpoint.pt")
        logger.info("Saved checkpoint: state_dict.pt")

    writer.close()

if __name__ == "__main__":
    main()