#!/usr/bin/env python3
import os
import hydra
import torch
import numpy as np
import logging

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from omegaconf import DictConfig, OmegaConf
from diffusers import DDIMScheduler
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from train_utils import train_spectral_guidance, coefficient_eval
from model import TimeConditionedEncoder
from data import get_dataset
from utils import save_eigenvalue_plot

logger = logging.getLogger("train")

def log_vram(logger, device, local_rank) -> None:
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved  = torch.cuda.memory_reserved(device)  / 1024**3
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    logger.info(f"[rank {local_rank}] VRAM — allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB | peak: {max_alloc:.2f} GB")

def log_cov_eig(cov_eig: Tensor, epoch: int, writer) -> None:
    writer.add_scalar(f"train/cov_eig_min", cov_eig.detach().cpu().min().item(), epoch)
    writer.add_scalar(f"train/cov_eig_max", cov_eig.detach().cpu().max().item(), epoch)
    writer.add_scalar(f"train/cov_eig_med", cov_eig.detach().cpu().median().item(), epoch)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info("Starting training.")
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = "LOCAL_RANK" in os.environ

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        logger.info("Not distributed")

    device = torch.device(f"cuda:{local_rank}")

    writer = SummaryWriter(".") if local_rank == 0 else None

    phi_encoder = TimeConditionedEncoder(
        image_size=cfg.dataset.image_size,
        out_dim=cfg.model.num_eigenfunctions,
        time_emb_dim=cfg.model.time_emb_dim,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        min_resolution=cfg.model.min_resolution,
        max_channels=cfg.model.max_channels,
        num_train_timesteps=cfg.scheduler.num_train_timesteps,
        num_res_blocks=cfg.model.num_res_blocks,
        dropout=cfg.model.dropout,
        append_last=cfg.model.append_last,
        pooling=cfg.model.pooling,
    )
    phi_encoder = phi_encoder.to(device)

    dataset = get_dataset(
        dataset=cfg.dataset.name,
        split="train",
        augment=True,
        cache_dir=cfg.dataset.cache_dir,
    )
    logger.info(f"Dataset {cfg.dataset.name} (length: {len(dataset)}) ready")

    sampler = None
    shuffle = True
    if is_distributed:
        shuffle = None
        sampler = DistributedSampler(dataset, shuffle=True)
        phi_encoder = DDP(phi_encoder, device_ids=[local_rank])

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        pin_memory=cfg.training.pin_memory,
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.persistent_workers,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=getattr(dataset, "collate_fn", None),
        drop_last=True,
    )

    logger.info("Model architecture:\n%s", phi_encoder)

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
        checkpoint = torch.load(cfg.training.checkpoint_path, map_location="cpu", weights_only=False)

        raw_model = phi_encoder.module if is_distributed else phi_encoder
        raw_model.load_state_dict(checkpoint["phi_encoder"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        
        eigenvalues_t = checkpoint.get("eigenvalues_t", {})
        eigenvalues_t = {k : v.cpu() for k,v in eigenvalues_t.items()}
        start_epoch = checkpoint.get("epoch", 0) + 1

    for epoch in range(start_epoch, cfg.training.epochs):
        torch.cuda.reset_peak_memory_stats(device)

        if is_distributed:
            sampler.set_epoch(epoch)
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=local_rank != 0)

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            batch_size = x.size(0)

            t = np.random.choice(range(
                cfg.scheduler.start,
                cfg.scheduler.num_train_timesteps,
                cfg.scheduler.step
            ))

            if is_distributed:
                t_tensor = torch.tensor(t, device=device)
                dist.broadcast(t_tensor, src=0)
                t = t_tensor.item()

            t_batch = torch.full((batch_size,), t, device=device)
            
            loss, eigenvalues, cov_eig = train_spectral_guidance(
                x,
                t_batch,
                model=phi_encoder,
                noise_scheduler=noise_scheduler,
                num_chunks=cfg.training.num_chunks,
                optimizer=optimizer,
                grad_clip=cfg.training.grad_clip,
                ridge=cfg.training.ridge,
                autocast_enabled=False,
            )

            total_loss += loss
            eigenvalues_t[t] = eigenvalues.cpu()              
  
            if batch_idx % 25 == 0:
                log_vram(logger, device, local_rank)
                if local_rank == 0:
                    save_eigenvalue_plot(eigenvalues_t, "evals.png", epoch, writer)
                    log_cov_eig(cov_eig, epoch, writer)

            pbar.set_postfix({
                "loss" : total_loss / (batch_idx + 1),
                "t" : t,
                "lr" : scheduler.get_last_lr()[0],
            })
        
        # --- End of Epoch ---
        scheduler.step()

        metrics = None
        if len(cfg.eval.label_indices) > 0 and epoch % 5 == 0:
            metrics = coefficient_eval(
                model=phi_encoder,
                loader=train_loader,
                noise_scheduler=noise_scheduler,
                t=cfg.eval.t,
                label_indices=cfg.eval.label_indices,
                device=device,
                is_distributed=is_distributed,
                local_rank=local_rank,
                ridge=cfg.training.ridge,
            )

        raw_model = phi_encoder.module if is_distributed else phi_encoder
        
        if local_rank == 0:
            if metrics:
                for label_idx, m in metrics.items():
                    for key, v in m.items():
                        writer.add_scalar(f"{key}/label_{label_idx}", v, epoch)

            writer.add_scalar(f"train/lr", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar(f"train/loss", total_loss / (batch_idx + 1), epoch)
        
            save_eigenvalue_plot(eigenvalues_t, "evals.png", epoch, writer)
            torch.save(raw_model.state_dict(), "state_dict.pt")
            torch.save({
                "phi_encoder": raw_model.state_dict(),
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