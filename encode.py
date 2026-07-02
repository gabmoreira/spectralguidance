#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import logging
from pathlib import Path
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
from tqdm.auto import tqdm
from typing import Tuple

from spectral import whitening
from model import TimeConditionedEncoder
from data import get_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

@torch.no_grad()
def encode_dataset(
    phi_encoder: nn.Module, 
    noise_scheduler: DDIMScheduler,
    loader: DataLoader,
    t: int,
    ridge: float,
    device: str = "cuda",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    device = torch.device(device)

    phi_encoder.to(device)
    phi_encoder.eval()

    phi = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)

        t_batch = torch.full((len(x),), t, device=device)
        noise = torch.randn_like(x)
        x_t = noise_scheduler.add_noise(x, noise, t_batch)
        phi.append(phi_encoder(x_t, t_batch))
    phi = torch.cat(phi)  

    mu, w = whitening(phi, ridge=ridge) # (1, K), (K, K)
    phi = (phi - mu) @ w # (N, K)
    norm = phi.pow(2).mean(0, keepdim=True).sqrt() # (1, K)
    phi /= norm
    
    ones = torch.ones(len(phi), 1, device=x.device)
    phi = torch.cat((phi, ones), dim=-1) # (N, K + 1)
    return phi, mu, w, norm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script configuration")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--experiment_dir", type=Path, required=True, help="Directory for experiment outputs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training/testing")
    parser.add_argument("--tmin", type=int, required=True, help="Inclusive")
    parser.add_argument("--tmax", type=int, required=True, help="Exclusive")
    parser.add_argument("--step", type=int, required=True, help="Step size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--persistent_workers", action="store_true", help="Persistent workers")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory flag for DataLoader")
    return parser.parse_args()

def main() -> None:
    """
    python encode.py --device cuda --experiment_dir ./runs/celeba-hq-mask/phi_1000-1_k512_chunks16/ --batch_size 256 --num_workers 8 --pin_memory --tmin 10 --tmax 1000 --step 10
    python encode.py --device cuda --experiment_dir ./runs/cifar10/phi_1000-1_k512_chunks2/ --batch_size 1024 --num_workers 8 --pin_memory --tmin 10 --tmax 1000 --step 10
    """
    args = parse_args()
    cfg = OmegaConf.load(os.path.join(args.experiment_dir, ".hydra/config.yaml"))

    assert args.tmin >= cfg.scheduler.start
    assert args.tmax <= cfg.scheduler.num_train_timesteps

    dataset = get_dataset(
        dataset=cfg.dataset.name,
        split="train",
        augment=False,
        cache_dir=cfg.dataset.cache_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        collate_fn = getattr(dataset, "collate_fn", None),
    )

    state_dict = torch.load(os.path.join(args.experiment_dir, "state_dict.pt"))
    noise_scheduler = DDIMScheduler.from_pretrained(cfg.scheduler.pretrained)
    
    phi_encoder = TimeConditionedEncoder(
        image_size=cfg.dataset.image_size,
        out_dim=cfg.model.num_eigenfunctions,
        time_emb_dim=cfg.model.time_emb_dim,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        min_resolution=cfg.model.min_resolution,
        max_channels=cfg.model.max_channels,
        num_res_blocks=cfg.model.num_res_blocks,
        append_last=cfg.model.append_last,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
        num_train_timesteps=cfg.scheduler.num_train_timesteps,
    )
    phi_encoder.load_state_dict(state_dict)
    phi_encoder.to(args.device)
    phi_encoder.eval()

    mu = {}
    w = {}
    norm = {}
    phi = {}

    ts = range(args.tmin, args.tmax, args.step) # e.g. (10,1000,10): 10, 20, 30, ..., 990
    for t in tqdm(ts):
        phi_t, mu_t, w_t, norm_t = encode_dataset(
            phi_encoder=phi_encoder,
            noise_scheduler=noise_scheduler,
            loader=loader,
            t=t,
            ridge=cfg.training.ridge,
            device=args.device,
        )
        phi[t] = phi_t.cpu()
        mu[t] = mu_t.cpu()
        w[t] = w_t.cpu()
        norm[t] = norm_t.cpu()
    
    # Update model with whitening buffers
    phi_encoder.set_whitening(w=w, mu=mu, norm=norm)

    state_dict_save_path = os.path.join(args.experiment_dir, "state_dict_whitened.pt")
    torch.save(phi_encoder.state_dict(), state_dict_save_path)
    logger.info(f"Updated state dict with buffers mu, w and norm saved to {state_dict_save_path}")

    phi_save_path = os.path.join(args.experiment_dir, "phi.pt")
    torch.save(phi, phi_save_path)
    logger.info(f"phi saved to {phi_save_path}")

if __name__ == "__main__" :
    main()