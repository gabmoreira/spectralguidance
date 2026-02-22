#!/usr/bin/env python3
import torch
import logging
import torch.nn as nn

from diffusers import DDIMScheduler
from torch import Tensor
from typing import Tuple, Optional
from torch.nn.utils import clip_grad_norm_

from spectral import compute_eigenvalues, whitening

logger = logging.getLogger(__name__)

def train_courant_fischer(
    x_batch: Tensor,
    t_batch: Tensor,
    model: nn.Module,
    noise_scheduler: DDIMScheduler,
    num_chunks: int,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float],
    eps: float,
    ridge: float,
    autocast_enabled: bool = False, # Don't autocast -- use fp32
) -> Tuple[Tensor, Tensor]:
    """
    Performs a training step using the Courant-Fischer eigenvalue loss.
    
    Args:
        x_batch: Input data tensor of shape (B, *).
        t_batch: Timestep tensor for the diffusion scheduler of shape (B,).
        model: Encoder being trained (B, *) -> (B, K).
        num_chunks: Number of chunks to split the batch for memory efficiency.
        autocast_enabled: If True, uses torch.cuda.amp for mixed precision.
    """
    device = x_batch.device
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    # Prepare noisy samples
    noise_a = torch.randn_like(x_batch)
    noise_b = torch.randn_like(x_batch)
    x_a = noise_scheduler.add_noise(x_batch, noise_a, t_batch)
    x_b = noise_scheduler.add_noise(x_batch, noise_b, t_batch)

    # Chunking for memory management
    chunks_a = torch.chunk(x_a, num_chunks, dim=0)
    chunks_b = torch.chunk(x_b, num_chunks, dim=0)
    chunks_t = torch.chunk(t_batch, num_chunks, dim=0)

    # Compute phi_a - no gradients
    chunks_phi_a = []
    for (chunk_a, chunk_t) in zip(chunks_a, chunks_t):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            chunk_phi_a = model(chunk_a, chunk_t)
        chunks_phi_a.append(chunk_phi_a)
        
    phi_a = torch.cat(chunks_phi_a).float()
    phi_a_mu, phi_a_whitener = whitening(phi_a, eps=eps, ridge=ridge)

    # Compute phi_b and backpropagate
    chunks_phi_b = []
    total_loss = 0.0
    for (chunk_phi_a, chunk_b, chunk_t) in zip(chunks_phi_a, chunks_b, chunks_t):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            chunk_phi_b = model(chunk_b, chunk_t)
            chunks_phi_b.append(chunk_phi_b.detach())

        eigenvalues = compute_eigenvalues(
            chunk_phi_a.float(), # no-grad
            chunk_phi_b.float(),
            mu=phi_a_mu, # no-grad
            whitener=phi_a_whitener, # no-grad
            normalize=True,
        )
        
        loss = (1.0 - eigenvalues.mean()) / num_chunks
        loss.backward()
        total_loss += loss.item()
        
    if grad_clip is not None:
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    optimizer.step()
    optimizer.zero_grad()       

    # Final eval metrics on the full batch for logging
    phi_b = torch.cat(chunks_phi_b)
    eigenvalues = compute_eigenvalues(
        phi_a.float().detach(),
        phi_b.float().detach(),
        mu=phi_a_mu,
        whitener=phi_a_whitener,
        normalize=True,
    )
    
    return total_loss, eigenvalues


def train_eckart_young(
    x_batch: Tensor,
    t_batch: Tensor,
    model: nn.Module,
    noise_scheduler: DDIMScheduler,
    optimizer,
    grad_clip: float,
) -> Tuple[Tensor, Tensor]:
    """
    Performs a single training step using the Eckart-Young-Mirksy Loss
    to learn the leading eigenspace of the diffusion's covariance operator.
    
    Objective:
        L = || T_t T_t^\ast - Phi @ Phi.T ||_HS
          = const - 2 * E[ <Phi, Phi> ] + || Phi ||_HS^2

    Args:
        x_batch (Tensor): Clean data batch (x0). Shape: (B, C, H, W) or (B, D).
        t_batch (Tensor): Timesteps for the noise scheduler. Shape: (B,).
        model (nn.Module): The backbone neural network. Output must be (B, K).
        noise_scheduler (DDIMScheduler): Scheduler for adding noise.
        optimizer (Optimizer): The torch optimizer.
        grad_clip (float, optional): Max norm for gradient clipping. Defaults to 1.0.

    Returns:
        loss (float): The scalar loss value for logging.
        eigenvalues (Tensor): The estimated eigenvalues of the operator
                              computed via post-hoc whitening. Shape: (K,).
    """
    batch_size = len(x_batch)

    optimizer.zero_grad()       

    noise_a = torch.randn_like(x_batch)
    noise_b = torch.randn_like(x_batch)

    xt_a = noise_scheduler.add_noise(x_batch, noise_a, t_batch)
    xt_b = noise_scheduler.add_noise(x_batch, noise_b, t_batch)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
        phi_a = model(xt_a, t_batch)
        phi_b = model(xt_b, t_batch)

        phi_a_f32 = phi_a.float()
        phi_b_f32 = phi_b.float()

        alignment = -2.0 * (phi_a_f32 * phi_b_f32).sum(dim=1).mean()

        cov_matrix_a = (phi_a_f32.T @ phi_a_f32) / batch_size
        cov_matrix_b = (phi_b_f32.T @ phi_b_f32) / batch_size
        
        cov_norm = 0.5 * (cov_matrix_a.pow(2).sum() + cov_matrix_b.pow(2).sum())

        loss = alignment + cov_norm

    loss.backward()

    if grad_clip is not None:
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    optimizer.step()
    
    # Monitoring (Post-hoc)
    with torch.no_grad():
        phi_a_mu, phi_a_whitener = whitening(phi_a_f32, eps=1e-6, ridge=1e-3)
        try:
            eigenvalues = compute_eigenvalues(
                phi_a_f32.detach(),
                phi_b_f32.detach(),
                mu=phi_a_mu,
                whitener=phi_a_whitener,
                normalize=True,
            )
        except:
            logger.warning("Could not compute eigenvalues.")
            eigenvalues = None
    
    return loss.item(), eigenvalues
