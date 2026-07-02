#!/usr/bin/env python3
import numpy as np
import contextlib
import torch
import logging
import torch.nn as nn
import torch.distributed as dist

from diffusers import DDIMScheduler
from torch import Tensor
from typing import Tuple, Optional
from torch.nn.utils import clip_grad_norm_
from spectral import whitening

logger = logging.getLogger(__name__)

def train_spectral_guidance(
    x_batch: Tensor,
    t_batch: Tensor,
    model: nn.Module,
    noise_scheduler: DDIMScheduler,
    num_chunks: int,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float],
    ridge: float,
    eps: float = 1e-6,
    autocast_enabled: bool = False,
) -> Tuple[float, Tensor, Tensor]:
    is_distributed = dist.is_initialized()
    device = x_batch.device
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    noise_a = torch.randn_like(x_batch)
    noise_b = torch.randn_like(x_batch)
    x_a = noise_scheduler.add_noise(x_batch, noise_a, t_batch)
    x_b = noise_scheduler.add_noise(x_batch, noise_b, t_batch)

    chunks_a = torch.chunk(x_a, num_chunks, dim=0)
    chunks_b = torch.chunk(x_b, num_chunks, dim=0)
    chunks_t = torch.chunk(t_batch, num_chunks, dim=0)

    # Phase 1 — phi_a forward without grad, build whitening (global via all-reduce inside whitening())
    phi_a_chunks = []
    for chunk_a, chunk_t in zip(chunks_a, chunks_t):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            phi_a_chunks.append(model(chunk_a, chunk_t).float())
    phi_a = torch.cat(phi_a_chunks, dim=0)

    if is_distributed:
        torch.cuda.synchronize()

    # Whitening is computed from all ranks
    mu, W, cov_eig = whitening(phi_a, ridge=ridge, return_eig=True)
    K = phi_a.shape[1]
    n_chunks_total = len(phi_a_chunks)

    # Phase 2 — per-chunk phi_b forward + per-chunk normalized-correlation loss + backward.
    # std_a is detached (phi_a is detached anyway). std_b carries grad, computed per chunk.
    total_loss = 0.0
    eig_accum = torch.zeros(K, device=device)

    for i, (chunk_phi_a, chunk_b, chunk_t) in enumerate(zip(phi_a_chunks, chunks_b, chunks_t)):
        is_last = (i == n_chunks_total - 1)
        ctx = contextlib.nullcontext() if (is_last or not is_distributed) else model.no_sync()

        with ctx:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                chunk_phi_b = model(chunk_b, chunk_t).float()  # (chunk, K), grad here

            a_w = (chunk_phi_a - mu) @ W   # (chunk, K), no grad
            b_w = (chunk_phi_b - mu) @ W   # (chunk, K), grad

            # Per-chunk per-column std. a_w.std is detached (no grad from it). b_w.std carries grad.
            s_a = a_w.std(dim=0, keepdim=True) + eps   # (1, K), no grad
            s_b = b_w.std(dim=0, keepdim=True) + eps   # (1, K), grad   

            a_wn = a_w / s_a
            b_wn = b_w / s_b

            # Normalized per-coordinate cross-correlation, mean over K gives the loss.
            eigvals_chunk = (a_wn * b_wn).mean(dim=0)                  # (K,)
            loss = -eigvals_chunk.mean() / n_chunks_total
            loss.backward()
            total_loss += loss.item()

            with torch.no_grad():
                eig_accum += eigvals_chunk.detach()

    if grad_clip is not None:
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    optimizer.step()
    optimizer.zero_grad()

    # Average per-chunk normalized correlations across chunks (and ranks) for logging.
    if is_distributed:
        dist.all_reduce(eig_accum, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1

    eigenvalues = eig_accum / (n_chunks_total * world_size)

    if is_distributed:
        loss_tensor = torch.tensor([total_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        total_loss = loss_tensor.item()

    return total_loss, eigenvalues, cov_eig


@torch.no_grad()
def _extract_features(
    model,
    loader,
    noise_scheduler,
    t: int,
    device: str,
    is_distributed: bool,
    num_chunks: int=16,
):
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model

    feats, labs = [], []
    for batch in loader:
        assert isinstance(batch, (list, tuple)) and len(batch) >= 2, \
            "Dataset must return (image, label) for linear probe eval"
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]

        B = x.size(0)
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_noisy = noise_scheduler.add_noise(x, noise, t_batch)
        del noise

        chunk_phis = []
        for xc, tc in zip(x_noisy.chunk(num_chunks), t_batch.chunk(num_chunks)):
            chunk_phis.append(raw_model(xc, tc).float().cpu())
        feats.append(torch.cat(chunk_phis, dim=0))
        labs.append(y.cpu())
        del x, x_noisy, t_batch

    features = torch.cat(feats, dim=0)
    labels = torch.cat(labs, dim=0)

    if is_distributed:
        ws = dist.get_world_size()
        f_dev, l_dev = features.to(device), labels.to(device)
        f_list = [torch.zeros_like(f_dev) for _ in range(ws)]
        l_list = [torch.zeros_like(l_dev) for _ in range(ws)]
        dist.all_gather(f_list, f_dev)
        dist.all_gather(l_list, l_dev)
        features = torch.cat(f_list, dim=0).cpu()
        labels = torch.cat(l_list, dim=0).cpu()

    return features.numpy(), labels.numpy()


def coefficient_eval(
    model,
    loader,
    noise_scheduler,
    t: int,
    label_indices,
    device: str,
    ridge: float,
    is_distributed: bool=False,
    local_rank: int=0,
):
    """
    At timestep t, extract features, whiten them so they form an orthonormal
    basis in empirical L²(p_t), then for each attribute h compute:
      - energy_ratio = ||c||² / Var(h)   ∈ [0, 1], recoverable fraction

    Returns {attr_idx: {...}} on rank 0, None elsewhere. Collective — call on
    all ranks when distributed.
    """
    X, Y = _extract_features(model, loader, noise_scheduler, t, device, is_distributed)

    if Y.ndim == 1:
        num_classes = int(Y.max()) + 1
        Y = np.eye(num_classes, dtype=np.float64)[Y.astype(int)]

    if is_distributed and local_rank != 0:
        return None

    X = X.astype(np.float64)
    N, K = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)

    # Whitening transform from the same covariance used in training
    Sigma = (Xc.T @ Xc) / (N - 1)
    eigvals, eigvecs = np.linalg.eigh(Sigma + ridge * np.eye(K))
    W = eigvecs * (1.0 / np.sqrt(np.clip(eigvals, 1e-12, None)))  # (K, K)
    Xw = Xc @ W  # (N, K), empirical cov ≈ I
    Xw = Xw / np.std(Xw, axis=0, keepdims=True)

    results = {}
    for label_idx in label_indices:
        h = Y[:, label_idx].astype(np.float64)
        h_c = h - h.mean()
        var_h = float((h_c ** 2).mean())
        if var_h < 1e-12:
            logger.warning(f"Label {label_idx}: constant label, skipping")
            continue

        c = (Xw.T @ h_c) / N    # (K,) MC estimate of c_{t,k}
        c_sq = c ** 2
        energy = float(c_sq.sum())

        results[int(label_idx)] = {
            "energy_ratio": energy / var_h,
        }

    return results