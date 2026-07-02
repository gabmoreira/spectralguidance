import torch
import torch.distributed as dist
import logging
from torch import Tensor
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

def whitening(
    x: torch.Tensor,
    ridge: float,
    return_eig: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """
    Computes centering mean and whitening matrix.
    
    Single-GPU: SVD on the centered data matrix (better condition number).
    Distributed: all-reduce scatter matrix then eigh on the global covariance
                 (SVD requires the full data matrix, which doesn't exist globally).

    Args:
        x: (N, K) feature matrix (local shard in distributed mode)
        ridge: ridge regularization (acts like eigenvalue floor)

    Returns:
        mu: (1, K) mean
        cov_inv_sqrt: (K, K) whitening matrix
    """
    dtype = x.dtype

    if dist.is_initialized():
        n_local  = torch.tensor(len(x), device=x.device, dtype=dtype)
        sum_local = x.sum(0)                               # (K,)
        dist.all_reduce(n_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_local, op=dist.ReduceOp.SUM)
        n_global = n_local
        mu = (sum_local / n_global).unsqueeze(0)           # (1, K)

        x_c = x - mu
        S = x_c.T @ x_c                                    # (K, K) local scatter
        dist.all_reduce(S, op=dist.ReduceOp.SUM)           # global scatter
        C = S / (n_global - 1)                             # (K, K) global covariance
        I = torch.eye(C.shape[0], device=C.device)

        try:
            L, V = torch.linalg.eigh(C + ridge * I)
        except RuntimeError:
            logger.info("torch.linalg.eigh failed to converge")
            U, S, _ = torch.linalg.svd((C + ridge * I).to(torch.float64))
            L, V = S.flip(-1).to(dtype), U.flip(-1).to(dtype)

        scale = 1.0 / torch.sqrt(L.clamp(min=0.0))
        W = (V @ torch.diag(scale)).to(dtype)
        eigvals = (L - ridge).clamp(min=0.0)  # already ascending from eigh

    else:
        n_samples = len(x)
        mu = x.mean(dim=0, keepdim=True)
        x_c = x - mu

        try:
            _, S, Vh = torch.linalg.svd(x_c, full_matrices=False)
        except RuntimeError:
            x64 = x_c.to(torch.float64)
            _, S, Vh = torch.linalg.svd(x64, full_matrices=False)
            S, Vh = S.to(dtype), Vh.to(dtype)

        V = Vh.transpose(-2, -1)
        scale = 1.0 / torch.sqrt(S**2 / (n_samples - 1) + ridge)
        W = (V @ torch.diag(scale)).to(dtype)
        eigvals = (S**2 / (n_samples - 1)).flip(0).clamp(min=0.0)

    if return_eig:
        return mu, W, eigvals
    else:
        return mu, W