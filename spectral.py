import torch
import logging
from torch import Tensor
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def whitening(
    x: Tensor, 
    eps: float, 
    ridge: float,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the centering mean and the ZCA-style whitening matrix.
    
    Uses Symmetric Eigendecomposition (eigh) to find the inverse square root 
    of the covariance matrix. Includes robust error handling for rank-deficient 
    or unstable matrices.
    Args:
        x: Features of shape (B, K).
        eps: Small scalar for numerical stability.
        ridge: Ridge factor so that cov is not singular.

    Returns:
        eigenvalues: Tensor of shape (K,) representing the diagonalized correlations.
    """
    dtype = x.dtype
    device = x.device
    n_samples = x.shape[0]
    n_features = x.shape[1]

    # Centering
    mu = x.mean(dim=0, keepdim=True)
    x_c = x - mu

    # Covariance Calculation
    cov = (x_c.T @ x_c) / (n_samples - 1) # (K, K)
    cov = 0.5 * (cov + cov.T)
    ridge_id = ridge * torch.eye(n_features, device=device, dtype=cov.dtype)
    matrix_to_decompose = cov + ridge_id

    try:
        s, U = torch.linalg.eigh(matrix_to_decompose)
    except (RuntimeError, torch._C._LinAlgError):
        logger.warning("linalg.eigh failed, attempting fallback with increased ridge")
        # Fallback: Move to float64 and increase regularization
        matrix_f64 = matrix_to_decompose.to(torch.float64)
        ridge_extra = (ridge * 10) * torch.eye(n_features, device=device, dtype=torch.float64)
        s, U = torch.linalg.eigh(matrix_f64 + ridge_extra)
        s, U = s.to(dtype), U.to(dtype)

    s_stable = torch.clamp(s, min=s.max().item() * 1e-12 + eps)
    inv_sqrt_s = torch.diag(1.0 / torch.sqrt(s_stable))
    cov_inv_sqrt = (U @ inv_sqrt_s).to(dtype)
    return mu, cov_inv_sqrt
     
def compute_eigenvalues(
    x: Tensor,
    y: Tensor,
    mu: Tensor,
    whitener: Tensor,
    std: Optional[Tensor] = None,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """
    Computes the approximate eigenvalues (correlations) between two sets of 
    whitened features. 

    In the context of the Courant-Fischer theorem, this estimates the 
    Rayleigh quotient in the transformed feature space.

    Args:
        x: First feature set of shape (B, K).
        y: Second feature set of shape (B, K).
        mu: Mean vector used for centering, shape (K,).
        whitener: Whitening matrix (e.g., from ZCA or PCA), shape (K, K).
        std: Optional pre-computed standard deviation for normalization.
        normalize: Whether to scale features to unit variance.
        eps: Small constant for numerical stability.

    Returns:
        eigenvalues: Tensor of shape (K,) representing the diagonalized correlations.
    """
    # 1. Centering and Whitening (Linear Transformation)
    # Using matmul (@) for (B, K) @ (K, K) -> (B, K)
    x_w = (x - mu) @ whitener # (B, K)
    y_w = (y - mu) @ whitener # (B, K)

    # 2. Variance Normalization
    if normalize:
        if std is None:
            x_w = x_w / (x_w.std(0, keepdim=True) + eps)
            y_w = y_w / (y_w.std(0, keepdim=True) + eps)
        else:
            x_w = x_w / (std + eps)
            y_w = y_w / (std + eps)

    # 3. Correlation Estimation (eigenvalues)
    eigenvalues = (x_w * y_w).mean(0)

    return eigenvalues # (K,)
