import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Dict

def save_eigenvalue_plot(
    eigenvalues_t,
    save_path: str,
    step: int,
    writer,
):
    evals = []
    ts = []
    for t, eigenvalues in eigenvalues_t.items():
        ts.append(t)
        idx = torch.argsort(eigenvalues_t[t]) # Will sort per timestep
        evals.append(eigenvalues[idx]) 
    evals = torch.stack(evals).cpu().numpy()
    ts = np.array(ts)
    fig, ax = plt.subplots(figsize=(6, 4))
    num_eigenfunctions = evals.shape[1]
    for i in range(num_eigenfunctions):
        c = plt.cm.turbo(i / (num_eigenfunctions-1))
        ax.plot(ts, evals[:,i], '.', c=c, markersize=2)

    ax.set_xlabel("Time ($t$)")
    ax.set_ylabel("Eigenvalue ($\lambda_k$)")
    ax.set_title(f"Spectrum of $P_t$")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if writer is not None:
        writer.add_figure('plots/eigenvalues', fig, global_step=step)
    plt.close(fig)
