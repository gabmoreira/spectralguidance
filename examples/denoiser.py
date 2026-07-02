import numpy as np
import torch
import torch.nn as nn

from torch import Tensor

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        half_dim = self.dim // 2

        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        time = time.float() # Ensure time is float
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if self.dim % 2 == 1:
            embeddings = nn.functional.pad(embeddings, (0, 1))

        return embeddings

class DenoiserResidualBlock(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim + time_emb_dim, dim)

        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = torch.cat([x, t_emb], dim=1)
        h = self.proj_in(h)
        return x + self.net(h)


class Denoiser(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_blocks: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.time_emb_dim = hidden * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        self.initial_proj = nn.Linear(in_dim, hidden)
        self.res_blocks = nn.ModuleList([
            DenoiserResidualBlock(hidden, self.time_emb_dim) for _ in range(num_blocks)
        ])
        self.final_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, in_dim)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_embed(t)
        h = self.initial_proj(x)
        for block in self.res_blocks:
            h = block(h, t_emb)
        return self.final_proj(h)