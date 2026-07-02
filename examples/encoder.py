import torch
import torch.nn as nn

from torch import Tensor

class FiLMResidualBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.mlp_cond = nn.Linear(cond_dim, dim * 2)
        self.block1 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.block2 = nn.Linear(dim, dim)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor, cond_embed: Tensor) -> Tensor:
        cond_params = self.mlp_cond(cond_embed)
        gamma, beta = torch.chunk(cond_params, 2, dim=-1)

        residual = self.block1(x)
        residual = self.norm(residual)
        residual = residual * (1 + gamma) + beta
        residual = self.activation(residual)
        residual = self.block2(residual)

        return self.activation(x + residual)

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t: Tensor) -> Tensor:
        t = t[:, None].float()
        sinusoid = torch.cat([
            torch.sin(t * self.inv_freq),
            torch.cos(t * self.inv_freq)
        ], dim=-1)
        return sinusoid

class TimeConditionedEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int=128, 
        t_dim: int=128,
        num_layers: int=6,
    ) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(dim=t_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            FiLMResidualBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.final_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_embed = self.time_mlp(self.time_embedding(t))
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, t_embed)
        return self.final_proj(x)