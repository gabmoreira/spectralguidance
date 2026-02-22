#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Tuple, Optional, Dict

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard DDPM/DDIM sinusoidal time embeddings.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeConditionedResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups=32
    ) -> None:
        super().__init__()
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Projection for time embedding to scale/shift the features
        self.time_proj = nn.Linear(time_emb_dim, out_channels * 2)
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Time conditioning: Scale and Shift
        # 1. Project time embedding to (2 * out_channels)
        # 2. Split into scale and shift
        # 3. Unsqueeze to match spatial dims (B, C, 1, 1)
        t_emb = self.time_proj(F.silu(time_emb))
        t_emb = t_emb[:, :, None, None]
        scale, shift = t_emb.chunk(2, dim=1)
        
        # Modulate
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class TimeConditionedEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        out_dim: int,
        time_emb_dim: int,
        base_channels: int,
        channel_mults: Tuple[int],
        min_resolution: int,
        max_channels: int,
        num_train_timesteps: int,
        num_timesteps: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        assert image_size & (image_size - 1) == 0, "image_size must be power of 2"
        self.num_train_timesteps = num_train_timesteps

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)

        num_downs = int(math.log2(image_size // min_resolution))
        downs = []
        in_ch = base_channels
        for i in range(num_downs):
            mult = channel_mults[min(i, len(channel_mults) - 1)]
            out_ch = min(base_channels * mult, max_channels)
            downs.append(TimeConditionedResBlock(in_ch, out_ch, time_emb_dim))
            downs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        self.downs = nn.ModuleList(downs)
        
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, out_dim)

        if num_timesteps:
            self.register_buffer("whiten_timesteps", torch.zeros(num_timesteps, dtype=torch.long))
            self.register_buffer("whiten_w", torch.zeros((num_timesteps, out_dim, out_dim)))
            self.register_buffer("whiten_mu", torch.zeros(num_timesteps, out_dim))
            self.register_buffer("whiten_norm", torch.zeros((num_timesteps, out_dim)))
            self._whiten_timestep_to_idx = {}

    def rebuild_whitening_index(self) -> None:
        if self.whiten_mu is None:
            return
        timesteps = self.whiten_timesteps
        self._whiten_timestep_to_idx = {int(t): int(i) for i, t in enumerate(timesteps)}

    @torch.no_grad()
    def set_whitening(
        self,
        w: Dict[int, Tensor],
        mu: Dict[int, Tensor],
        norm: Dict[int, Tensor],
    ) -> None:
        timesteps = sorted(w.keys())
        self.register_buffer("whiten_timesteps", torch.tensor(timesteps, dtype=torch.long))
        self.register_buffer("whiten_w", torch.stack([w[t] for t in timesteps]))
        self.register_buffer("whiten_mu", torch.stack([mu[t].flatten() for t in timesteps]))   
        self.register_buffer("whiten_norm", torch.stack([norm[t].flatten() for t in timesteps]))

    def forward_whitened(self, x: Tensor, timesteps: Tensor) -> Tensor:
        raw_phi = self.forward(x, timesteps) # (B, K)

        idx = torch.tensor(
            [self._whiten_timestep_to_idx[int(t)] for t in timesteps],
            device=x.device,
        )

        phi = raw_phi - self.whiten_mu[idx]
        phi = torch.bmm(phi.unsqueeze(1), self.whiten_w[idx]).squeeze(1)
        phi = phi / self.whiten_norm[idx]
        ones = torch.ones(len(phi), 1, device=x.device)
        return torch.cat((phi, ones), dim=-1)

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor: 
        """
        x: (B, 3, 256, 256)
        timesteps: (B,) tensor of ints or floats
        """
        timesteps = timesteps.float() * (1000.0 / self.num_train_timesteps)

        t_emb = self.time_mlp(timesteps)
        x = self.init_conv(x)
        
        for layer in self.downs:
            if isinstance(layer, TimeConditionedResBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.avg_pool(x) # (B, C, 1, 1)
        x = x.flatten(1)     # (B, C)
        x = self.fc(x)       # (B, k)
        return x