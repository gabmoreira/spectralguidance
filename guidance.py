#!/usr/bin/env python3
import time
import torch
import logging
import numpy as np
from pathlib import Path
from torch import Tensor
from omegaconf import OmegaConf
from diffusers import DDIMPipeline
from torchvision.transforms.functional import to_pil_image
from PIL.Image import Image
from typing import List, Union, Tuple, Dict, Optional
from tqdm.auto import tqdm

from model import TimeConditionedEncoder

logger = logging.getLogger(__name__)

class SpectralGuidance:
    """
    Spectral guidance class for conditional sampling using DDIM and spectral encodings.
    """
    def __init__(
        self,
        model_path: str,
        guidance_tmin: int = 0,
        guidance_tmax: int = 1000,
        device: str = "cuda",
        xformers: bool = True,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Initialize the spectral guidance model.

        Args:
            model_path (str): Path to folder containing checkpoint and config.
            device (str): Device for computations ('cpu' or 'cuda').
        """
        self._model_path = Path(model_path)
        self._guidance_tmin = guidance_tmin
        self._guidance_tmax = guidance_tmax
        self._device = device
        self._dtype = dtype

        self._config_path = self._model_path / ".hydra" / "config.yaml"
        self._state_dict_path = self._model_path / "state_dict_whitened.pt"

        assert self._model_path.exists(), f"Model path does not exist: {self._model_path}"
        assert self._config_path.exists(), "Config file missing"
        assert self._state_dict_path .exists(), "State dict + Whitening missing"

        logger.info("Loading config from: %s", self._config_path)
        self._cfg = OmegaConf.load(self._config_path)

        logger.info("Loading state dict from: %s", self._state_dict_path)
        state_dict = torch.load(self._state_dict_path, map_location="cpu", weights_only=False)

        self._pipeline = DDIMPipeline.from_pretrained(self._cfg.scheduler.pretrained)
        self._pipeline.to(device)
        self._pipeline.unet.to(device, dtype=self._dtype)
        self._pipeline.unet.eval()
        if xformers:
            self._pipeline.unet.enable_xformers_memory_efficient_attention() 
        logger.info(
            "Initialized DDIM pipeline from pretrained: %s",
            self._cfg.scheduler.pretrained
        )

        self._phi_encoder = TimeConditionedEncoder(
            image_size=self._cfg.dataset.image_size,
            out_dim=self._cfg.model.num_eigenfunctions,
            time_emb_dim=self._cfg.model.time_emb_dim,
            base_channels=self._cfg.model.base_channels,
            channel_mults=self._cfg.model.channel_mults,
            min_resolution=self._cfg.model.min_resolution,
            max_channels=self._cfg.model.max_channels,
            num_train_timesteps=self._cfg.scheduler.num_train_timesteps,
            num_timesteps=len(state_dict["whiten_timesteps"]),
        )
        self._phi_encoder.load_state_dict(state_dict)
        self._phi_encoder.rebuild_whitening_index()
        self._phi_encoder.to(self._device)
        self._phi_encoder.eval()
        logger.info(
            "Loaded phi encoder state dict (%d timesteps)",
            len(state_dict["whiten_timesteps"])
        )

        self._phi = torch.load(self._model_path / "phi.pt", map_location="cpu", weights_only=False)
        self._phi = {
            t: phi_t.to(self._device)
            for t, phi_t in self._phi.items()
            if self._guidance_tmin <= t <= self._guidance_tmax
        }
        logger.info("Loaded phi (%d timesteps)", len(self._phi))

        self._guidance_timesteps = [
            int(t)
            for t in self._phi_encoder.whiten_timesteps
            if self._guidance_tmin <= t <= self._guidance_tmax
        ]
        logger.info(
            "SpectralGuidance initialized with %d guidance timesteps.",
            len(self._guidance_timesteps)
        )

        self._spectral_coefs = {}
        
    def manual_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        logger.info("Random seed set to %d", seed)

    @torch.no_grad()
    def predict_noise(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict noise using the DDIM UNet"""
        with torch.autocast(device_type="cuda", dtype=self._dtype):
            pred_noise = self._pipeline.unet(x, t, return_dict=True)["sample"]
        return pred_noise
    
    @torch.no_grad()
    def posterior_mean(self, x: Tensor, t: int, pred_noise: Tensor) -> Tensor:
        """Compute posterior mean estimate for x0 at timestep t."""
        alpha_bar = self._pipeline.scheduler.alphas_cumprod[t]
        x0_hat = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
        return x0_hat
    
    def set_timesteps(self, timesteps: int) -> None:
        """Set the number of DDIM timesteps for the scheduler."""
        self._pipeline.scheduler.set_timesteps(timesteps, device=self._device)
        self._ddim_timesteps = [int(t) for t in self._pipeline.scheduler.timesteps]
        logger.info("Scheduler timesteps set to %d", timesteps)

    def compute_coefficients(self, t: int, p_y_x0: Tensor) -> Tensor:
        """Project guidance tensor onto spectral basis."""
        n = len(self._phi[t])
        assert p_y_x0.shape == (n,), f"p_y_x0 must have shape ({n},)"
        coefs = (self._phi[t].T @ p_y_x0.float()) / n
        return coefs

    def set_guidance(self, y: Tensor) -> None:
        """Precomputes spectral coefficients."""
        self._spectral_coefs = {}
        for t in self._guidance_timesteps:
            self._spectral_coefs[t] = self.compute_coefficients(t, y) # shape (K,)
    
    def compute_psi(self, x: Tensor, num_samples: int = 50) -> Dict[int, Tensor]:
        """Computes psi (minus eigenvalue scaling)"""
        psi = {}
        for t in self._guidance_timesteps:
            psi[t] = []
            for _ in range(num_samples):
                noise = torch.randn_like(x)
                t_batch = torch.full((len(x),), t, device=self._device)
                x_t = self._pipeline.noise_scheduler.add_noise(x, noise, t_batch)
                with torch.no_grad():
                    phi = self._phi_encoder.forward_whitened(x_t, t_batch)
                psi[t].append(phi)
            psi[t] = torch.cat(psi[t]).mean(0)
        return psi

    def sample(
        self,
        num_samples: int,
        guidance_strength: float,
        eta: float,
        return_posterior_mean: bool = False,
        batch_size: int = 16,
        guidance_tmin: int = 0,
        guidance_tmax: int = 1000,
        grad_clip_min: float = 0.1, 
        grad_clip_max: float = 0.3, 
        top_k: Optional[int] = None,
        verbose: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """See definition of batch_sample"""
        samples = []
        posterior_mean = []
        count = 0
        with tqdm(total=num_samples, unit="sample", disable=not verbose) as pbar:
            while count < num_samples:
                batch = self.sample_batch(
                    num_samples=min(batch_size, num_samples - count),
                    guidance_strength=guidance_strength,
                    eta=eta,
                    return_posterior_mean=return_posterior_mean,
                    guidance_tmin=guidance_tmin,
                    guidance_tmax=guidance_tmax,
                    grad_clip_min=grad_clip_min,
                    grad_clip_max=grad_clip_max,
                    top_k=top_k,
                )

                if return_posterior_mean:
                    batch_samples, batch_posterior_mean = batch
                    posterior_mean.append(batch_posterior_mean)
                else:
                    batch_samples = batch
                    
                count += len(batch_samples)
                samples.append(batch_samples)
                pbar.update(len(batch_samples))
                
        samples = torch.cat(samples)
        
        if return_posterior_mean:
            return samples, torch.cat(posterior_mean, dim=1)
        else:
            return samples
    
    def sample_batch(
        self,
        num_samples: int,
        guidance_strength: float,
        eta: float,
        return_posterior_mean: bool = False,
        guidance_tmin: int = 0,
        guidance_tmax: int = 1000,
        grad_clip_min: float = 0.1,
        grad_clip_max: float = 0.3,
        top_k: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Generate samples conditioned on guidance.

        Args:
            num_samples (int): Number of samples to generate.
            p_y_x0 (Tensor): Guidance tensor for target.
            guidance_strength (float): Strength of spectral guidance.
            eta (float): DDIM eta parameter.
            return_posterior_mean (bool): Whether to return posterior mean.
            grad_clip_min (float): Min gradient norm for clamping.
            grad_clip_max (float): Max gradient norm for clamping.

        Returns:
            Tensor or Tuple[Tensor, Dict[int, List[Image]]]: Generated samples and optionally posterior mean.
        """
        if top_k is None:
            top_k = self._cfg.model.num_eigenfunctions + 1

        if torch.cuda.is_available():
            current_usage = torch.cuda.memory_allocated() / (1024**3)
            #print(f"Memory before start: {current_usage:.3f} GB")
            torch.cuda.reset_peak_memory_stats(self._device)
            
        # --- METRICS STORAGE ---
        metrics = {
            "step_times": [],   # Time for one guide_step
            "image_times": [],  # Time for full generation of one image
            "peak_memory": [],  # Store peak MB per batch
        }
        #print(f"Benchmarking with Batch Size: {num_samples}...")
        
        x_shape = (num_samples, 3, self._cfg.dataset.image_size, self._cfg.dataset.image_size)
        x = torch.randn(x_shape, device=self._device)

        posterior_mean = []
        # --- START BATCH TIMER ---
        torch.cuda.synchronize()
        batch_start = time.perf_counter()
        for t in self._ddim_timesteps:
            # --- START STEP TIMER ---
            if torch.cuda.is_available(): 
                torch.cuda.synchronize()
                step_start = time.perf_counter()
            
            t_batch = torch.full((len(x),), t, device=self._device, dtype=torch.long)
            pred_noise = self.predict_noise(x, t_batch)

            if t in self._guidance_timesteps and t >= guidance_tmin and t <= guidance_tmax:
                if return_posterior_mean:
                    x0_hat = self.posterior_mean(x, t, pred_noise)
                    posterior_mean.append(x0_hat)
                    
                x.requires_grad_(True)
                if x.grad is not None:
                    x.grad.zero_()

                c = self._spectral_coefs[t] # shape (K,)
                phi_x = self._phi_encoder.forward_whitened(x, t_batch) # shape (num_samples, K)
                probs = phi_x[:, -top_k:] @ c[-top_k:] / phi_x[:, -top_k:].pow(2).sum(-1).sqrt() # shape (self._n,)
                probs.backward(torch.ones_like(probs))
                grad_log_p = x.grad.detach() / probs.detach().view(-1,1,1,1)
                grad_norm = grad_log_p.flatten(1, 3).norm(dim=1).view(-1,1,1,1)
                grad_norm_clamped = torch.clamp(grad_norm, min=grad_clip_min, max=grad_clip_max)
                guidance_score = grad_log_p / grad_norm * grad_norm_clamped

                current_alpha_bar = self._pipeline.scheduler.alphas_cumprod[t]
                sigma_t = (1 - current_alpha_bar)
                guidance = guidance_strength * guidance_score * sigma_t**2
                x = x + guidance
                x = x.detach()
                logger.debug("Step t=%d applied spectral guidance", t)

            x = self._pipeline.scheduler.step(pred_noise, t, x, eta=eta).prev_sample

             # --- END STEP TIMER ---
            torch.cuda.synchronize()
            step_end = time.perf_counter()
            metrics["step_times"].append((step_end - step_start) * 1000)

        # --- END BATCH TIMER ---
        torch.cuda.synchronize()
        batch_end = time.perf_counter()
        # Calculate time per image (handling BS > 1 just in case)
        total_batch_time = batch_end - batch_start
        time_per_image = total_batch_time / num_samples
        metrics["image_times"].append(time_per_image)
        # --- MEMORY TRACKING END ---
        peak_mem_bytes = 0
        peak_mem_bytes = torch.cuda.max_memory_allocated(self._device)
        peak_mem_gb = peak_mem_bytes / (1024**3)
        # --- REPORTING ---
        avg_step_ms = np.mean(metrics["step_times"])
        avg_img_s = np.mean(metrics["image_times"])
        #print("\n" + "="*30)
        #print(f"BENCHMARK REPORT (BS={num_samples})")
        #print(f"Avg Step Latency:  {avg_step_ms:.2f} ms")
        #print(f"Avg Image Latency: {avg_img_s:.4f} s")
        #print(f"Peak VRAM Usage:    {peak_mem_gb:.3f} GB")
        #print("="*30 + "\n")
        
        if return_posterior_mean:
            return x, torch.stack(posterior_mean)
        else:
            return x
    
    @staticmethod
    def to_pil_image(x: Tensor) -> List[Image]:
        """Convert tensor images to PIL images."""
        return [to_pil_image(((x[i] + 1) / 2).clamp(0, 1)) for i in range(len(x))]