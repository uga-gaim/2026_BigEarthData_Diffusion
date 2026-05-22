import torch
import os
import pickle
import random
from tqdm.auto import tqdm
from typing import Optional, List

import logging
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

logger = logging.getLogger(__name__)


class DDPM:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [
                torch.tensor([1.], device=self.device),
                self.alphas_cumprod[:-1]
            ],
            dim=0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


class Diffusion(pl.LightningModule):
    def __init__(self, model, lr=1e-4, optimizer_cls=torch.optim.Adam, timesteps=1000, beta_start=1e-4, beta_end=0.02, cond_dropout_prob=0.1):
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.criterion = nn.MSELoss()
        self.diffusion = DDPM(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end, device=self.device)
        self.timesteps = timesteps
        self.cond_dropout_prob = cond_dropout_prob

    def forward(self, noisy_hr, coarse, t):
        return self.model(noisy_hr, coarse, t)

    def training_step(self, batch, batch_idx):
        x_start = batch['target']  # clean high-res
        coarse = batch['coarse']   # coarse input
        land_mask = batch['target_land_mask']
        
        B = x_start.size(0)
        device = x_start.device
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start, t, noise)
        x_noisy[~land_mask] = x_start[~land_mask]
        
        # Classifier-free guidance: randomly drop condition during training
        if self.training and torch.rand(1).item() < self.cond_dropout_prob:
            coarse_input = torch.zeros_like(coarse)
        else:
            coarse_input = coarse
        
        pred_noise = self(x_noisy, coarse_input, t)

        # Calculate loss only for land mask
        masked_pred = pred_noise[land_mask]
        masked_noise = noise[land_mask]
        noise_loss = self.criterion(masked_pred, masked_noise)

        # --- Regularization on Reconstruction: x0_pred vs x_start ---
        alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t])[:, None, None, None]

        x0_pred = (x_noisy - sqrt_one_minus_alphas_cumprod * pred_noise) / sqrt_alphas_cumprod
        recon_loss = self.criterion(x0_pred[land_mask], x_start[land_mask])

        recon_weight = 0.5
        noise_weight = 1 - recon_weight
        
        total_loss = noise_weight * noise_loss + recon_weight * recon_loss

        self.log('train_loss', total_loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log('recon_loss', recon_loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x_start = batch['target']
        coarse = batch['coarse']
        land_mask = batch['target_land_mask']
        
        B = x_start.size(0)
        device = x_start.device
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start, t, noise)
        x_noisy[~land_mask] = x_start[~land_mask]

        pred_noise = self(x_noisy, coarse, t)
        
        # Calculate loss only for land mask for predictions and targets
        masked_pred = pred_noise[land_mask]
        masked_noise = noise[land_mask]
        loss = self.criterion(masked_pred, masked_noise)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        coarse = batch['coarse'].to(self.device)
        land_mask = batch['target_land_mask'].to(self.device)
        known_data = batch['target'].to(self.device)
        B, _, H, W = known_data.shape

        residual = self.ddpm_sample(
            coarse=coarse,
            shape=(B, 1, H, W),
            land_mask=land_mask,
            known_data=known_data,
        )

        return {
            'residual': residual,
            'forecast_init_time': batch['forecast_init_time'],
            'valid_time': batch['valid_time'],
        }

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), lr=self.lr)
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=100,
                min_lr=1e-6,
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
    
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


    @torch.no_grad()
    def ddim_sample(
        self,
        coarse,
        shape,
        land_mask,
        known_data,
        ddim_steps=50,
        eta=0.0,
        guidance_scale=1.0,
        temperature=1.0,
        return_progressive=False,
    ):
        device = coarse.device
        B = shape[0]
        img = torch.randn(shape, device=device) * temperature
        img[~land_mask] = known_data[~land_mask]
        timesteps = np.linspace(0, self.timesteps-1, ddim_steps, dtype=int)[::-1]
        alphas_cumprod = self.diffusion.alphas_cumprod

        progressive_imgs = []

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            if guidance_scale == 1.0:
                # Standard unconditional sampling
                pred_noise = self(img, coarse, t_tensor)
            else:
                # Classifier-free guidance: assume model can take None for unconditional
                # Make sure you have trained the model with conditional dropout 
                pred_noise_uncond = self(img, torch.zeros_like(coarse), t_tensor)
                pred_noise_cond = self(img, coarse, t_tensor)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

            # Apply temperature scaling to noise prediction
            pred_noise = pred_noise * temperature

            alpha_cumprod = alphas_cumprod[t]
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
            x0_pred = (img - sqrt_one_minus_alpha_cumprod * pred_noise) / sqrt_alpha_cumprod

            if i < ddim_steps - 1:
                next_t = timesteps[i+1]
                alpha_cumprod_next = alphas_cumprod[next_t]
                sigma = eta * torch.sqrt((1 - alpha_cumprod_next) / (1 - alpha_cumprod)) * torch.sqrt(1 - alpha_cumprod / alpha_cumprod_next)
                noise = torch.randn_like(img) * temperature if eta > 0 else 0
                img = torch.sqrt(alpha_cumprod_next) * x0_pred + torch.sqrt(1 - alpha_cumprod_next - sigma**2) * pred_noise + sigma * noise
            else:
                img = x0_pred

            img[~land_mask] = known_data[~land_mask]

            if return_progressive:
                progressive_imgs.append(img.clone().detach().cpu())

        if return_progressive:
            return img, progressive_imgs
        else:
            return img

    @torch.no_grad()
    def ddpm_sample(
        self,
        coarse,
        shape,
        land_mask,
        known_data,
        ddpm_steps=None,
        temperature=1.0,
        return_progressive=False,
        pbar=False,
    ):
        device = coarse.device
        B = shape[0]
        img = torch.randn(shape, device=device) * temperature
        img[~land_mask] = known_data[~land_mask]
        if ddpm_steps is not None:
            indices = np.linspace(0, self.timesteps-1, ddpm_steps, dtype=int)[::-1]
        else:
            indices = list(range(self.timesteps))[::-1]
        alphas_cumprod = self.diffusion.alphas_cumprod
        betas = self.diffusion.betas
        progressive_imgs = []
        for t in tqdm(indices, disable=not pbar):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            # Clamp known region before model prediction
            img[~land_mask] = known_data[~land_mask]
            pred_noise = self(img, coarse, t_tensor)
            beta = betas[t]
            alpha = 1.0 - beta

            alpha_cumprod = alphas_cumprod[t]
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
            x0_pred = (img - sqrt_one_minus_alpha_cumprod * pred_noise) / sqrt_alpha_cumprod

            # Compute mean and variance for posterior
            if t > 0:
                noise = torch.randn_like(img) * temperature
            else:
                noise = torch.zeros_like(img)
            coef1 = (
                torch.sqrt(self.diffusion.alphas_cumprod_prev[t]) * beta / (1. - self.diffusion.alphas_cumprod[t])
                if t > 0 else torch.tensor(0.0, device=device)
            )
            coef2 = (
                torch.sqrt(alpha) * (1. - self.diffusion.alphas_cumprod_prev[t]) / (1. - self.diffusion.alphas_cumprod[t])
                if t > 0 else torch.tensor(0.0, device=device)
            )
            mean = (coef1 * x0_pred + coef2 * img) if t > 0 else x0_pred
            var = (betas[t] * (1. - self.diffusion.alphas_cumprod_prev[t]) / (1. - self.diffusion.alphas_cumprod[t])) if t > 0 else torch.tensor(0.0, device=device)
            std = torch.sqrt(var) if t > 0 else torch.tensor(0.0, device=device)
            img = (mean + std * noise) if t > 0 else mean
            # Clamp known region after update (for extra safety)
            img[~land_mask] = known_data[~land_mask]
            if return_progressive:
                progressive_imgs.append(img.clone().detach().cpu())
        if return_progressive:
            return img, progressive_imgs
        else:
            return img

    def on_fit_start(self):
        device = next(self.parameters()).device
        self.diffusion.to(device)
        logger.info(f"Training on device: {device}")


class SampleCallback(pl.Callback):
    def __init__(self, dataset, save_dir, name, version, every_n_epochs=100, sample_indices: Optional[List[int]] = None):
        super().__init__()
        self.val_dataset = dataset
        self.every_n_epochs = every_n_epochs

        self.output_dir = os.path.join(save_dir, name, version, "samples")
        os.makedirs(self.output_dir, exist_ok=True)

        if sample_indices is None:
            self.sample_indices = random.sample(range(len(self.val_dataset)), 4)
        else:
            self.sample_indices = sample_indices

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()
        all_samples = []
        all_mse = []
        with torch.no_grad():
            for idx in self.sample_indices:
                batch = self.val_dataset[idx]
                x_start = batch['target'].to(pl_module.device).unsqueeze(0)
                coarse = batch['coarse'].to(pl_module.device).unsqueeze(0)
                land_mask = batch['target_land_mask'].to(pl_module.device).unsqueeze(0)

                # Generate samples
                samples = pl_module.ddim_sample(
                    coarse=coarse,
                    shape=x_start.shape,
                    land_mask=land_mask,
                    known_data=x_start,
                )
                mse = torch.nanmean((samples - x_start) ** 2).item()
                all_mse.append(mse)
                all_samples.append(samples.cpu().numpy())
        save_path = os.path.join(self.output_dir, f"samples_{epoch:05d}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(np.concatenate(all_samples, axis=0), f)
        avg_mse = float(np.nanmean(all_mse)) if len(all_mse) > 0 else np.nan
        pl_module.log("val_sample_mse", avg_mse, prog_bar=True)
