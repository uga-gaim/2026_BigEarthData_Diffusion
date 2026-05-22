import torch

import pytorch_lightning as pl
import torch.nn.functional as F


# References
#
# Cherel, N., Almansa, A., Gousseau, Y., & Newson, A. (2024, August).
# Diffusion-based image inpainting with internal learning.
# In 2024 32nd European Signal Processing Conference (EUSIPCO) (pp. 446-450). IEEE.
#


class ModelRecon(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.model = net

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['masked_field']        # (B, 1, H, W)
        null_mask = batch['null_mask']
        mask = 1 - null_mask             # (B, 1, H, W), 1s for valid and 0s for missing

        y_hat = self(torch.cat([x, mask], dim=1))

        # Only compute loss on unknown values (null_mask=1)
        loss = F.mse_loss(y_hat * null_mask, batch['full_field'] * null_mask)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['masked_field']
        mask = 1 - batch['null_mask']    # (B, 1, H, W), 1s for valid and 0s for missing

        y_hat = self(torch.cat([x, mask], dim=1))
        val_loss = F.mse_loss(y_hat, batch['full_field'])

        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx=None):
        x = batch['masked_field']
        mask = 1 - batch['null_mask']
        y_hat = self(torch.cat([x, mask], dim=1))

        return {
            'prediction': y_hat,
            'full_field': batch['full_field'],
            'null_mask': batch['null_mask'],
            'masked_field': x,
        }
        
    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class ModelReconFineTune(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.model = net

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['full_field'] * batch['valid_mask_as_control']  # (B, 1, H, W)
        mask = batch['valid_mask_as_control']                     # (B, 1, H, W), 1s for valid and 0s for missing

        y_hat = self(torch.cat([x, mask], dim=1))

        loss = F.mse_loss(
                y_hat * batch['valid_mask_as_target'],
                batch['full_field'] * batch['valid_mask_as_target'])

        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['full_field'] * batch['valid_mask_as_control']
        mask = batch['valid_mask_as_control']

        y_hat = self(torch.cat([x, mask], dim=1))

        val_loss = F.mse_loss(
                y_hat * batch['valid_mask_as_target'],
                batch['full_field'] * batch['valid_mask_as_target'])

        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx=None):
        x = batch['full_field'] * batch['valid_mask_as_control']
        mask = batch['valid_mask_as_control']
        y_hat = self(torch.cat([x, mask], dim=1))

        return {
            'prediction': y_hat,
            'valid_mask_as_control': batch['valid_mask_as_control'],
            'valid_mask_as_target': batch['valid_mask_as_target'],
            'full_field': batch['full_field'],
        }

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
