import torch
import torch.nn as nn
import pytorch_lightning as pl

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_factor=1., min_weight=1.0, max_weight=5.0):
        super().__init__()
        self.weight_factor = weight_factor
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def forward(self, pred, target):
        # Base MSE (per element)
        mse = (pred - target) ** 2
        base = target.detach()
        
        # Compute weights from base
        weights = torch.clamp(
            self.min_weight + self.weight_factor * base,
            min=self.min_weight,
            max=self.max_weight
        )
        
        # Apply weights
        weighted_mse = mse * weights
        return torch.mean(weighted_mse)


class Direct(pl.LightningModule):
    def __init__(self, model, lr=1e-4, optimizer_cls=torch.optim.Adam):
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.criterion = WeightedMSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['coarse']  # [B, C, H, W]
        y = batch['target']  # [B, 1, H, W]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['coarse']
        y = batch['target']
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
    def predict_step(self, batch, batch_idx):
        x = batch['coarse']
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), lr=self.lr)
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
    
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

