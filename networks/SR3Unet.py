import torch

import torch.nn as nn
import torch.nn.functional as F

from .AttResUnet import get_timestep_embedding

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if emb_dim is not None:
            self.emb_proj = nn.Linear(emb_dim, out_channels)
        else:
            self.emb_proj = None

    def forward(self, x, t_emb=None):
        h = self.double_conv[0](x)
        h = self.double_conv[1](h)
        h = self.double_conv[2](h)
        if self.emb_proj is not None and t_emb is not None:
            # Add timestep embedding after first conv+bn+relu
            emb = self.emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + emb
        h = self.double_conv[3](h)
        h = self.double_conv[4](h)
        h = self.double_conv[5](h)
        return h

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=None):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, emb_dim)

    def forward(self, x, t_emb=None):
        x = self.maxpool(x)
        return self.conv(x, t_emb)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=None, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, emb_dim)

    def forward(self, x1, x2, t_emb=None):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, t_emb)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SR3Unet(nn.Module):
    def __init__(self, in_channels, cond_channels=None, out_channels=1, base_c=64, emb_dim=256, bilinear=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.cond_channels = cond_channels
        self.use_cond = cond_channels is not None and cond_channels > 0
        inc_channels = in_channels + (cond_channels if self.use_cond else 0)
        self.inc = DoubleConv(inc_channels, base_c, emb_dim)
        self.down1 = Down(base_c, base_c * 2, emb_dim)
        self.down2 = Down(base_c * 2, base_c * 4, emb_dim)
        self.down3 = Down(base_c * 4, base_c * 8, emb_dim)
        factor = 2 
        self.down4 = Down(base_c * 8, base_c * 16 // factor, emb_dim)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, emb_dim, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, emb_dim, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, emb_dim, bilinear)
        self.up4 = Up(base_c * 2, base_c, emb_dim, bilinear)
        self.outc = OutConv(base_c, out_channels)

    def forward(self, x, cond=None, t=None):
        # x: [B, in_channels, H, W]
        # cond: [B, cond_channels, H, W] or None
        # t: [B] (timesteps) or None
        
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected x to be 4D tensor, got {x.dim()}D")
        if cond is not None and cond.dim() != 4:
            raise ValueError(f"Expected cond to be 4D tensor, got {cond.dim()}D")
        if t is not None and t.dim() != 1:
            raise ValueError(f"Expected t to be 1D tensor, got {t.dim()}D")
        
        if self.use_cond and cond is not None:
            x = torch.cat([x, cond], dim=1)
        # If cond is not used, just use x
        t_emb = get_timestep_embedding(t, self.emb_dim) if t is not None else None
        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)
        x = self.up1(x5, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        logits = self.outc(x)
        return logits