import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Sinusoidal timestep embedding for diffusion models.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0,1,0,0))
    return emb

class SCAttention(nn.Module):
    """
    Attention mechanism with spatial and channel components. 
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Spatial attention for precipitation patterns
        self.spatial_conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        
        # Channel attention for feature importance
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Spatial attention - focus on precipitation patterns
        spatial_att = torch.sigmoid(self.spatial_conv(x))
        
        # Channel attention - focus on important features
        avg_pool = self.channel_avg(x).squeeze(-1).squeeze(-1)
        max_pool = self.channel_max(x).squeeze(-1).squeeze(-1)
        channel_att = self.channel_fc(avg_pool + max_pool).unsqueeze(-1).unsqueeze(-1)
        
        return x * spatial_att * channel_att

class AttResBlock(nn.Module):
    """
    Convolution block with residual connections and attention.
    """
    def __init__(self, in_channels, out_channels, emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.residual = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Timestep embedding
        self.emb_proj = None if emb_dim is None else nn.Linear(emb_dim, out_channels)
            
        # Attention
        self.attention = SCAttention(out_channels)
        
    def forward(self, x, t_emb=None):
        residual = x if self.residual is None else self.residual(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        if self.emb_proj is not None and t_emb is not None:
            emb = self.emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            out = out + emb
        out = F.relu(self.bn2(self.conv2(out)))
        
        # Apply attention
        out = self.attention(out)
        
        return out + residual

class DownBlock(nn.Module):
    """Downsampling block with precipitation-specific features."""
    def __init__(self, in_channels, out_channels, emb_dim=None):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = AttResBlock(in_channels, out_channels, emb_dim)
        
    def forward(self, x, t_emb=None):
        x = self.maxpool(x)
        return self.conv(x, t_emb)

class UpBlock(nn.Module):
    """Upsampling block with precipitation-specific features."""
    def __init__(self, in_channels, out_channels, emb_dim=None, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = AttResBlock(in_channels, out_channels, emb_dim)
        
    def forward(self, decoder_features, encoder_skip, t_emb=None):
        # First upsampling 
        upsampled = self.up(decoder_features)

        # Then skip connection
        # Handle size mismatch
        diffY = encoder_skip.size()[2] - upsampled.size()[2]
        diffX = encoder_skip.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([encoder_skip, upsampled], dim=1)

        return self.conv(x, t_emb)

class AttResUnet(nn.Module):
    def __init__(self, in_channels, cond_channels=None, out_channels=1, 
                 base_c=64, emb_dim=256, bilinear=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.cond_channels = cond_channels
        self.use_cond = cond_channels is not None and cond_channels > 0
        
        inc_channels = in_channels + (cond_channels if self.use_cond else 0)
        
        # Initial convolution
        self.inc = AttResBlock(inc_channels, base_c, emb_dim)
        
        # Encoder (downsampling)
        self.down1 = DownBlock(base_c, base_c * 2, emb_dim)
        self.down2 = DownBlock(base_c * 2, base_c * 4, emb_dim)
        self.down3 = DownBlock(base_c * 4, base_c * 8, emb_dim)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            AttResBlock(base_c * 8, base_c * 16, emb_dim),
            AttResBlock(base_c * 16, base_c * 16, emb_dim),
            AttResBlock(base_c * 16, base_c * 8, emb_dim)
        ])
        
        # Decoder (upsampling)
        self.up1 = UpBlock(base_c * 8, base_c * 4, emb_dim, bilinear)
        self.up2 = UpBlock(base_c * 4, base_c * 2, emb_dim, bilinear)
        self.up3 = UpBlock(base_c * 2, base_c, emb_dim, bilinear)
        
        # Final convolution - output real values for log-transformed precipitation
        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)
        
    def forward(self, x, cond=None, t=None):
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected x to be 4D tensor, got {x.dim()}D")
        if cond is not None and cond.dim() != 4:
            raise ValueError(f"Expected cond to be 4D tensor, got {cond.dim()}D")
        if t is not None and t.dim() != 1:
            raise ValueError(f"Expected t to be 1D tensor, got {t.dim()}D")
        
        # Concatenate condition if provided
        if self.use_cond and cond is not None:
            x = torch.cat([x, cond], dim=1)
            
        # Get timestep embedding
        t_emb = get_timestep_embedding(t, self.emb_dim) if t is not None else None
        
        # Encoder path
        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        
        # Enhanced bottleneck
        x5 = x4
        for block in self.bottleneck:
            x5 = block(x5, t_emb)
        
        # Decoder path
        x = self.up1(x5, x3, t_emb)
        x = self.up2(x, x2, t_emb)
        x = self.up3(x, x1, t_emb)
        
        # Final output
        return self.final_conv(x) 