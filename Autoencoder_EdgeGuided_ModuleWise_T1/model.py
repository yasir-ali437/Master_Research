# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ----------------------------
# Edge extractor (Sobel mag, per-image normalized)
# ----------------------------
class EdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1., 2., 1.],
                           [0., 0., 0.],
                           [-1., -2., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_kx", kx)
        self.register_buffer("sobel_ky", ky)

    def forward(self, x_1ch: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x_1ch, self.sobel_kx, padding=1)
        gy = F.conv2d(x_1ch, self.sobel_ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        amax = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (mag / amax).clamp(0.0, 1.0)

# ----------------------------
# Soft brain ROI from MRI (differentiable)
# ----------------------------
class SoftBrainMask(nn.Module):
    """
    Soft mask to downweight background/skull in losses and sharpening.
    y = sigmoid(k * (mri - t))
    """
    def __init__(self, t: float = 0.10, k: float = 20.0):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(t), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(k), requires_grad=False)

    def forward(self, mri_1ch: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.k * (mri_1ch - self.t))

# ----------------------------
# Small helpers for learnable unsharp masking
# ----------------------------
def _gaussian_kernel(k: int = 5, sigma: float = 1.2) -> torch.Tensor:
    ax = torch.arange(k) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

class GaussianBlur2d(nn.Module):
    def __init__(self, channels: int = 1, k: int = 5, sigma: float = 1.2):
        super().__init__()
        ker = _gaussian_kernel(k, sigma).view(1, 1, k, k)
        weight = ker.repeat(channels, 1, 1, 1)  # depthwise
        self.register_buffer("weight", weight)
        self.k = k
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype, device=x.device)
        pad = self.k // 2
        return F.conv2d(x, w, padding=pad, groups=self.groups)

class LearnableUnsharp(nn.Module):
    """
    y' = clamp( y + gain * (y - blur(y)), 0, 1 )
    gain = max_gain * sigmoid(gain_logit) in [0, max_gain]
    Applied only inside a (soft) ROI mask.
    """
    def __init__(self, max_gain: float = 1.0, k: int = 5, sigma: float = 1.2):
        super().__init__()
        self.max_gain = max_gain
        self.gain_logit = nn.Parameter(torch.tensor(0.0))  # starts ~0.5*max_gain
        self.blur = GaussianBlur2d(channels=1, k=k, sigma=sigma)

    def forward(self, y: torch.Tensor, roi: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        gain = self.max_gain * torch.sigmoid(self.gain_logit)
        detail = y - self.blur(y)
        y_sharp = y + gain * detail
        if roi is not None:
            y_sharp = roi * y_sharp + (1.0 - roi) * y  # sharpen only where brain exists
        return y_sharp.clamp(0.0, 1.0), gain.detach()

# ----------------------------
# Edge-Guided Attention
# ----------------------------
class EdgeGuidedAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(1, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        att = torch.sigmoid(self.proj(edge))
        return feat * (1.0 + self.gamma * att)

# ----------------------------
# Encoder
# ----------------------------
class FusionEncoder(nn.Module):
    def __init__(self, input_nc=2, base_ch=32, base_ch2=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc + 1, base_ch, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch * 2, base_ch2, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2, base_ch2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 2, base_ch2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 3, base_ch2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.att1 = EdgeGuidedAttention(base_ch)
        self.att2 = EdgeGuidedAttention(base_ch)

    def forward(self, x_2ch: torch.Tensor, edge: torch.Tensor):
        x0 = torch.cat([x_2ch, edge], dim=1)
        G11 = self.att1(self.conv1(x0), edge)
        G21 = self.att2(self.conv2(G11), edge)
        G31 = self.conv3(torch.cat([G11, G21], 1))
        G41 = self.conv4(G31)
        G51 = self.conv5(torch.cat([G31, G41], 1))
        G61 = self.conv6(torch.cat([G31, G41, G51], 1))
        return [G11, G21, G31, G41, G51, G61]

# ----------------------------
# Decoder + auxiliary edge head
# ----------------------------
class FusionDecoder(nn.Module):
    def __init__(self, base_ch=32, base_ch2=64, output_nc=1):
        super().__init__()
        self.conv66 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2, base_ch2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv55 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 2, base_ch2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv44 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 2, base_ch2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv33 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 2, base_ch2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 + base_ch, base_ch, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv11 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch * 2, output_nc, kernel_size=3, stride=1),
            nn.Sigmoid()
        )
        self.up = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False)

        self.edge_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, 1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, feats: List[torch.Tensor]):
        G6_2 = self.conv66(feats[5])
        G5_2 = self.conv55(torch.cat([feats[4], G6_2], 1))
        G4_2 = self.conv44(torch.cat([feats[3], G5_2], 1))
        G3_2 = self.conv33(torch.cat([feats[2], G4_2], 1))
        G2_2 = self.conv22(torch.cat([feats[1], self.up(G3_2)], 1))
        fused = self.conv11(torch.cat([feats[0], G2_2], 1))
        edge_pred = self.edge_head(G2_2)
        return fused, edge_pred

# ----------------------------
# Top-level model
# ----------------------------
class Fusion_net(nn.Module):
    """
    - Encodes x=[MRI,PET]
    - Decodes to fused image in [0,1] + edge prediction
    - Applies learnable unsharp inside brain ROI to lift edge metrics
    """
    def __init__(self, input_nc=2, output_nc=1, base_ch=32, base_ch2=64,
                 roi_t: float = 0.10, roi_k: float = 20.0, unsharp_max_gain: float = 1.0):
        super().__init__()
        self.edge_extractor = EdgeExtractor()
        self.brain_masker = SoftBrainMask(t=roi_t, k=roi_k)
        self.encoder = FusionEncoder(input_nc=input_nc, base_ch=base_ch, base_ch2=base_ch2)
        self.decoder = FusionDecoder(base_ch=base_ch, base_ch2=base_ch2, output_nc=output_nc)
        self.unsharp = LearnableUnsharp(max_gain=unsharp_max_gain, k=5, sigma=1.2)

    def forward(self, x_2ch: torch.Tensor):
        mri = x_2ch[:, 0:1]
        edge = self.edge_extractor(mri)
        feats = self.encoder(x_2ch, edge)
        fused, edge_pred = self.decoder(feats)
        roi = self.brain_masker(mri)
        fused_sharp, gain = self.unsharp(fused, roi)
        return fused_sharp, edge_pred, roi, gain

    # (Kept for compatibility if you need features)
    def encode_with_edge(self, x_2ch: torch.Tensor):
        mri = x_2ch[:, 0:1, :, :]
        edge = self.edge_extractor(mri)
        feats = self.encoder(x_2ch, edge)
        return feats, edge

    def decode_from_feats(self, feats: List[torch.Tensor]):
        return self.decoder(feats)
