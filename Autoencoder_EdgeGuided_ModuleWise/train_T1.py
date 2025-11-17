# ================================================================
# Dual-Branch Edge-Enhanced + Cross-Scale Transformer Fusion
# (Implements all 10 upgrades discussed)
# - EAM/FE edge-aware attention (channel + spatial)                [#1]
# - Max-gradient target from both modalities                       [#2]
# - VGG19 perceptual loss (light)                                  [#3]
# - Optional PET color handling + color-fidelity (Lab ΔE)          [#4]
# - Multi-path (shallow + deep) fusion in decoder                  [#5]
# - EEDB-style blocks (coord + edge spatial attention, dense)      [#6]
# - Mini Cross-Scale Transformer at bottleneck                     [#7]
# - Task-aware loss presets                                        [#8]
# - Optional 2-stage curriculum: encoder denoising pretrain        [#9]
# - True dual-branch encoders + per-scale fusion + CSTF            [#10]
# ================================================================

import os, glob, math, random
from typing import Tuple, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3,7")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ---------- Optional: MS-SSIM ----------
try:
    from pytorch_msssim import ssim as ssim_fn  # pip install pytorch-msssim
except Exception as e:
    raise RuntimeError("Please install pytorch-msssim: pip install pytorch-msssim") from e

# ================================================================
# Utils: color conversions (Y, YCbCr, RGB<->Lab) and helpers
# ================================================================
def to_tensor_uint(img: Image.Image) -> torch.Tensor:
    """To tensor [0,1], shape [C,H,W]."""
    return transforms.ToTensor()(img)

def rgb_to_ycbcr_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb: [B,3,H,W] in [0,1]
    returns ycbcr: [B,3,H,W], Y in [0,1], Cb/Cr roughly [0,1]
    """
    r, g, b = rgb[:,0:1], rgb[:,1:2], rgb[:,2:3]
    # BT.601
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = -0.168736*r - 0.331264*g + 0.5*b + 0.5
    cr = 0.5*r - 0.418688*g - 0.081312*b + 0.5
    return torch.cat([y, cb.clamp(0,1), cr.clamp(0,1)], dim=1)

def ycbcr_to_rgb_torch(ycbcr: torch.Tensor) -> torch.Tensor:
    """
    ycbcr: [B,3,H,W], Y [0,1], Cb/Cr ~[0,1]
    returns rgb [B,3,H,W] in [0,1]
    """
    y, cb, cr = ycbcr[:,0:1], ycbcr[:,1:2]-0.5, ycbcr[:,2:3]-0.5
    r = y + 1.402*cr
    g = y - 0.344136*cb - 0.714136*cr
    b = y + 1.772*cb
    return torch.cat([r,g,b], dim=1).clamp(0,1)

def rgb_to_lab_torch(rgb: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    rgb: [B,3,H,W], [0,1], sRGB -> Lab (D65)
    Uses approximate conversion; sufficient for loss.
    """
    # sRGB -> linear
    def srgb_to_linear(u):
        a = 0.04045
        return torch.where(u <= a, u/12.92, ((u+0.055)/1.055).pow(2.4))
    lin = srgb_to_linear(rgb.clamp(0,1))

    # linear RGB -> XYZ (D65)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=lin.dtype, device=lin.device)
    B,_,H,W = lin.shape
    x = (M[0,0]*lin[:,0]+M[0,1]*lin[:,1]+M[0,2]*lin[:,2]).unsqueeze(1)
    y = (M[1,0]*lin[:,0]+M[1,1]*lin[:,1]+M[1,2]*lin[:,2]).unsqueeze(1)
    z = (M[2,0]*lin[:,0]+M[2,1]*lin[:,1]+M[2,2]*lin[:,2]).unsqueeze(1)

    # Normalize by white point D65 (Xn,Yn,Zn)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = x/(Xn+eps), y/(Yn+eps), z/(Zn+eps)

    def f(t):
        delta = 6/29
        return torch.where(t > delta**3, t.pow(1/3), t/(3*delta**2) + 4/29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    # Scale L to [0,100], a/b roughly [-110,110]
    lab = torch.cat([L, a, b], dim=1)
    return lab

def deltaE76_torch(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """ΔE*76 = Euclidean in Lab. lab*: [B,3,H,W]; returns [B,1,H,W]."""
    return torch.sqrt(((lab1 - lab2)**2).sum(dim=1, keepdim=True) + 1e-6)

# ================================================================
# Dataset for MRI/PET pairs (optionally color PET)
# ================================================================
class PairFolder(Dataset):
    """
    Expects:
      root/mri/*.{png,jpg,jpeg}
      root/pet/*.{png,jpg,jpeg}
    Matching by filename stem. PET can be grayscale or RGB (flag).
    """
    def __init__(self, root: str, size: int = 128, exts=("*.png","*.jpg","*.jpeg"),
                 augment=True, pet_color: bool=False):
        self.mri_dir = os.path.join(root, "mri")
        self.pet_dir = os.path.join(root, "pet")
        mri_files, pet_files = [], []
        for p in exts:
            mri_files += glob.glob(os.path.join(self.mri_dir, p))
            pet_files += glob.glob(os.path.join(self.pet_dir, p))

        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        mri_map = {stem(p): p for p in mri_files}
        pet_map = {stem(p): p for p in pet_files}
        common = sorted(set(mri_map.keys()) & set(pet_map.keys()))
        self.pairs = [(mri_map[s], pet_map[s]) for s in common]
        if len(self.pairs) == 0:
            raise FileNotFoundError(f"No matching pairs in {self.mri_dir} and {self.pet_dir}")

        self.size = size
        self.augment = augment
        self.pet_color = pet_color
        self.rand_hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.rand_vflip = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self): return len(self.pairs)

    def _rand_crop_pair(self, a: Image.Image, b: Image.Image) -> Tuple[Image.Image, Image.Image]:
        W, H = a.size
        size = self.size
        if W < size or H < size:
            scale = max(size / W, size / H)
            newW, newH = int(round(W * scale)), int(round(H * scale))
            a = a.resize((newW, newH), Image.BICUBIC)
            b = b.resize((newW, newH), Image.BICUBIC)
            W, H = newW, newH
        x = random.randint(0, W - size)
        y = random.randint(0, H - size)
        box = (x, y, x + size, y + size)
        return a.crop(box), b.crop(box)

    def __getitem__(self, idx: int):
        mri_path, pet_path = self.pairs[idx]
        mri = Image.open(mri_path).convert("L")         # always 1ch
        pet = Image.open(pet_path).convert("RGB" if self.pet_color else "L")

        mri, pet = self._rand_crop_pair(mri, pet)

        if self.augment:
            if random.random() < 0.5:
                mri = self.rand_hflip(mri); pet = self.rand_hflip(pet)
            if random.random() < 0.5:
                mri = self.rand_vflip(mri); pet = self.rand_vflip(pet)

        mri_t = to_tensor_uint(mri)     # [1,H,W]
        pet_t = to_tensor_uint(pet)     # [1,H,W] or [3,H,W]
        # Construct model input:
        #   - Always include MRI 1ch as first channel
        #   - If PET color, include Y only for fusion input (keep CbCr for color loss)
        if pet_t.shape[0] == 3:
            ycbcr = rgb_to_ycbcr_torch(pet_t.unsqueeze(0)).squeeze(0)  # [3,H,W]
            pet_y = ycbcr[0:1]
            x = torch.cat([mri_t, pet_y], dim=0)  # [2,H,W]
            return x, mri_t, pet_t  # keep pet RGB for color loss
        else:
            x = torch.cat([mri_t, pet_t], dim=0)  # [2,H,W]
            return x, mri_t, pet_t

# ================================================================
# Core ops: Sobel gradient, TV, Charbonnier
# ================================================================
class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1., 2., 1.],
                           [0., 0., 0.],
                           [-1., -2., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x_1ch: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(device=x_1ch.device, dtype=x_1ch.dtype)
        ky = self.ky.to(device=x_1ch.device, dtype=x_1ch.dtype)
        gx = F.conv2d(x_1ch, kx, padding=1)
        gy = F.conv2d(x_1ch, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        amax = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (mag / amax).clamp(0.0, 1.0)

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def tv_loss(img):
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (dy.abs().mean() + dx.abs().mean())

# ================================================================
# Attention & Blocks: FE/EAM, CoordAttention, EdgeSpatial, EEDB
# ================================================================
class CoordAttention(nn.Module):
    """Coordinate attention (direction-aware channel attention)."""
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        hidden = max(8, in_ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.conv_h = nn.Conv2d(hidden, in_ch, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(hidden, in_ch, kernel_size=1, bias=True)

    def forward(self, x):
        B,C,H,W = x.shape
        x_h = self.pool_h(x)                      # [B,C,H,1]
        x_w = self.pool_w(x).transpose(2,3)       # [B,C,1,W] -> [B,C,W,1] (after conv below will transpose back)
        y = torch.cat([x_h, x_w], dim=2)          # concat along spatial height
        y = self.conv1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.transpose(2,3)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w

class EdgeSpatialAttention(nn.Module):
    """Spatial attention modulated by external edge map (auto-resizes edge)."""
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, feat, edge_1ch):
        # feat: [B,C,H,W], edge_1ch: [B,1,h,w] (any size)
        if edge_1ch.shape[-2:] != feat.shape[-2:]:
            edge_1ch = F.interpolate(edge_1ch, size=feat.shape[-2:], mode='bilinear', align_corners=False)

        avg_out = torch.mean(feat, dim=1, keepdim=True)          # [B,1,H,W]
        max_out, _ = torch.max(feat, dim=1, keepdim=True)        # [B,1,H,W]
        # Inject edge into the avg branch
        x = torch.cat([avg_out + edge_1ch, max_out], dim=1)      # [B,2,H,W]
        att = torch.sigmoid(self.conv(x))                        # [B,1,H,W]
        return feat * att


class EdgeFeatureEnhancer(nn.Module):
    """
    Edge-aware CBAM-like FE: edge->conv to feature; channel & spatial attention
    F_out = F_in * (1 + γs * spatial(edge)) + F_in * (γc * Mc)

    Now robust to any edge size: auto-resizes edge to feat spatial dims.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, in_ch, kernel_size=1), nn.LeakyReLU(0.1, inplace=True)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(8, in_ch//16), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(8, in_ch//16), in_ch, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = EdgeSpatialAttention(in_ch)
        self.gamma_c = nn.Parameter(torch.tensor(0.5))
        self.gamma_s = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat, edge):
        # Resize edge to match feat spatial size
        if edge.shape[-2:] != feat.shape[-2:]:
            edge = F.interpolate(edge, size=feat.shape[-2:], mode='bilinear', align_corners=False)

        e = self.edge_proj(edge)      # [B,C,H,W] edge features for channel attention
        mc = self.ca(e)               # [B,C,1,1]
        fs = self.sa(feat, edge)      # [B,C,H,W] spatial edge-aware attention (already resized)
        return feat * (1 + self.gamma_s * fs) + feat * (self.gamma_c * mc)


class EEDBBlock(nn.Module):
    """Edge-Enhancement Dense Block: conv->conv with dense concat + coord+edge attention."""
    def __init__(self, in_ch, growth=32):
        super().__init__()
        mid = growth
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch+mid, mid, 3, padding=1), nn.ReLU(inplace=True))
        self.ca = CoordAttention(in_ch + 2*mid)
        self.ea = EdgeFeatureEnhancer(in_ch + 2*mid)

    def forward(self, x, edge):
        y1 = self.conv1(x)
        y2 = self.conv2(torch.cat([x, y1], dim=1))
        y = torch.cat([x, y1, y2], dim=1)
        y = self.ca(y)
        y = self.ea(y, edge)
        return y

# ================================================================
# Cross-Scale Transformer (mini)
# ================================================================
class CrossScaleTransformerMini(nn.Module):
    """
    Simple token mixer at the bottleneck:
      - Downsample higher-scale feat to lowest spatial size
      - Concatenate, project to tokens, apply 1-2 layers of MHA
      - Project back and reshape
    """
    def __init__(self, in_ch, num_heads=2, depth=1):
        super().__init__()
        self.proj_in = nn.Conv2d(in_ch, in_ch, 1)
        enc_layer = nn.TransformerEncoderLayer(d_model=in_ch, nhead=num_heads, batch_first=True, dim_feedforward=in_ch*2)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.proj_out = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor):
        # f_low, f_high: [B,C,H,W] with f_low being lower resolution
        B,C,H,W = f_low.shape
        f_high_ds = F.interpolate(f_high, size=(H,W), mode='bilinear', align_corners=False)
        x = self.proj_in(f_low + f_high_ds)     # [B,C,H,W]
        tokens = x.flatten(2).transpose(1,2)    # [B,HW,C]
        tokens = self.encoder(tokens)
        x = tokens.transpose(1,2).reshape(B,C,H,W)
        return self.proj_out(x)

# ================================================================
# Dual-Branch Encoder (MRI & PET), per-scale fusion, Decoder
# ================================================================
class EncoderBranch(nn.Module):
    """
    A shallow hierarchy using EEDB blocks and strided conv for downsample.
    Returns feature list: [s1, s2, s3, s4] (from high to low resolution).
    """
    def __init__(self, in_ch, base=32, growth=32):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*2, base*2
        self.stem = nn.Sequential(nn.Conv2d(in_ch, c1, 3, padding=1), nn.ReLU(True))
        self.e1 = EEDBBlock(c1, growth=growth)
        self.down1 = nn.Conv2d(c1+2*growth, c2, 3, stride=2, padding=1)

        self.e2 = EEDBBlock(c2, growth=growth)
        self.down2 = nn.Conv2d(c2+2*growth, c3, 3, stride=2, padding=1)

        self.e3 = EEDBBlock(c3, growth=growth)
        self.down3 = nn.Conv2d(c3+2*growth, c4, 3, stride=2, padding=1)

        self.e4 = EEDBBlock(c4, growth=growth)

    def forward(self, x, edge):
        s1 = self.e1(self.stem(x), edge)                # [B,C1',H,W]
        x2 = self.down1(s1);  s2 = self.e2(x2, edge)    # [B,C2',H/2,W/2]
        x3 = self.down2(s2);  s3 = self.e3(x3, edge)    # [B,C3',H/4,W/4]
        x4 = self.down3(s3);  s4 = self.e4(x4, edge)    # [B,C4',H/8,W/8]
        return [s1, s2, s3, s4]

class PerScaleFusion(nn.Module):
    """Fuse MRI/PET features at the same scale via concat + 1x1 + FE edge gating."""
    def __init__(self, ch_mri, ch_pet, out_ch):
        super().__init__()
        self.mix = nn.Conv2d(ch_mri + ch_pet, out_ch, 1)
        self.fe = EdgeFeatureEnhancer(out_ch)

    def forward(self, fm, fp, edge):
        x = self.mix(torch.cat([fm, fp], dim=1))
        return self.fe(x, edge)

class FusionDecoder(nn.Module):
    """
    U-like decoder with shallow+deep parallel heads before final fusion.
    Also predicts auxiliary edge.
    """
    def __init__(self, ch_s1, ch_s2, ch_s3, ch_s4, out_nc=1):
        super().__init__()
        # up paths
        self.up4 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

        # deep path (more convs)
        self.d4 = nn.Sequential(nn.Conv2d(ch_s4, ch_s4, 3, padding=1), nn.ReLU(True))
        self.d3 = nn.Sequential(nn.Conv2d(ch_s3+ch_s4, ch_s3, 3, padding=1), nn.ReLU(True))
        self.d2 = nn.Sequential(nn.Conv2d(ch_s2+ch_s3, ch_s2, 3, padding=1), nn.ReLU(True))
        self.d1 = nn.Sequential(nn.Conv2d(ch_s1+ch_s2, ch_s1, 3, padding=1), nn.ReLU(True))

        # shallow path (lighter convs)
        self.s4 = nn.Conv2d(ch_s4, ch_s4, 1)
        self.s3 = nn.Conv2d(ch_s3+ch_s4, ch_s3, 1)
        self.s2 = nn.Conv2d(ch_s2+ch_s3, ch_s2, 1)
        self.s1 = nn.Conv2d(ch_s1+ch_s2, ch_s1, 1)

        # combine shallow + deep before final head
        self.combine = nn.Conv2d(2*ch_s1, ch_s1, 3, padding=1)
        self.head = nn.Sequential(nn.Conv2d(ch_s1, out_nc, 3, padding=1), nn.Sigmoid())

        # aux edge from the penultimate feature
        self.edge_head = nn.Sequential(
            nn.Conv2d(ch_s1, ch_s1, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(ch_s1, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, fs: List[torch.Tensor]):
        """
        fs: [s1,s2,s3,s4] fused features (s4 lowest-res)
        """
        s1, s2, s3, s4 = fs

        # deep path
        x4d = self.d4(s4)
        x3d = self.d3(torch.cat([s3, self.up4(x4d)], dim=1))
        x2d = self.d2(torch.cat([s2, self.up3(x3d)], dim=1))
        x1d = self.d1(torch.cat([s1, self.up2(x2d)], dim=1))

        # shallow path
        x4s = self.s4(s4)
        x3s = self.s3(torch.cat([s3, self.up4(x4s)], dim=1))
        x2s = self.s2(torch.cat([s2, self.up3(x3s)], dim=1))
        x1s = self.s1(torch.cat([s1, self.up2(x2s)], dim=1))

        feat = self.combine(torch.cat([x1d, x1s], dim=1))
        fused = self.head(feat)
        edge_pred = self.edge_head(feat)
        return fused, edge_pred

# ================================================================
# Top-level FusionNet
# ================================================================
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

class FusionNet(nn.Module):
    """
    Dual-branch encoder (MRI & PET-Y), per-scale fusion + mini CSTF, decoder to fused Y.
    """
    def __init__(self, input_nc=2, out_nc=1, base=32, growth=32, cstf_heads=2, cstf_depth=1):
        super().__init__()
        assert input_nc == 2, "Expected input_nc=2: [MRI_1ch, PET_Y_1ch]"
        self.edge_extractor = EdgeExtractor()
        # Branch encoders
        self.enc_mri = EncoderBranch(in_ch=1, base=base, growth=growth)
        self.enc_pet = EncoderBranch(in_ch=1, base=base, growth=growth)

        # Per-scale fusion (after each scale)
        # Determine channel dims after EEDB: roughly base + 2*growth at first, then conv downs
        c1 = base + 2*growth
        c2 = base*2 + 2*growth
        c3 = base*2 + 2*growth
        c4 = base*2 + 2*growth
        self.fuse1 = PerScaleFusion(c1, c1, c1)
        self.fuse2 = PerScaleFusion(c2, c2, c2)
        self.fuse3 = PerScaleFusion(c3, c3, c3)
        self.fuse4 = PerScaleFusion(c4, c4, c4)

        # Mini CSTF at bottleneck (use scales 3 & 4)
        self.cstf = CrossScaleTransformerMini(in_ch=c4, num_heads=cstf_heads, depth=cstf_depth)

        # Decoder
        self.decoder = FusionDecoder(ch_s1=c1, ch_s2=c2, ch_s3=c3, ch_s4=c4, out_nc=out_nc)

    def forward(self, x_2ch: torch.Tensor):
        mri = x_2ch[:,0:1]
        pet_y = x_2ch[:,1:2]
        edge = self.edge_extractor(mri)

        fm = self.enc_mri(mri, edge)  # [s1..s4]
        fp = self.enc_pet(pet_y, edge)

        s1 = self.fuse1(fm[0], fp[0], edge)
        s2 = self.fuse2(fm[1], fp[1], edge)
        s3 = self.fuse3(fm[2], fp[2], edge)
        s4 = self.fuse4(fm[3], fp[3], edge)

        # cross-scale transformer at bottleneck
        s4_enh = self.cstf(s4, s3)

        fused, edge_pred = self.decoder([s1, s2, s3, s4_enh])
        return fused, edge_pred, edge  # return edge used (for logging if needed)

    # helpers for compatibility
    def encode_with_edge(self, x_2ch: torch.Tensor):
        mri = x_2ch[:,0:1]
        pet_y = x_2ch[:,1:2]
        edge = self.edge_extractor(mri)
        fm = self.enc_mri(mri, edge)
        fp = self.enc_pet(pet_y, edge)
        return [fm, fp], edge

    def decode_from_feats(self, feats_tuple: List[List[torch.Tensor]]):
        fm, fp = feats_tuple
        # For safety, fuse again at inference path
        # (expect fm[k] & fp[k] to be same resolution)
        # NOTE: edge is not available here; use zeros (no-op) – or require forward path
        raise NotImplementedError("Use forward(x) for this model.")


# ================================================================
# Losses with all upgrades
# ================================================================
class VGGPerceptual(nn.Module):
    def __init__(self, layers=(2,7,16,25), weight=1.0, requires_grad=False):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, l+1)]))
            prev = l+1
        if not requires_grad:
            for p in self.parameters(): p.requires_grad = False
        self.weight = weight

    def forward(self, x, y):
        # x,y: [B,1,H,W] or [B,3,H,W] in [0,1] -> replicate to 3ch if needed and normalize
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1); y = y.repeat(1,3,1,1)
        # VGG expects ImageNet normalization
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
        x_n = (x - mean)/std
        y_n = (y - mean)/std
        loss = 0.0
        for sl in self.slices:
            x_n = sl(x_n); y_n = sl(y_n)
            loss = loss + F.l1_loss(x_n, y_n)
        return self.weight * loss

class EdgeGuidedFusionLossV2(nn.Module):
    """
    Total:
      λs_m * (1-SSIM(Y,MRI)) + λs_p * (1-SSIM(Y,PET_Y))
    + λl_m * |Y - MRI| + λl_p * |Y - PET_Y|
    + λg * sum_scales || ∇Y - max(∇MRI, ∇PET_Y) ||_Charb
    + λaux * |Ehat - ∇MRI|
    + λtv * TV(Y)
    + λperc * VGG(Y, MRI and/or PET)
    + λcol * ΔE76( Y⊕CbCr_pet , PET_RGB )
    """
    def __init__(self,
                 w_ssim_mri=4.0, w_ssim_pet=1.0,
                 w_l1_mri=1.0,  w_l1_pet=0.25,
                 w_grad=2.0, w_aux=1.0, w_tv=0.5,
                 w_pet_sal=1.0, w_perc=0.02, w_color=0.1,
                 task:str='petmri'):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sobel = SobelGrad()
        self.w_ssim_mri = w_ssim_mri
        self.w_ssim_pet = w_ssim_pet
        self.w_l1_mri = w_l1_mri
        self.w_l1_pet = w_l1_pet
        self.w_grad = w_grad
        self.w_aux = w_aux
        self.w_tv = w_tv
        self.w_pet_sal = w_pet_sal
        self.w_perc = w_perc
        self.w_color = w_color
        self.task = task
        self.vgg = VGGPerceptual(weight=1.0)

    def multi_scale_maxgrad(self, y, mri, pet_y):
        def sob(a): return self.sobel(a)
        g_y1 = sob(y)
        g_m1 = sob(mri)
        g_p1 = sob(pet_y)
        g_max1 = torch.maximum(g_m1, g_p1)

        # half scale
        y2, m2, p2 = F.avg_pool2d(y,2,2), F.avg_pool2d(mri,2,2), F.avg_pool2d(pet_y,2,2)
        g_y2, g_m2, g_p2 = sob(y2), sob(m2), sob(p2)
        g_max2 = torch.maximum(g_m2, g_p2)

        return charbonnier(g_y1 - g_max1).mean() + charbonnier(g_y2 - g_max2).mean()

    def pet_saliency(self, pet_y):
        # emphasize bright PET regions
        pet_n = (pet_y - pet_y.amin(dim=(2,3), keepdim=True)) / \
                (pet_y.amax(dim=(2,3), keepdim=True) - pet_y.amin(dim=(2,3), keepdim=True) + 1e-6)
        return torch.sigmoid(6.0 * (pet_n - 0.4))  # [B,1,H,W]

    def forward(self, y, mri, pet, ehat):
        """
        y: fused [B,1,H,W]; mri: [B,1,H,W]; pet: [B,1,H,W] or [B,3,H,W]
        """
        if pet.shape[1] == 3:
            ycbcr = rgb_to_ycbcr_torch(pet)
            pet_y, pet_cb, pet_cr = ycbcr[:,0:1], ycbcr[:,1:2], ycbcr[:,2:3]
        else:
            pet_y = pet

        # SSIM & L1
        ssim_m = 1.0 - ssim_fn(y, mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet_y, data_range=1.0)
        l1_m = self.l1(y, mri)
        l1_p = self.l1(y, pet_y)

        # PET saliency-weighted L1
        S_pet = self.pet_saliency(pet_y)
        l1_pet_sal = (S_pet * (y - pet_y).abs()).mean()

        # Gradient consistency to max(∇MRI, ∇PET_Y)
        g_cons = self.multi_scale_maxgrad(y, mri, pet_y)

        # Aux edge target: MRI Sobel
        target_e = self.sobel(mri).detach()
        aux = self.l1(ehat, target_e)

        # TV
        tv = tv_loss(y)

        # Perceptual loss (light) – compare fused Y to both sources (avg)
        perc = 0.5*self.vgg(y, mri) + 0.5*self.vgg(y, pet_y)

        # Color fidelity if PET is RGB: construct fused RGB by (Y_fused ⊕ CbCr_pet)
        color = torch.tensor(0.0, device=y.device)
        if pet.shape[1] == 3:
            ycbcr_fused = torch.cat([y, ycbcr[:,1:2], ycbcr[:,2:3]], dim=1)
            rgb_fused = ycbcr_to_rgb_torch(ycbcr_fused).clamp(0,1)
            lab_fused = rgb_to_lab_torch(rgb_fused)
            lab_pet   = rgb_to_lab_torch(pet)
            de = deltaE76_torch(lab_fused, lab_pet).mean()
            color = de

        # Task-aware weights (simple presets)
        if self.task.lower() == 'ctmri':
            w_ssim_mri = self.w_ssim_mri*1.2; w_ssim_pet = self.w_ssim_pet*0.5
            w_l1_mri   = self.w_l1_mri*1.2;  w_l1_pet   = self.w_l1_pet*0.5
        else:
            w_ssim_mri = self.w_ssim_mri;    w_ssim_pet = self.w_ssim_pet
            w_l1_mri   = self.w_l1_mri;      w_l1_pet   = self.w_l1_pet

        loss = (w_ssim_mri*ssim_m + w_ssim_pet*ssim_p +
                w_l1_mri*l1_m + w_l1_pet*l1_p +
                self.w_pet_sal*l1_pet_sal +
                self.w_grad*g_cons + self.w_aux*aux + self.w_tv*tv +
                self.w_perc*perc + self.w_color*color)

        terms = {
            "ssim_m": float(ssim_m.item()), "ssim_p": float(ssim_p.item()),
            "l1_m": float(l1_m.item()), "l1_p": float(l1_p.item()),
            "g_cons": float(g_cons.item()), "aux": float(aux.item()), "tv": float(tv.item()),
            "perc": float(perc.item()), "color": float(color.item()) if torch.is_tensor(color) else float(color)
        }
        return loss, terms

# ================================================================
# Optional Stage-1: Encoder denoising pretrain (noise-conditioned)
# ================================================================
class ReconHead(nn.Module):
    """Tiny upsampling head from lowest-scale encoder feature to reconstruct a 1ch image."""
    def __init__(self, ch_low, ch_mid, out_ch=1):
        super().__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
                                 nn.Conv2d(ch_low, ch_mid, 3, padding=1), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
                                 nn.Conv2d(ch_mid, ch_mid, 3, padding=1), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
                                 nn.Conv2d(ch_mid, ch_mid//2, 3, padding=1), nn.ReLU(True))
        self.out = nn.Sequential(nn.Conv2d(ch_mid//2, out_ch, 3, padding=1), nn.Sigmoid())

    def forward(self, s1,s2,s3,s4):
        x = self.up1(s4)
        x = self.up2(x)
        x = self.up3(x)
        return self.out(x)

# ================================================================
# Training
# ================================================================
def seed_everything(seed=1234):
    import numpy as np, random as pyrand
    pyrand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    import argparse
    parser = argparse.ArgumentParser("Dual-branch Edge+Transformer MRI-PET fusion trainer")
    parser.add_argument("--data_root", type=str, default="/data6/yasir/data/train", help="contains mri/ and pet/")
    parser.add_argument("--save_dir",  type=str, default="./checkpoints_fusion_v2")
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--batch_size",type=int, default=8)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--size",      type=int, default=128)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--amp",       action="store_true", help="use mixed precision")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pet_color", action="store_true", help="load PET as RGB and use color loss")
    parser.add_argument("--pretrain_encoder", action="store_true", help="stage-1 denoising pretrain")
    parser.add_argument("--epochs_pre", type=int, default=3)

    # weights
    parser.add_argument("--w_ssim_mri", type=float, default=3.0)
    parser.add_argument("--w_ssim_pet", type=float, default=2.0)
    parser.add_argument("--w_l1_mri",   type=float, default=1.0)
    parser.add_argument("--w_l1_pet",   type=float, default=0.25)
    parser.add_argument("--w_grad",     type=float, default=2.0)
    parser.add_argument("--w_aux",      type=float, default=1.0)
    parser.add_argument("--w_tv",       type=float, default=0.5)
    parser.add_argument("--w_pet_sal",  type=float, default=1.0)
    parser.add_argument("--w_perc",     type=float, default=0.02)
    parser.add_argument("--w_color",    type=float, default=0.1)
    parser.add_argument("--task",       type=str, default="petmri", choices=["petmri","ctmri"])

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(42)

    # Data
    train_set = PairFolder(args.data_root, size=args.size, augment=True, pet_color=args.pet_color)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    device = args.device

    # Model
    model = FusionNet(input_nc=2, out_nc=1, base=32, growth=32, cstf_heads=2, cstf_depth=1).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = EdgeGuidedFusionLossV2(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad=args.w_grad, w_aux=args.w_aux, w_tv=args.w_tv,
        w_pet_sal=args.w_pet_sal, w_perc=args.w_perc, w_color=args.w_color,
        task=args.task
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ------------------------------------------------------------
    # Stage-1: optional encoder denoising pretrain (noise curriculum)
    # ------------------------------------------------------------
    if args.pretrain_encoder:
        recon_head_mri = ReconHead(ch_low=32*2+2*32, ch_mid=128).to(device)  # matches EncoderBranch last channels
        recon_head_pet = ReconHead(ch_low=32*2+2*32, ch_mid=128).to(device)
        pre_params = list(model.enc_mri.parameters()) + list(model.enc_pet.parameters()) + \
                     list(recon_head_mri.parameters()) + list(recon_head_pet.parameters())
        opt_pre = torch.optim.Adam(pre_params, lr=args.lr)
        sched_pre = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pre, T_max=args.epochs_pre)

        print("==> Stage-1: encoder denoising pretrain")
        for epoch in range(1, args.epochs_pre+1):
            pbar = tqdm(train_loader, desc=f"[Pretrain] Epoch {epoch}/{args.epochs_pre}", ncols=120)
            epoch_loss, steps = 0.0, 0
            for x, mri, pet in pbar:
                x, mri, pet = x.to(device), mri.to(device), pet.to(device)
                opt_pre.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    # pick source randomly (MRI or PET_Y) and add noise
                    if pet.shape[1] == 3:
                        pet_y = rgb_to_ycbcr_torch(pet)[:,0:1]
                    else:
                        pet_y = pet
                    if random.random() < 0.5:
                        src = mri
                        fm = model.enc_mri(src, model.edge_extractor(src))  # [s1..s4]
                        rec = recon_head_mri(*fm)
                    else:
                        src = pet_y
                        fp = model.enc_pet(src, model.edge_extractor(mri))  # edge from MRI for stability
                        rec = recon_head_pet(*fp)
                    sigma = random.choice([0.02, 0.05, 0.1])
                    noisy = (src + sigma*torch.randn_like(src)).clamp(0,1)
                    loss_pre = F.l1_loss(rec, src) + 0.5*F.mse_loss(rec, noisy)  # denoise & identity
                scaler.scale(loss_pre).backward()
                scaler.unscale_(opt_pre)
                torch.nn.utils.clip_grad_norm_(pre_params, 1.0)
                scaler.step(opt_pre); scaler.update()

                epoch_loss += float(loss_pre.item()); steps += 1
                pbar.set_postfix(loss=f"{epoch_loss/max(1,steps):.4f}")
            sched_pre.step()
        # drop heads
        del recon_head_mri, recon_head_pet

    # ------------------------------------------------------------
    # Stage-2: Fusion training
    # ------------------------------------------------------------
    print("==> Stage-2: fusion training")
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=140)
        epoch_loss, steps = 0.0, 0

        for x, mri, pet in pbar:
            x, mri, pet = x.to(device), mri.to(device), pet.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                fused, edge_pred, edge_in = model(x)     # fused [B,1,H,W]
                loss, terms = EdgeGuidedFusionLossV2.__call__(loss_fn, fused, mri, pet, edge_pred)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()

            epoch_loss += float(loss.item()); steps += 1
            pbar.set_postfix(
                loss=f"{epoch_loss/max(1,steps):.4f}",
                ssim_m=f"{terms['ssim_m']:.3f}",
                ssim_p=f"{terms['ssim_p']:.3f}",
                grad=f"{terms['g_cons']:.3f}",
                aux=f"{terms['aux']:.3f}",
                perc=f"{terms['perc']:.3f}",
                col=f"{terms['color']:.3f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}"
            )

        scheduler.step()

        # save best + last
        avg_loss = epoch_loss / max(1, steps)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "fusion_best.pth"))
        torch.save(model.state_dict(), os.path.join(args.save_dir, "fusion_last.pth"))

    print("Training complete. Checkpoints saved to:", args.save_dir)

if __name__ == "__main__":
    main()
