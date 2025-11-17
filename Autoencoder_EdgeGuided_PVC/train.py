# -*- coding: utf-8 -*-
"""
MRI–PET Fusion with Cortex-Mask Partial Volume Correction (PVC)

- Inputs: MRI (1ch), PET (1ch), MASK (1ch; cortex binary)
- Model: Edge-guided encoder–decoder with Sobel attention (unchanged logic)
- New Loss: Cortex-mask–aware PVC terms that (a) suppress fused activity outside cortex,
            (b) match PET activity inside cortex (esp. around the boundary ring),
            and (c) keep sharp MRI edges and structure via Sobel/SSIM/L1.
- Output: Fused image in [0,1] with PET activity confined to cortex and crisp MRI edges.

Folder layout (filenames must match by stem across folders):
  data_root/
    mri/*.png|jpg
    pet/*.png|jpg
    mask/*.png|jpg  (binary ~ {0,255} or {0,1})

Usage (example):
  python train_pvc_fusion.py --data_root /path/to/data --epochs 20 --amp
"""

import os, glob, random
from typing import Tuple, List

# Optional: pick GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    from pytorch_msssim import ssim as ssim_fn  # pip install pytorch-msssim
except Exception as e:
    raise RuntimeError("Please install pytorch-msssim: pip install pytorch-msssim") from e


# ----------------------------
# Module: Edge Extraction
# ----------------------------
class EdgeExtractor(nn.Module):
    """
    Computes Sobel edge magnitude from a single-channel MRI image.
    """
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
        # x_1ch: [B,1,H,W] in [0,1]
        gx = F.conv2d(x_1ch, self.sobel_kx, padding=1)
        gy = F.conv2d(x_1ch, self.sobel_ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        amax = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (mag / amax).clamp(0.0, 1.0)


# ----------------------------
# Module: Edge-Guided Attention
# ----------------------------
class EdgeGuidedAttention(nn.Module):
    """
    Projects a 1ch edge map to a per-channel gate and scales features.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(1, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        att = torch.sigmoid(self.proj(edge))
        return feat * (1.0 + self.gamma * att)


# ----------------------------
# Module: Encoder
# ----------------------------
class FusionEncoder(nn.Module):
    """
    Encoder with edge-guided attention in first two blocks.
    NOTE: input_nc now defaults to 3 (MRI, PET, MASK); conv1 sees input_nc + 1 (adds edge).
    """
    def __init__(self, input_nc=3, base_ch=32, base_ch2=64):
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

    def forward(self, x_3ch: torch.Tensor, edge: torch.Tensor):
        # x_3ch: [B,3,H,W] -> [MRI, PET, MASK]
        x0 = torch.cat([x_3ch, edge], dim=1)  # add edge channel

        G11 = self.conv1(x0); G11 = self.att1(G11, edge)
        G21 = self.conv2(G11); G21 = self.att2(G21, edge)
        G31 = self.conv3(torch.cat([G11, G21], 1))
        G41 = self.conv4(G31)
        G51 = self.conv5(torch.cat([G31, G41], 1))
        G61 = self.conv6(torch.cat([G31, G41, G51], 1))
        return [G11, G21, G31, G41, G51, G61]


# ----------------------------
# Module: Decoder & Heads
# ----------------------------
class FusionDecoder(nn.Module):
    """
    Decoder with auxiliary edge prediction head.
    """
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
        fused = self.conv11(torch.cat([feats[0], G2_2], 1))  # [B,1,H,W] in [0,1]
        edge_pred = self.edge_head(G2_2)
        return fused, edge_pred


# ----------------------------
# Module: FusionNet
# ----------------------------
class Fusion_net(nn.Module):
    """
    Top-level fusion net; now expects 3-channel input [MRI, PET, MASK].
    """
    def __init__(self, input_nc=3, output_nc=1, base_ch=32, base_ch2=64):
        super().__init__()
        self.edge_extractor = EdgeExtractor()
        self.encoder = FusionEncoder(input_nc=input_nc, base_ch=base_ch, base_ch2=base_ch2)
        self.decoder = FusionDecoder(base_ch=base_ch, base_ch2=base_ch2, output_nc=output_nc)

    def forward(self, x_3ch: torch.Tensor):
        mri = x_3ch[:, 0:1, :, :]
        edge = self.edge_extractor(mri)
        feats = self.encoder(x_3ch, edge)
        fused, edge_pred = self.decoder(feats)
        return fused, edge_pred

    def encode_with_edge(self, x_3ch: torch.Tensor):
        mri = x_3ch[:, 0:1, :, :]
        edge = self.edge_extractor(mri)
        feats = self.encoder(x_3ch, edge)
        return feats, edge

    def decode_from_feats(self, feats: List[torch.Tensor]):
        return self.decoder(feats)


# ----------------------------
# Dataset: MRI/PET/MASK
# ----------------------------
class PairMaskFolder(Dataset):
    """
    Expects:
      root/mri/*.png|jpg
      root/pet/*.png|jpg
      root/mask/*.png|jpg   (binary cortex masks; same stem)
    """
    def __init__(self, root: str, size: int = 128, exts=("*.png","*.jpg","*.jpeg"), augment=True, bin_thresh=0.5):
        self.mri_dir = os.path.join(root, "MRI")
        self.pet_dir = os.path.join(root, "PET")
        self.mask_dir = os.path.join(root, "GM")

        def collect(d):
            files = []
            for p in exts:
                files += glob.glob(os.path.join(d, p))
            return files

        mri_files = collect(self.mri_dir)
        pet_files = collect(self.pet_dir)
        mask_files = collect(self.mask_dir)

        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        mri_map = {stem(p): p for p in mri_files}
        pet_map = {stem(p): p for p in pet_files}
        mask_map = {stem(p): p for p in mask_files}

        common = sorted(set(mri_map.keys()) & set(pet_map.keys()) & set(mask_map.keys()))
        if len(common) == 0:
            raise FileNotFoundError(f"No matching triplets found under {self.mri_dir}, {self.pet_dir}, {self.mask_dir}")

        self.triplets = [(mri_map[s], pet_map[s], mask_map[s]) for s in common]
        self.size = size
        self.augment = augment
        self.bin_thresh = bin_thresh

        self.to_tensor = transforms.ToTensor()  # [0,1]
        self.rand_hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.rand_vflip = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self):
        return len(self.triplets)

    def _load_gray(self, path: str) -> Image.Image:
        return Image.open(path).convert("L")

    def _ensure_min_size(self, imgs: List[Image.Image]) -> List[Image.Image]:
        W, H = imgs[0].size
        size = self.size
        if W < size or H < size:
            scale = max(size / W, size / H)
            newW, newH = int(round(W * scale)), int(round(H * scale))
            imgs = [im.resize((newW, newH), Image.BICUBIC) for im in imgs]
        return imgs

    def _rand_crop_triplet(self, a: Image.Image, b: Image.Image, c: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image]:
        W, H = a.size
        size = self.size
        x = random.randint(0, W - size)
        y = random.randint(0, H - size)
        box = (x, y, x + size, y + size)
        return a.crop(box), b.crop(box), c.crop(box)

    def __getitem__(self, idx: int):
        mri_path, pet_path, mask_path = self.triplets[idx]
        mri = self._load_gray(mri_path)
        pet = self._load_gray(pet_path)
        mask = self._load_gray(mask_path)

        # make sure same size and minimum size
        mri, pet, mask = self._ensure_min_size([mri, pet, mask])
        mri, pet, mask = self._rand_crop_triplet(mri, pet, mask)

        if self.augment:
            if random.random() < 0.5:
                mri = self.rand_hflip(mri); pet = self.rand_hflip(pet); mask = self.rand_hflip(mask)
            if random.random() < 0.5:
                mri = self.rand_vflip(mri); pet = self.rand_vflip(pet); mask = self.rand_vflip(mask)

        mri_t = self.to_tensor(mri)   # [1,H,W]
        pet_t = self.to_tensor(pet)   # [1,H,W]
        mask_t = self.to_tensor(mask) # [1,H,W] in [0,1]
        mask_t = (mask_t > self.bin_thresh).float()  # binarize robustly

        x = torch.cat([mri_t, pet_t, mask_t], dim=0)  # [3,H,W]
        return x, mri_t, pet_t, mask_t


# ----------------------------
# Loss Utilities
# ----------------------------
class SobelGrad(nn.Module):
    """Differentiable Sobel edge magnitude with per-image max normalization."""
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
    """Total variation on [B,1,H,W]."""
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (dy.abs().mean() + dx.abs().mean())


def dilate(mask: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Binary dilation via max-pool (mask in {0,1})."""
    pad = k // 2
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)


def erode(mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Binary erosion via min-pool (by negating and max-pooling)."""
    pad = k // 2
    return -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)


# ----------------------------
# Cortex-Mask PVC Loss
# ----------------------------
class PVECortexConstrainedFusionLoss(nn.Module):
    """
    Total loss:
      MRI structure & denoise terms:
        λs_m * (1-SSIM(Y,MRI)) + λl_m * |Y - MRI| + λg * multi-scale Sobel(Y, MRI)
        + λaux * |Ehat - Sobel(MRI)| + λtv * TV(Y)

      PET fidelity (soft saliency):
        λs_p * (1-SSIM(Y,PET)) + λl_p * |Y - PET| + λpet_sal * ⟨S_pet, |Y-PET|⟩

      NEW: Cortex-mask PVE constraints:
        λin  * ⟨M, w_in * |Y - PET|⟩                 (pull PET activity inside cortex)
        λout * ⟨(1-M), Y⟩                             (suppress fused activity outside)
        λring_in  * ⟨B_in, |Y - PET|⟩                 (focus correction on inner ring)
        λring_out * ⟨B_out, Y⟩                        (strongly suppress outer ring spill)
        λmass * | sum(Y · dil(M)) - sum(PET · dil(M)) | / (sum(PET · dil(M)) + eps)
    """
    def __init__(self,
                 # legacy terms
                 w_ssim_mri=4.0, w_ssim_pet=1.0,
                 w_l1_mri=1.0,  w_l1_pet=0.25,
                 w_grad=2.0, w_aux=1.0, w_tv=0.5, w_pet_sal=1.0,
                 # new mask/PVC terms
                 w_in=2.0, w_out=3.0,
                 w_ring_in=1.0, w_ring_out=4.0,
                 w_mass=0.1,
                 ring_out_k=7, ring_in_ero_k=5):
        super().__init__()
        self.w_ssim_mri = w_ssim_mri
        self.w_ssim_pet = w_ssim_pet
        self.w_l1_mri = w_l1_mri
        self.w_l1_pet = w_l1_pet
        self.w_grad = w_grad
        self.w_aux = w_aux
        self.w_tv = w_tv
        self.w_pet_sal = w_pet_sal

        self.w_in = w_in
        self.w_out = w_out
        self.w_ring_in = w_ring_in
        self.w_ring_out = w_ring_out
        self.w_mass = w_mass

        self.ring_out_k = ring_out_k
        self.ring_in_ero_k = ring_in_ero_k

        self.l1 = nn.L1Loss()
        self.sobel = SobelGrad()

    def multi_scale_grad(self, y, mri):
        def one(a): return self.sobel(a)
        gy1, gm1 = one(y), one(mri)
        y2, m2 = F.avg_pool2d(y, 2, 2), F.avg_pool2d(mri, 2, 2)
        gy2, gm2 = one(y2), one(m2)
        return charbonnier(gy1 - gm1).mean() + charbonnier(gy2 - gm2).mean()

    def forward(self, y, mri, pet, mask, ehat):
        # --- Normalization helpers ---
        pet_min = pet.amin(dim=(2,3), keepdim=True)
        pet_max = pet.amax(dim=(2,3), keepdim=True)
        pet_n = (pet - pet_min) / (pet_max - pet_min + 1e-6)  # [0,1]
        S_pet = torch.sigmoid(6.0 * (pet_n - 0.4))            # emphasize hot regions

        # --- Legacy structure/fidelity terms ---
        ssim_m = 1.0 - ssim_fn(y, mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet, data_range=1.0)

        l1_m = self.l1(y, mri)
        l1_p = self.l1(y, pet)
        l1_pet_sal = (S_pet * (y - pet).abs()).mean()

        g_cons = self.multi_scale_grad(y, mri)

        target_e = self.sobel(mri).detach()
        aux = self.l1(ehat, target_e)

        tv = tv_loss(y)

        # # --- Cortex-mask PVC terms ---
        # M = (mask > 0.5).float()
        # M_dil = dilate(M, self.ring_out_k)  # for mass and outer ring
        # M_ero = erode(M, self.ring_in_ero_k)

        # B_out = (M_dil - M).clamp(0, 1)     # just outside cortex boundary
        # B_in  = (M - M_ero).clamp(0, 1)     # just inside cortex boundary

        # # Inside: match PET within cortex (weighted more where PET is hot)
        # L_in = (M * (S_pet * (y - pet).abs())).sum() / (M.sum() + 1e-6)

        # # Outside: suppress fused intensity outside mask
        # L_out = ((1.0 - M) * y).sum() / ((1.0 - M).sum() + 1e-6)

        # # Boundary ring focus
        # L_ring_in  = (B_in  * (y - pet).abs()).sum() / (B_in.sum()  + 1e-6)
        # L_ring_out = (B_out * y).sum()            / (B_out.sum() + 1e-6)

        # # Mass conservation near cortex (prevents trivial suppression)
        # mass_y   = (y   * M_dil).sum(dim=(2,3))
        # mass_pet = (pet * M_dil).sum(dim=(2,3))
        # L_mass = ((mass_y - mass_pet).abs() / (mass_pet.abs() + 1e-6)).mean()

        # --- Simplified Cortex-mask PVC terms ---
        M = (mask > 0.5).float()  # Cortex mask

        # Inside: Match PET within cortex (weighted more where PET is hot)
        L_in = (M * (pet - y).abs()).sum() / (M.sum() + 1e-6)

        # Outside: Suppress fused intensity outside cortex mask
        L_out = ((1.0 - M) * y).sum() / ((1.0 - M).sum() + 1e-6)

        # Mass conservation near cortex (prevents trivial suppression)
        mass_y = (y * M).sum(dim=(2, 3))
        mass_pet = (pet * M).sum(dim=(2, 3))
        L_mass = ((mass_y - mass_pet).abs() / (mass_pet.abs() + 1e-6)).mean()
        
        loss = (
            self.w_ssim_mri * ssim_m + self.w_ssim_pet * ssim_p +
            self.w_l1_mri  * l1_m    + self.w_l1_pet  * l1_p  +
            self.w_pet_sal * l1_pet_sal +
            self.w_grad    * g_cons + self.w_aux * aux + self.w_tv * tv +
            # self.w_in      * L_in   + self.w_out * L_out +
            # self.w_ring_in * L_ring_in + self.w_ring_out * L_ring_out +
            # self.w_mass    * L_mass
            self.w_in * L_in + 
            self.w_out * L_out +
            self.w_mass * L_mass
        )

        terms = {
            "ssim_m": float(ssim_m.item()), "ssim_p": float(ssim_p.item()),
            "l1_m": float(l1_m.item()), "l1_p": float(l1_p.item()),
            "g_cons": float(g_cons.item()), "aux": float(aux.item()), "tv": float(tv.item()),
            # "L_in": float(L_in.item()), "L_out": float(L_out.item()),
            # "L_ring_in": float(L_ring_in.item()), "L_ring_out": float(L_ring_out.item()),
            # "L_mass": float(L_mass.item())
            "L_in": float(L_in.item()), "L_out": float(L_out.item()),
            "L_mass": float(L_mass.item())
        }
        return loss, terms


# ----------------------------
# Training
# ----------------------------
def seed_everything(seed=1234):
    import numpy as np, random as pyrand
    pyrand.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    import argparse
    parser = argparse.ArgumentParser("Cortex-PVC Edge-guided MRI-PET fusion trainer")
    parser.add_argument("--data_root", type=str, required=False, default='/mnt/yasir/PET_Dataset_Processed/Separated_Centered_PET_MRI_Filtered_Dataset_Filtered', help="contains mri/, pet/, mask/")
    parser.add_argument("--save_dir",  type=str, default="./checkpoints_pvc")
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--lr",        type=float, default=1e-5)
    parser.add_argument("--size",      type=int, default=128)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--amp",       action="store_true", help="use mixed precision")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # legacy weights
    parser.add_argument("--w_ssim_mri", type=float, default=2.0)
    parser.add_argument("--w_ssim_pet", type=float, default=2.0)
    parser.add_argument("--w_l1_mri",   type=float, default=1.0)
    parser.add_argument("--w_l1_pet",   type=float, default=0.25)
    parser.add_argument("--w_grad",     type=float, default=2.0)
    parser.add_argument("--w_aux",      type=float, default=1.0)
    parser.add_argument("--w_tv",       type=float, default=0.5)
    parser.add_argument("--w_pet_sal",  type=float, default=1.0)

    # NEW PVC weights
    parser.add_argument("--w_in",       type=float, default=2)
    parser.add_argument("--w_out",      type=float, default=3)
    parser.add_argument("--w_ring_in",  type=float, default=0)
    parser.add_argument("--w_ring_out", type=float, default=0)
    parser.add_argument("--w_mass",     type=float, default=1)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(42)

    # Data
    train_set = PairMaskFolder(args.data_root, size=args.size, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # Model
    model = Fusion_net(input_nc=3, output_nc=1).to(args.device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = PVECortexConstrainedFusionLoss(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad=args.w_grad, w_aux=args.w_aux, w_tv=args.w_tv, w_pet_sal=args.w_pet_sal,
        w_in=args.w_in, w_out=args.w_out, w_ring_in=args.w_ring_in,
        w_ring_out=args.w_ring_out, w_mass=args.w_mass
    ).to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Train
    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=130)
        running = 0.0
        for x, mri, pet, mask in pbar:
            x    = x.to(args.device)    # [B,3,H,W]
            mri  = mri.to(args.device)  # [B,1,H,W]
            pet  = pet.to(args.device)  # [B,1,H,W]
            mask = mask.to(args.device) # [B,1,H,W]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                feats, _edge_in = model.encode_with_edge(x)
                fused, edge_pred = model.decode_from_feats(feats)  # fused ∈ [0,1]
                loss, terms = loss_fn(fused, mri, pet, mask, edge_pred)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            global_step += 1
            pbar.set_postfix(
                loss=f"{running / (global_step or 1):.4f}",
                ssim_m=f"{terms['ssim_m']:.3f}",
                L_in=f"{terms['L_in']:.3f}",
                L_out=f"{terms['L_out']:.3f}",
                # ring_out=f"{terms['L_ring_out']:.3f}"
            )

        # Save epoch checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        # torch.save(ckpt, os.path.join(args.save_dir, f"pvc_edgeguided_e{epoch:03d}.pt"))

    # Final weights
    torch.save(model.state_dict(), os.path.join(args.save_dir, "pvc_edgeguided_final.pth"))
    print("Training complete. Checkpoints saved to:", args.save_dir)


if __name__ == "__main__":
    main()
