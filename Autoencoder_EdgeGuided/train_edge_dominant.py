import os, glob, random
from typing import Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

from net import TwoFusion_net   # use your edge-guided net from earlier


# -----------------------------
# Dataset: MRI/PET pairs
# -----------------------------
class PairFolder(Dataset):
    """
    Expects:
      root/mri/*.png (or jpg)
      root/pet/*.png (or jpg)
    Matching by filename stem.
    """
    def __init__(self, root: str, size: int = 128, exts=("*.png","*.jpg","*.jpeg"), augment=True):
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
        if len(common) == 0:
            raise FileNotFoundError(f"No matching MRI/PET stems under {self.mri_dir} and {self.pet_dir}")
        self.pairs = [(mri_map[s], pet_map[s]) for s in common]

        self.size = size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.rand_hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.rand_vflip = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self): return len(self.pairs)

    def _load_gray(self, path: str) -> Image.Image:
        return Image.open(path).convert("L")

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
        mri = self._load_gray(mri_path)
        pet = self._load_gray(pet_path)
        mri, pet = self._rand_crop_pair(mri, pet)

        if self.augment:
            if random.random() < 0.5:
                mri = self.rand_hflip(mri); pet = self.rand_hflip(pet)
            if random.random() < 0.5:
                mri = self.rand_vflip(mri); pet = self.rand_vflip(pet)

        mri_t = self.to_tensor(mri)  # [1,H,W] in [0,1]
        pet_t = self.to_tensor(pet)  # [1,H,W]
        x = torch.cat([mri_t, pet_t], dim=0)  # [2,H,W]
        return x, mri_t, pet_t


# -----------------------------
# Edge operators (Sobel/Scharr)
# -----------------------------
class GradMag(nn.Module):
    """
    Differentiable gradient magnitude with per-image max normalization.
    mode='sobel' (default) or 'scharr' (slightly better rotational accuracy).
    """
    def __init__(self, mode: str = "sobel"):
        super().__init__()
        if mode == "scharr":
            kx = torch.tensor([[3., 0., -3.],
                               [10., 0., -10.],
                               [3., 0., -3.]], dtype=torch.float32)
            ky = torch.tensor([[3., 10., 3.],
                               [0.,  0., 0.],
                               [-3., -10., -3.]], dtype=torch.float32)
        else:  # sobel
            kx = torch.tensor([[1., 0., -1.],
                               [2., 0., -2.],
                               [1., 0., -1.]], dtype=torch.float32)
            ky = torch.tensor([[1., 2., 1.],
                               [0., 0., 0.],
                               [-1., -2., -1.]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1,1,3,3))
        self.register_buffer("ky", ky.view(1,1,3,3))

    def forward(self, x_1ch: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(device=x_1ch.device, dtype=x_1ch.dtype)
        ky = self.ky.to(device=x_1ch.device, dtype=x_1ch.dtype)
        gx = F.conv2d(x_1ch, kx, padding=1)
        gy = F.conv2d(x_1ch, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        amax = mag.amax(dim=(2,3), keepdim=True).clamp_min(1e-6)
        return (mag / amax).clamp(0.0, 1.0)


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def tv_loss(img):
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (dy.abs().mean() + dx.abs().mean())


# -----------------------------
# Loss: edge-dominant fusion
# -----------------------------
class EdgeDominantFusionLoss(nn.Module):
    """
    Strong Sobel/Scharr emphasis (MRI) + PET preservation.

    Total:
      + a_m * (1-SSIM(Y,M)) + a_p * (1-SSIM(Y,P))
      + b_m * |Y - M| + b_p * |Y - P|
      + g_m * sum_scales Charb(∇Y - ∇M)         (multi-scale)
      + g_p * sum_scales Charb(∇Y - ∇P)         (weak, optional)
      + e_h * mean(ReLU(E_M - E_Y))             (edge hinge; punish missing edges only)
      + e_i * mean(E_M * |Y - M|)               (edge-weighted intensity to MRI)
      + p_s * mean(S(P) * |Y - P|)              (PET saliency-weighted)
      + tv  * TV(Y)
    """
    def __init__(self,
                 w_ssim_mri=2.5, w_ssim_pet=2.0,
                 w_l1_mri=0.5,  w_l1_pet=0.5,
                 w_grad_mri=3.0, w_grad_pet=0.5,
                 w_edge_hinge=2.0, w_edge_int=1.0,
                 w_pet_sal=1.5, w_tv=0.3,
                 edge_mode="sobel"):
        super().__init__()
        self.w_ssim_mri = w_ssim_mri
        self.w_ssim_pet = w_ssim_pet
        self.w_l1_mri   = w_l1_mri
        self.w_l1_pet   = w_l1_pet
        self.w_grad_mri = w_grad_mri
        self.w_grad_pet = w_grad_pet
        self.w_edge_hinge = w_edge_hinge
        self.w_edge_int   = w_edge_int
        self.w_pet_sal = w_pet_sal
        self.w_tv = w_tv
        self.grad = GradMag(mode=edge_mode)
        self.l1 = nn.L1Loss()

    def pet_saliency(self, pet):
        # min–max normalize per-sample and sigmoid-threshold ~0.4
        pet_n = (pet - pet.amin(dim=(2,3), keepdim=True)) / \
                (pet.amax(dim=(2,3), keepdim=True) - pet.amin(dim=(2,3), keepdim=True) + 1e-6)
        return torch.sigmoid(6.0 * (pet_n - 0.4))

    def multi_scale_grad(self, a):
        g1 = self.grad(a)
        a2 = F.avg_pool2d(a, 2, 2)
        g2 = self.grad(a2)
        a4 = F.avg_pool2d(a2, 2, 2)
        g4 = self.grad(a4)
        return (g1, g2, g4)

    def forward(self, y, mri, pet):
        # Core similarities
        ssim_m = 1.0 - ssim_fn(y, mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet, data_range=1.0)
        l1_m   = self.l1(y, mri)
        l1_p   = self.l1(y, pet)

        # Multi-scale gradient consistency
        gy1, gy2, gy4 = self.multi_scale_grad(y)
        gm1, gm2, gm4 = self.multi_scale_grad(mri)
        gp1, gp2, gp4 = self.multi_scale_grad(pet)

        g_cons_mri = (charbonnier(gy1 - gm1).mean() +
                      charbonnier(gy2 - gm2).mean() +
                      charbonnier(gy4 - gm4).mean())

        g_cons_pet = (charbonnier(gy1 - gp1).mean() +
                      charbonnier(gy2 - gp2).mean() +
                      charbonnier(gy4 - gp4).mean())

        # Edge hinge + edge-weighted intensity (MRI)
        E_m = gm1.detach()  # highest-res edge map of MRI, stop gradient
        E_y = gy1
        edge_hinge = torch.relu(E_m - E_y).mean()
        edge_int = (E_m * (y - mri).abs()).mean()

        # PET saliency (bright regions)
        S_pet = self.pet_saliency(pet)
        l1_pet_sal = (S_pet * (y - pet).abs()).mean()

        # TV
        tv = tv_loss(y)

        loss = (self.w_ssim_mri * ssim_m + self.w_ssim_pet * ssim_p +
                self.w_l1_mri * l1_m   + self.w_l1_pet * l1_p +
                self.w_grad_mri * g_cons_mri + self.w_grad_pet * g_cons_pet +
                self.w_edge_hinge * edge_hinge + self.w_edge_int * edge_int +
                self.w_pet_sal * l1_pet_sal + self.w_tv * tv)

        terms = dict(
            ssim_m=float(ssim_m.item()), ssim_p=float(ssim_p.item()),
            l1_m=float(l1_m.item()), l1_p=float(l1_p.item()),
            g_m=float(g_cons_mri.item()), g_p=float(g_cons_pet.item()),
            hinge=float(edge_hinge.item()), edge_int=float(edge_int.item()),
            pet_sal=float(l1_pet_sal.item()), tv=float(tv.item())
        )
        return loss, terms


# -----------------------------
# Training
# -----------------------------
def seed_everything(seed=1234):
    import numpy as np, random as pyrand
    pyrand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    import argparse
    p = argparse.ArgumentParser("Edge-dominant Sobel-guided MRI-PET fusion trainer")
    p.add_argument("--data_root", type=str, default="/data6/yasir/data/train")
    p.add_argument("--save_dir",  type=str, default="./checkpoints_edge_dominant")
    p.add_argument("--epochs",    type=int, default=20)
    p.add_argument("--batch_size",type=int, default=16)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--size",      type=int, default=128)
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--amp",       action="store_true")
    p.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Loss weights (tuned for stronger MRI edge fidelity + PET preservation)
    p.add_argument("--w_ssim_mri", type=float, default=2.0)
    p.add_argument("--w_ssim_pet", type=float, default=2.0)
    p.add_argument("--w_l1_mri",   type=float, default=0.5)
    p.add_argument("--w_l1_pet",   type=float, default=0.5)
    p.add_argument("--w_grad_mri", type=float, default=3.0)
    p.add_argument("--w_grad_pet", type=float, default=0.5)
    p.add_argument("--w_edge_hinge", type=float, default=2.0)
    p.add_argument("--w_edge_int",   type=float, default=1.0)
    p.add_argument("--w_pet_sal",    type=float, default=1.5)
    p.add_argument("--w_tv",         type=float, default=0.3)
    p.add_argument("--edge_mode",    type=str, choices=["sobel","scharr"], default="sobel")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(42)

    # Data
    train_set = PairFolder(args.data_root, size=args.size, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # Model + Loss
    model = TwoFusion_net(input_nc=2, output_nc=1).to(args.device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = EdgeDominantFusionLoss(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad_mri=args.w_grad_mri, w_grad_pet=args.w_grad_pet,
        w_edge_hinge=args.w_edge_hinge, w_edge_int=args.w_edge_int,
        w_pet_sal=args.w_pet_sal, w_tv=args.w_tv,
        edge_mode=args.edge_mode
    ).to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Train
    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        running = 0.0
        for x, mri, pet in pbar:
            x   = x.to(args.device)   # [B,2,H,W] (MRI at [:,0], PET at [:,1])
            mri = mri.to(args.device) # [B,1,H,W]
            pet = pet.to(args.device) # [B,1,H,W]

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                feats, edge_in = model.encoder(x)
                y, edge_pred = model.decoder((feats, edge_in))  # y in [0,1]
                loss, terms = loss_fn(y, mri, pet)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item(); global_step += 1
            pbar.set_postfix(loss=f"{running/(global_step or 1):.4f}",
                             gM=f"{terms['g_m']:.3f}",
                             hinge=f"{terms['hinge']:.3f}",
                             petS=f"{terms['pet_sal']:.3f}")

        # Save checkpoint
        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "opt": opt.state_dict(), "args": vars(args)}
        torch.save(ckpt, os.path.join(args.save_dir, f"edge_dom_e{epoch:03d}.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "edge_dom_final.pth"))
    print("Done. Saved to:", args.save_dir)


if __name__ == "__main__":
    main()
