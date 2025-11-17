# train.py
import os, glob, random
from typing import Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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

from model import Fusion_net

# -----------------------------
# Dataset: MRI/PET pairs
# -----------------------------
class PairFolder(Dataset):
    """
    root/mri/*.png|jpg
    root/pet/*.png|jpg
    matched by filename stem
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
        self.pairs = [(mri_map[s], pet_map[s]) for s in common]
        if len(self.pairs) == 0:
            raise FileNotFoundError(f"No matching pairs under {self.mri_dir} and {self.pet_dir}")

        self.size = size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.rand_hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.rand_vflip = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self):
        return len(self.pairs)

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
# Differentiable Sobel (mag + vector)
# -----------------------------
class SobelMag(nn.Module):
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
        gx = F.conv2d(x_1ch, self.kx, padding=1)
        gy = F.conv2d(x_1ch, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        amax = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (mag / amax).clamp(0.0, 1.0)

class SobelVec(nn.Module):
    """Returns (gx, gy) without per-image normalization; use eps when normalizing."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1., 2., 1.],
                           [0., 0., 0.]],
                          dtype=torch.float32).view(1, 1, 3, 2)  # dummy init to avoid lint
        ky = torch.tensor([[1., 2., 1.],
                           [0., 0., 0.],
                           [-1., -2., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x_1ch: torch.Tensor):
        gx = F.conv2d(x_1ch, self.kx, padding=1)
        gy = F.conv2d(x_1ch, self.ky, padding=1)
        return gx, gy

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def tv_loss(img):
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (dy.abs().mean() + dx.abs().mean())

def weighted_l1(a, b, w=None, eps=1e-6):
    if w is None:
        return (a - b).abs().mean()
    num = (w * (a - b).abs()).sum()
    den = w.sum().clamp_min(eps)
    return num / den

# -----------------------------
# Composite fusion loss (ROI-weighted + new edge terms)
# -----------------------------
class EdgeGuidedFusionLoss(nn.Module):
    """
    Total:
      λs_m * (1-SSIM(Y,MRI)) + λs_p * (1-SSIM(Y,PET))
    + λl_m * |Y - MRI|_ROI + λl_p * |Y - PET|_ROI
    + λg * multi-scale grad consistency to MRI (Charbonnier)
    + λepi * orientation (1 - cosine) on edges
    + λfloor * ReLU(β*|∇MRI| - |∇Y|) on edges   <-- lifts edge magnitude
    + λaux * |Ehat - ∇MRI|
    + λtv * TV(Y)
    + λpet_sal * |Y - PET| on PET-salient regions
    """
    def __init__(self,
                 w_ssim_mri=3.0, w_ssim_pet=2.0,
                 w_l1_mri=1.0,  w_l1_pet=0.5,
                 w_grad=2.0, w_epi=1.0, w_floor=0.5,
                 edge_beta=0.8, edge_tau=0.25,
                 w_aux=1.0, w_tv=0.5, w_pet_sal=1.0):
        super().__init__()
        self.w_ssim_mri = w_ssim_mri
        self.w_ssim_pet = w_ssim_pet
        self.w_l1_mri = w_l1_mri
        self.w_l1_pet = w_l1_pet
        self.w_grad = w_grad
        self.w_epi = w_epi
        self.w_floor = w_floor
        self.edge_beta = edge_beta
        self.edge_tau = edge_tau
        self.w_aux = w_aux
        self.w_tv = w_tv
        self.w_pet_sal = w_pet_sal

        self.sobel_mag = SobelMag()
        self.sobel_vec = SobelVec()
        self.l1 = nn.L1Loss()

    def multi_scale_grad(self, y, mri, roi=None):
        def mag(a): return self.sobel_mag(a)
        g_y1, g_m1 = mag(y), mag(mri)
        y2, m2 = F.avg_pool2d(y, 2, 2), F.avg_pool2d(mri, 2, 2)
        g_y2, g_m2 = mag(y2), mag(m2)
        d1 = charbonnier(g_y1 - g_m1)
        d2 = charbonnier(g_y2 - g_m2)
        if roi is not None:
            d1 = (d1 * roi).sum() / roi.sum().clamp_min(1e-6)
            roi2 = F.avg_pool2d(roi, 2, 2)
            d2 = (d2 * roi2).sum() / roi2.sum().clamp_min(1e-6)
            return d1 + d2
        return d1.mean() + d2.mean()

    def orientation_loss(self, y, mri, roi=None):
        gyx, gyy = self.sobel_vec(y)
        gmx, gmy = self.sobel_vec(mri)
        dot = gyx * gmx + gyy * gmy
        ny = torch.sqrt(gyx**2 + gyy**2 + 1e-6)
        nm = torch.sqrt(gmx**2 + gmy**2 + 1e-6)
        cos = dot / (ny * nm)
        # weight near true MRI edges
        w = torch.sigmoid(10.0 * (nm - self.edge_tau))
        if roi is not None:
            w = w * roi
        return (w * (1.0 - cos)).sum() / w.sum().clamp_min(1e-6)

    def edge_floor_loss(self, y, mri, roi=None):
        gy = self.sobel_mag(y)
        gm = self.sobel_mag(mri)
        # focus where MRI has edges
        edge_w = torch.sigmoid(10.0 * (gm - self.edge_tau))
        pen = F.relu(self.edge_beta * gm - gy)  # drive gy >= beta * gm
        w = edge_w if roi is None else edge_w * roi
        return (w * pen).sum() / w.sum().clamp_min(1e-6)

    def forward(self, y, mri, pet, ehat, roi):
        # SSIM terms (unweighted; they already average over spatial dims)
        ssim_m = 1.0 - ssim_fn(y, mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet, data_range=1.0)

        # ROI-weighted L1 to both sources
        l1_m = weighted_l1(y, mri, roi)
        l1_p = weighted_l1(y, pet, roi)

        # PET saliency (inside ROI)
        pet_n = (pet - pet.amin(dim=(2,3), keepdim=True)) / \
                (pet.amax(dim=(2,3), keepdim=True) - pet.amin(dim=(2,3), keepdim=True) + 1e-6)
        S_pet = torch.sigmoid(6.0 * (pet_n - 0.4))
        l1_pet_sal = weighted_l1(y, pet, roi * S_pet)

        # gradient consistency + new edge terms
        g_cons = self.multi_scale_grad(y, mri, roi)
        epi = self.orientation_loss(y, mri, roi)
        floor = self.edge_floor_loss(y, mri, roi)

        # auxiliary edge supervision (predict MRI Sobel)
        target_e = self.sobel_mag(mri).detach()
        aux = weighted_l1(ehat, target_e, roi)

        tv = tv_loss(y)

        loss = (self.w_ssim_mri * ssim_m + self.w_ssim_pet * ssim_p +
                self.w_l1_mri * l1_m + self.w_l1_pet * l1_p +
                self.w_pet_sal * l1_pet_sal +
                self.w_grad * g_cons + self.w_epi * epi + self.w_floor * floor +
                self.w_aux * aux + self.w_tv * tv)

        terms = {
            "ssim_m": float(ssim_m.item()),
            "ssim_p": float(ssim_p.item()),
            "l1_m": float(l1_m.item()),
            "l1_p": float(l1_p.item()),
            "g_cons": float(g_cons.item()),
            "epi": float(epi.item()),
            "floor": float(floor.item()),
            "aux": float(aux.item()),
            "tv": float(tv.item())
        }
        return loss, terms

# -----------------------------
# Training
# -----------------------------
def seed_everything(seed=1234):
    import numpy as np, random as pyrand
    pyrand.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    import argparse
    parser = argparse.ArgumentParser("Edge-guided MRI-PET fusion trainer (ROI + edge boosting)")
    parser.add_argument("--data_root", type=str, default="/data6/yasir/data/train")
    parser.add_argument("--save_dir",  type=str, default="./checkpoints_edgeguided")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--size",      type=int, default=128)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--amp",       action="store_true")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # weights
    parser.add_argument("--w_ssim_mri", type=float, default=2.0)
    parser.add_argument("--w_ssim_pet", type=float, default=3.0)
    parser.add_argument("--w_l1_mri",   type=float, default=1.0)
    parser.add_argument("--w_l1_pet",   type=float, default=1.0)
    parser.add_argument("--w_pet_sal",  type=float, default=1.0)
    parser.add_argument("--w_grad",     type=float, default=2.0)
    parser.add_argument("--w_epi",      type=float, default=1.0)
    parser.add_argument("--w_floor",    type=float, default=0.5)
    parser.add_argument("--w_aux",      type=float, default=1.0)
    parser.add_argument("--w_tv",       type=float, default=0.5)

    # edge-floor controls
    parser.add_argument("--edge_beta",  type=float, default=0.1)
    parser.add_argument("--edge_tau",   type=float, default=0.05)

    # ROI / unsharp
    parser.add_argument("--roi_t",      type=float, default=0.00)
    parser.add_argument("--roi_k",      type=float, default=0.0)
    parser.add_argument("--unsharp_max_gain", type=float, default=0.0)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(42)

    # Data
    train_set = PairFolder(args.data_root, size=args.size, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # Model
    model = Fusion_net(input_nc=2, output_nc=1,
                       roi_t=args.roi_t, roi_k=args.roi_k,
                       unsharp_max_gain=args.unsharp_max_gain).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay)

    loss_fn = EdgeGuidedFusionLoss(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad=args.w_grad, w_epi=args.w_epi, w_floor=args.w_floor,
        edge_beta=args.edge_beta, edge_tau=args.edge_tau,
        w_aux=args.w_aux, w_tv=args.w_tv, w_pet_sal=args.w_pet_sal
    ).to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Train
    global_step = 0
    model.train()
    best_ssim = -1.0
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=130)
        running = 0.0
        for x, mri, pet in pbar:
            x   = x.to(args.device)   # [B,2,H,W]
            mri = mri.to(args.device) # [B,1,H,W]
            pet = pet.to(args.device) # [B,1,H,W]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                fused, edge_pred, roi, gain = model(x)  # fused in [0,1]
                loss, terms = loss_fn(fused, mri, pet, edge_pred, roi)

            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()

            running += loss.item()
            global_step += 1

            # quick Tenengrad proxy (mean Sobel^2 inside ROI)
            with torch.no_grad():
                sob = SobelMag().to(args.device)
                g = sob(fused)
                ten = ((roi * g * g).sum() / roi.sum().clamp_min(1e-6)).item()

            pbar.set_postfix(
                loss=f"{running / (global_step or 1):.4f}",
                ssim_m=f"{terms['ssim_m']:.3f}",
                epi=f"{terms['epi']:.3f}",
                floor=f"{terms['floor']:.3f}",
                gain=f"{float(gain):.2f}",
                ten=f"{ten:.3f}"
            )

        # Save checkpoint each epoch (keep best by SSIM vs MRI)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"edgeguided_e{epoch:03d}.pt"))

        if terms["ssim_m"] > best_ssim:
            best_ssim = terms["ssim_m"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "edgeguided_best.pth"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "edgeguided_final.pth"))
    print("Training complete. Checkpoints saved to:", args.save_dir)

if __name__ == "__main__":
    main()
