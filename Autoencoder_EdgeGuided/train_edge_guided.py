import os, glob, random
from typing import Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

from net import TwoFusion_net


# -----------------------------
# Dataset: MRI/PET pairs
# -----------------------------
class PairFolder(Dataset):
    """
    Expects:
      root/mri/*.png (or jpg)
      root/pet/*.png (or jpg)
    Matching is by filename stem.
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
            raise FileNotFoundError(f"No matching MRI/PET pairs found under {self.mri_dir} and {self.pet_dir}")

        self.size = size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()  # [0,1]
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

        mri_t = self.to_tensor(mri)  # [1,H,W]
        pet_t = self.to_tensor(pet)  # [1,H,W]
        x = torch.cat([mri_t, pet_t], dim=0)  # [2,H,W]
        return x, mri_t, pet_t


# -----------------------------
# Losses: MS-SSIM/L1 + gradient + aux + TV
# -----------------------------
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

    # def forward(self, x_1ch: torch.Tensor) -> torch.Tensor:
    #     gx = F.conv2d(x_1ch, self.kx, padding=1)
    #     gy = F.conv2d(x_1ch, self.ky, padding=1)
    #     mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    #     amax = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    #     return (mag / amax).clamp(0.0, 1.0)
    
    def forward(self, x_1ch: torch.Tensor) -> torch.Tensor:
        # ensure buffers match the incoming tensor’s device & dtype
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


class EdgeGuidedFusionLoss(nn.Module):
    """
    Total:
      λs_m * (1-SSIM(Y,MRI)) + λs_p * (1-SSIM(Y,PET))
    + λl_m * |Y - MRI| + λl_p * |Y - PET|
    + λg * sum_scales || ∇Y - ∇MRI ||_Charbonnier
    + λaux * |Ehat - ∇MRI|
    + λtv * TV(Y)
    """
    def __init__(self,
                 w_ssim_mri=4.0, w_ssim_pet=1.0,
                 w_l1_mri=1.0,  w_l1_pet=0.25,
                 w_grad=2.0, w_aux=1.0, w_tv=0.5, w_pet_sal=1.0):
        super().__init__()
        self.w_ssim_mri = w_ssim_mri
        self.w_ssim_pet = w_ssim_pet
        self.w_l1_mri = w_l1_mri
        self.w_l1_pet = w_l1_pet
        self.w_grad = w_grad
        self.w_aux = w_aux
        self.w_tv = w_tv
        self.l1 = nn.L1Loss()
        self.sobel = SobelGrad()
        self.w_pet_sal = w_pet_sal
        
    def multi_scale_grad(self, y, mri):
        def one_scale(a):
            return self.sobel(a)
        # scale 1
        g_y1 = one_scale(y); g_m1 = one_scale(mri)
        # scale 1/2
        y2, m2 = F.avg_pool2d(y, 2, 2), F.avg_pool2d(mri, 2, 2)
        g_y2 = one_scale(y2); g_m2 = one_scale(m2)
        # Charbonnier distance
        return charbonnier(g_y1 - g_m1).mean() + charbonnier(g_y2 - g_m2).mean()

    def forward(self, y, mri, pet, ehat):
        # MS-SSIM terms
        ssim_m = 1.0 - ssim_fn(y, mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet, data_range=1.0)

        # L1 to both sources (MRI stronger)
        l1_m = self.l1(y, mri)
        l1_p = self.l1(y, pet)

        # PET saliency (soft mask): emphasize bright/upright PET regions
        pet_n = (pet - pet.amin(dim=(2,3), keepdim=True)) / \
                (pet.amax(dim=(2,3), keepdim=True) - pet.amin(dim=(2,3), keepdim=True) + 1e-6)
        S_pet = torch.sigmoid(6.0 * (pet_n - 0.4))  # threshold ~0.4; 6.0 sharpness
        l1_pet_sal = (S_pet * (y - pet).abs()).mean()

        # gradient consistency to MRI (multi-scale)
        g_cons = self.multi_scale_grad(y, mri)

        # auxiliary edge supervision (predict MRI Sobel)
        target_e = self.sobel(mri).detach()
        aux = self.l1(ehat, target_e)

        # TV
        tv = tv_loss(y)

        loss = (self.w_ssim_mri * ssim_m + self.w_ssim_pet * ssim_p +
                self.w_l1_mri * l1_m + self.w_l1_pet * l1_p +
                self.w_grad * g_cons + self.w_aux * aux + self.w_tv * tv+ self.w_pet_sal * l1_pet_sal)

        # pack some monitors
        terms = {
            "ssim_m": ssim_m.item(), "ssim_p": ssim_p.item(),
            "l1_m": l1_m.item(), "l1_p": l1_p.item(),
            "g_cons": g_cons.item(), "aux": aux.item(), "tv": tv.item()
        }
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
    parser = argparse.ArgumentParser("Edge-guided MRI-PET fusion trainer")
    parser.add_argument("--data_root", type=str, default="/data6/yasir/data/train", help="contains mri/ and pet/")
    parser.add_argument("--save_dir",  type=str, default="./checkpoints_edgeguided")
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--size",      type=int, default=128)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--amp",       action="store_true", help="use mixed precision")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # weights
    parser.add_argument("--w_ssim_mri", type=float, default=4.0)
    parser.add_argument("--w_ssim_pet", type=float, default=1.0)
    parser.add_argument("--w_l1_mri",   type=float, default=1.0)
    parser.add_argument("--w_l1_pet",   type=float, default=0.25)
    parser.add_argument("--w_grad",     type=float, default=2.0)
    parser.add_argument("--w_aux",      type=float, default=1.0)
    parser.add_argument("--w_tv",       type=float, default=0.5)
    parser.add_argument("--w_pet_sal",       type=float, default=1)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(42)

    # Data
    train_set = PairFolder(args.data_root, size=args.size, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # Model
    model = TwoFusion_net(input_nc=2, output_nc=1).to(args.device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = EdgeGuidedFusionLoss(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad=args.w_grad, w_aux=args.w_aux, w_tv=args.w_tv
    ).to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Train
    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        running = 0.0
        for x, mri, pet in pbar:
            x   = x.to(args.device)   # [B,2,H,W]
            mri = mri.to(args.device) # [B,1,H,W]
            pet = pet.to(args.device) # [B,1,H,W]

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # Forward: encoder returns (feats, internal_edge); decoder returns (fused, aux_edge_pred)
                feats, _edge_in = model.encoder(x)
                fused, edge_pred = model.decoder((feats, _edge_in))  # fused ∈ [0,1]

                loss, terms = loss_fn(fused, mri, pet, edge_pred)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{running / (global_step or 1):.4f}",
                             ssim_m=f"{terms['ssim_m']:.3f}",
                             grad=f"{terms['g_cons']:.3f}",
                             aux=f"{terms['aux']:.3f}")

        # Save checkpoint each epoch
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        # torch.save(ckpt, os.path.join(args.save_dir, f"edgeguided_e{epoch:03d}.pt"))

    # Final
    torch.save(model.state_dict(), os.path.join(args.save_dir, "edgeguided_final.pth"))
    print("Training complete. Checkpoints saved to:", args.save_dir)


if __name__ == "__main__":
    main()
