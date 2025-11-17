import os, glob, random
from typing import Tuple, List

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
from torchvision.utils import save_image

try:
    from pytorch_msssim import ssim as ssim_fn  # pip install pytorch-msssim
except Exception as e:
    raise RuntimeError("Please install pytorch-msssim: pip install pytorch-msssim") from e

from model import Fusion_net


# ----------------------------
# Dataset: MRI/PET/MASK
# ----------------------------
class PairFolder(Dataset):
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

        # if self.augment:
        #     if random.random() < 0.5:
        #         mri = self.rand_hflip(mri); pet = self.rand_hflip(pet); mask = self.rand_hflip(mask)
        #     if random.random() < 0.5:
        #         mri = self.rand_vflip(mri); pet = self.rand_vflip(pet); mask = self.rand_vflip(mask)

        mri_t = self.to_tensor(mri)   # [1,H,W]
        pet_t = self.to_tensor(pet)   # [1,H,W]
        mask_t = self.to_tensor(mask) # [1,H,W] in [0,1]
        mask_t = (mask_t > self.bin_thresh).float()  # binarize robustly
        
        # # assume you already have tensors: mri_t, pet_t, mask_t  [1,H,W], values in [0,1]
        # rand_id = random.randint(1000, 9999)

        # # concatenate along width (dim=-1)
        # merged = torch.cat([mri_t, pet_t, mask_t], dim=-1)  # [1,H, W*3]

        # # make sure output dir exists
        # save_dir = "./saved_triplets"
        # os.makedirs(save_dir, exist_ok=True)

        # # build save path
        # save_path = os.path.join(save_dir, f"triplet_{rand_id}.png")

        # # save merged tensor as image
        # save_image(merged, save_path)

        x = torch.cat([mri_t, pet_t, mask_t], dim=0)  # [3,H,W]
        return x, mri_t, pet_t, mask_t


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

def masked_l1(a,b,m):
    w=m.float(); denom=w.sum().clamp_min(1.0)
    return ((w*(a-b).abs()).sum()/denom)

def sliced_w2_1d(a,b,m):
    av=a[m.bool()].flatten(); bv=b[m.bool()].flatten()
    if av.numel()<8 or bv.numel()<8: return torch.tensor(0.,device=a.device)
    av,_=torch.sort(av); bv,_=torch.sort(bv)
    n=min(av.numel(),bv.numel())
    return ((av[:n]-bv[:n])**2).mean()

class EdgeGuidedFusionLoss(nn.Module):
    """
    Total:
      λs_m * (1-SSIM(Y,MRI)) + λs_p * (1-SSIM(Y,PET))
    + λl_m * |Y - MRI| + λl_p * |Y - PET|
    + λg * sum_scales || ∇Y - ∇MRI ||_Charbonnier
    + λaux * |Ehat - ∇MRI|
    + λtv * TV(Y)
    + λin * OT(PET_in_mask, Y_in_mask)
    + λout * |Y_outside_mask|
    """
    def __init__(self,
                 w_ssim_mri=4.0, w_ssim_pet=1.0,
                 w_l1_mri=1.0,  w_l1_pet=0.25,
                 w_grad=2.0, w_aux=1.0, w_tv=0.5, w_pet_sal=1.0,
                 w_in=1, w_out=1, w_mass=1, w_sot=0.2):
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
        self.w_in = w_in
        self.w_out = w_out
        self.w_mass = w_mass
        self.w_sot = w_sot

    def multi_scale_grad(self, y, mri):
        def one_scale(a):
            return self.sobel(a)
        g_y1, g_m1 = one_scale(y), one_scale(mri)
        y2, m2 = F.avg_pool2d(y, 2, 2), F.avg_pool2d(mri, 2, 2)
        g_y2, g_m2 = one_scale(y2), one_scale(m2)
        return charbonnier(g_y1 - g_m1).mean() + charbonnier(g_y2 - g_m2).mean()

    def ot_loss_surrogate(self, src, tgt):
        # Surrogate for Wasserstein: mean L1 over normalized histograms
        src_flat = src.view(src.size(0), -1)
        tgt_flat = tgt.view(tgt.size(0), -1)
        src_norm = src_flat / (src_flat.sum(dim=1, keepdim=True) + 1e-6)
        tgt_norm = tgt_flat / (tgt_flat.sum(dim=1, keepdim=True) + 1e-6)
        return F.l1_loss(src_norm, tgt_norm)

    def forward(self, y, mri, pet, mask, ehat):
        # ----------------
        # Classical terms
        # ----------------
        ssim_m = 1.0 - ssim_fn(y, mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet, data_range=1.0)
        l1_m = self.l1(y, mri)
        l1_p = self.l1(y, pet)

        pet_n = (pet - pet.amin(dim=(2,3), keepdim=True)) / \
                (pet.amax(dim=(2,3), keepdim=True) - pet.amin(dim=(2,3), keepdim=True) + 1e-6)
        S_pet = torch.sigmoid(6.0 * (pet_n - 0.4))
        l1_pet_sal = (S_pet * (y - pet).abs()).mean()

        g_cons = self.multi_scale_grad(y, mri)
        target_e = self.sobel(mri).detach()
        aux = self.l1(ehat, target_e)
        tv = tv_loss(y)

        # ----------------
        # Cortex-guided OT terms
        # ----------------
        y_in, pet_in = y * mask, pet * mask
        y_out = y * (1 - mask)

        # OT surrogate inside cortex
        ot_in = self.ot_loss_surrogate(pet_in, y_in)

        # Penalize fused activity outside cortex
        out_penalty = y_out.abs().mean()

        # Mass conservation: total PET vs fused inside mask
        mass_loss = (pet_in.sum(dim=(1,2,3)) - y_in.sum(dim=(1,2,3))).abs().mean()

        # ----------------
        # Final loss
        # ----------------
        loss = (self.w_ssim_mri * ssim_m + self.w_ssim_pet * ssim_p +
                self.w_l1_mri * l1_m + self.w_l1_pet * l1_p +
                self.w_grad * g_cons + self.w_aux * aux +
                self.w_tv * tv + self.w_pet_sal * l1_pet_sal )#+
                # self.w_in * ot_in + self.w_out * out_penalty +
                # self.w_mass * mass_loss)

        terms = {
            "ssim_m": ssim_m.item(), "ssim_p": ssim_p.item(),
            "l1_m": l1_m.item(), "l1_p": l1_p.item(),
            "g_cons": g_cons.item(), "aux": aux.item(), "tv": tv.item(),
            # "ot_in": ot_in.item(), "out_penalty": out_penalty.item(), "mass_loss": mass_loss.item()
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

out_dir="./samples"
os.makedirs(out_dir, exist_ok=True)

def main():
    import argparse
    parser = argparse.ArgumentParser("Edge-guided MRI-PET fusion trainer")
    parser.add_argument("--data_root", type=str, default="/mnt/yasir/PET_Dataset_Processed/Separated_Centered_PET_MRI_Filtered_Dataset_Filtered_Splitted/train", help="contains mri/ and pet/")
    parser.add_argument("--save_dir",  type=str, default="./checkpoints_edgeguided")
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--size",      type=int, default=128)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--amp",       action="store_true", help="use mixed precision")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # weights
    parser.add_argument("--w_ssim_mri", type=float, default=2.5)
    parser.add_argument("--w_ssim_pet", type=float, default=2.0)
    parser.add_argument("--w_l1_mri",   type=float, default=1.0)
    parser.add_argument("--w_l1_pet",   type=float, default=1.0)
    parser.add_argument("--w_grad",     type=float, default=2.0)
    parser.add_argument("--w_aux",      type=float, default=1.0)
    parser.add_argument("--w_tv",       type=float, default=0.0)  #setting tv to 0, improved numerical results, but not ssim
    parser.add_argument("--w_pet_sal",       type=float, default=3.0)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(42)

    # Data
    train_set = PairFolder(args.data_root, size=args.size, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # Model
    model = Fusion_net(input_nc=3, output_nc=1).to(args.device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = EdgeGuidedFusionLoss(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad=args.w_grad, w_aux=args.w_aux, w_tv=args.w_tv
    ).to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Train
    counter = 0
    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        running = 0.0
        for x, mri, pet, mask in pbar:
            x   = x.to(args.device)   # [B,2,H,W]
            mri = mri.to(args.device) # [B,1,H,W]
            pet = pet.to(args.device) # [B,1,H,W]
            mask = mask.to(args.device) # [B,1,H,W]
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # Forward: encoder returns (feats, internal_edge); decoder returns (fused, aux_edge_pred)
                feats, edge_in = model.encode_with_edge(x)
                # print(f"Number of feature maps from encoder: {feats}")
                fused, edge_pred = model.decode_from_feats(feats)  # fused ∈ [0,1]

                loss, terms = loss_fn(fused, mri, pet, edge_pred, mask)

                if counter % 50 == 0:
                    with torch.no_grad():
                        mri1 = mri[:1]; pet1 = pet[:1]; mask1 = mask[:1]; fused1 = fused[:1]
                        panel = torch.cat([mri1, pet1, mask1, fused1], dim=-1)  # horizontal strip
                        save_image(panel, os.path.join(out_dir, f"panel_{epoch}_{random.randint(1000, 9999)}.png"))
                counter += 1
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
