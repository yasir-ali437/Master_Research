import os, glob, random
from typing import Tuple, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy import ndimage as ndi
from geomloss import SamplesLoss
from piq import ssim
from scipy.ndimage import gaussian_filter
import wandb

try:
    from pytorch_msssim import ssim as ssim_fn
except Exception as e:
    raise RuntimeError("Please install pytorch-msssim: pip install pytorch-msssim") from e

from model import Fusion_net  # Ensure model outputs RGB fused image (3 channels)

# =====================================================
# Weights & Biases Setup
# =====================================================
run = wandb.init(
    entity="yasir517390-bilkent-university",
    project="PET-MRI Fusion RGB",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "|Custom| MRI-PET (RGB)",
        "epochs": 20,
    },
)

# =====================================================
# Dataset
# =====================================================
class PairFolder(Dataset):
    """Dataset with grayscale MRI, RGB PET, and GM cortex mask."""

    def __init__(self, root: str, size: int = 300, exts=("*.png", "*.jpg", "*.jpeg"),
                 augment=True, bin_thresh=0.5):
        self.mri_dir = os.path.join(root, "MRI")
        self.pet_dir = os.path.join(root, "PET_Deskulled_Colored")
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

        # print(f"Found {len(mri_map)} MRI, {len(pet_map)} PET, {len(mask_map)} GM mask files.")
        common = sorted(set(mri_map.keys()) & set(pet_map.keys()) & set(mask_map.keys()))
        if not common:
            raise FileNotFoundError(f"No matching triplets in {root}")

        self.triplets = [(mri_map[s], pet_map[s], mask_map[s]) for s in common]
        self.size = size
        self.augment = augment
        self.bin_thresh = bin_thresh

        self.to_tensor = transforms.ToTensor()
        self.rand_hflip = transforms.RandomHorizontalFlip(p=1.0)
        self.rand_vflip = transforms.RandomVerticalFlip(p=1.0)

    def __len__(self):
        return len(self.triplets)

    def _load_gray(self, path): return Image.open(path).convert("L")
    def _load_rgb(self, path): return Image.open(path).convert("RGB")

    def _ensure_min_size(self, imgs):
        W, H = imgs[0].size
        size = self.size
        if W < size or H < size:
            scale = max(size / W, size / H)
            newW, newH = int(W * scale), int(H * scale)
            imgs = [im.resize((newW, newH), Image.BICUBIC) for im in imgs]
        return imgs

    def _rand_crop_triplet(self, a, b, c):
        W, H = a.size
        size = self.size
        x = random.randint(0, W - size)
        y = random.randint(0, H - size)
        box = (x, y, x + size, y + size)
        return a.crop(box), b.crop(box), c.crop(box)

    def __getitem__(self, idx):
        mri_path, pet_path, mask_path = self.triplets[idx]
        mri, pet, mask = self._load_gray(mri_path), self._load_rgb(pet_path), self._load_gray(mask_path)
        mri, pet, mask = self._ensure_min_size([mri, pet, mask])
        mri, pet, mask = self._rand_crop_triplet(mri, pet, mask)

        if self.augment:
            if random.random() < 0.5:
                mri = self.rand_hflip(mri); pet = self.rand_hflip(pet); mask = self.rand_hflip(mask)
            if random.random() < 0.5:
                mri = self.rand_vflip(mri); pet = self.rand_vflip(pet); mask = self.rand_vflip(mask)

        mri_t = self.to_tensor(mri)        # [1,H,W]
        pet_t = self.to_tensor(pet)        # [3,H,W]
        mask_t = (self.to_tensor(mask) > self.bin_thresh).float()  # [1,H,W]

        # Concatenate into a single tensor [1+3+1=5 channels]
        x = torch.cat([mri_t, pet_t, mask_t], dim=0)
        return x, mri_t, pet_t, mask_t


# =====================================================
# Sobel Gradient (channel-wise)
# =====================================================
class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1., 2., 1.],[0., 0., 0.],[-1., -2., -1.]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        # Handle RGB input
        if x.shape[1] > 1:
            grads = []
            for c in range(x.shape[1]):
                gx = F.conv2d(x[:,c:c+1], self.kx, padding=1)
                gy = F.conv2d(x[:,c:c+1], self.ky, padding=1)
                mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
                mag = mag / (mag.amax(dim=(2,3), keepdim=True).clamp_min(1e-6))
                grads.append(mag)
            return torch.stack(grads, dim=1).mean(1, keepdim=True)  # avg over channels
        else:
            gx = F.conv2d(x, self.kx, padding=1)
            gy = F.conv2d(x, self.ky, padding=1)
            mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
            mag = mag / (mag.amax(dim=(2,3), keepdim=True).clamp_min(1e-6))
            return mag


def charbonnier(x, eps=1e-3): return torch.sqrt(x*x + eps*eps)

def tv_loss(img):
    dy = img[:,:,1:,:] - img[:,:,:-1,:]
    dx = img[:,:,:,1:] - img[:,:,:,:-1]
    return (dy.abs().mean() + dx.abs().mean())


# =====================================================
# Cortex Transport and Loss Components (RGB compatible)
# =====================================================
def intensity_preservation_loss(y, pet, mask, alpha=1.0):
    """Preserve PET intensities inside the cortex."""
    pet_in = pet * mask
    y_in = y * mask
    return alpha * F.mse_loss(y_in, pet_in)

def gaussian_smooth(x, sigma=1.0):
    x_np = x.detach().cpu().numpy()
    smoothed = np.zeros_like(x_np)
    for b in range(x_np.shape[0]):
        for c in range(x_np.shape[1]):
            smoothed[b, c] = gaussian_filter(x_np[b, c], sigma=sigma)
    return torch.tensor(smoothed).to(x.device)


# =====================================================
# Full Loss
# =====================================================
class EdgeGuidedFusionLoss(nn.Module):
    def __init__(self, **weights):
        super().__init__()
        for k,v in weights.items(): setattr(self,k,v)
        self.l1 = nn.L1Loss()
        self.sobel = SobelGrad()
        self.sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)
        self.alpha, self.beta, self.gamma = 0.8, 0.3, 0.5

    def multi_scale_grad(self, y, mri):
        g_y1, g_m1 = self.sobel(y.mean(1,keepdim=True)), self.sobel(mri)
        y2, m2 = F.avg_pool2d(y,2,2), F.avg_pool2d(mri,2,2)
        g_y2, g_m2 = self.sobel(y2.mean(1,keepdim=True)), self.sobel(m2)
        return charbonnier(g_y1-g_m1).mean() + charbonnier(g_y2-g_m2).mean()

    def forward(self, y, mri, pet, mask, ehat):
        ssim_m = 1.0 - ssim_fn(y.mean(1,keepdim=True), mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet, data_range=1.0)
        l1_m, l1_p = self.l1(y.mean(1,keepdim=True), mri), self.l1(y, pet)

        g_cons = self.multi_scale_grad(y, mri)
        target_e = self.sobel(mri).detach()
        aux = self.l1(ehat, target_e)
        tv  = tv_loss(y)

        # Transport + outside penalty
        B = pet.size(0)
        L_transport, L_out, L_ssim = 0.0, 0.0, 0.0

        for i in range(B):
            pet_i = (pet[i] * mask[i]).view(3, -1).T
            y_i = (y[i] * mask[i]).view(3, -1).T
            valid = mask[i].flatten() > 0.1
            if valid.sum() < 32: continue
            if pet_i.shape[0] > 4096:
                idx = torch.randperm(pet_i.shape[0])[:4096]
                pet_i, y_i = pet_i[idx], y_i[idx]
            L_transport += self.sinkhorn(y_i, pet_i)
            L_out += torch.mean((y[i] * (1 - mask[i])) ** 2)
             # Ensure same channels before SSIM
            mri_i_rgb = mri[i:i+1].repeat(1, 3, 1, 1)
            
            L_ssim += 1 - ssim(y[i:i+1]*mask[i:i+1], mri_i_rgb*mask[i:i+1], data_range=1.0)

        L_transport /= B; L_out /= B; L_ssim /= B
        loss_transport = self.alpha*L_transport + self.beta*L_out + self.gamma*L_ssim

        L_intensity = intensity_preservation_loss(y, pet, mask, alpha=2.5)
        smoothed_fused = gaussian_smooth(y)
        L_smooth = F.mse_loss(smoothed_fused, y)

        total_loss = (
            # self.w_ssim_mri * ssim_m +
            loss_transport + L_intensity + 0.2 * L_smooth)
        

        run.log({"loss": total_loss})
        return total_loss, {
            "ssim_m": ssim_m.mean().item(),
            "ssim_p": ssim_p.mean().item(),
            "grad": g_cons.mean().item(),
            "aux": aux.mean().item()
        }


# =====================================================
# Training
# =====================================================
def seed_everything(seed=1234):
    import numpy as np, random as pyrand
    pyrand.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    import argparse
    parser = argparse.ArgumentParser("MRI-PET fusion (RGB PET, grayscale MRI)")
    parser.add_argument("--data_root", type=str, default="/mnt/yasir/PET_Dataset_Processed/Separated_Centered_PET_MRI_Updated_Dataset_Splitted/train")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_rgbfusion")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--w_ssim_mri", type=float, default=0.01)
    parser.add_argument("--w_ssim_pet", type=float, default=2.5)
    parser.add_argument("--w_l1_mri", type=float, default=1.0)
    parser.add_argument("--w_l1_pet", type=float, default=1.0)
    parser.add_argument("--w_grad", type=float, default=2.0)
    parser.add_argument("--w_aux", type=float, default=1.0)
    parser.add_argument("--w_tv", type=float, default=0.0)
    parser.add_argument("--w_pet_sal", type=float, default=3.0)
    parser.add_argument("--w_outside", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(42)

    train_set = PairFolder(args.data_root, size=args.size, augment=True)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = Fusion_net().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = EdgeGuidedFusionLoss(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad=args.w_grad, w_aux=args.w_aux,
        w_tv=args.w_tv, w_pet_sal=args.w_pet_sal,
        w_outside=args.w_outside
    ).to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    os.makedirs("./samples_rgb", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        for i, (x, mri, pet, mask) in enumerate(pbar):
            x, mri, pet, mask = x.to(args.device), mri.to(args.device), pet.to(args.device), mask.to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                fused, edge_pred = model(mri, pet, mask)
                loss, terms = loss_fn(fused, mri, pet, mask, edge_pred)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            if i % 50 == 0:
                with torch.no_grad():
                    panel = torch.cat([mri[:1].repeat(1,3,1,1), pet[:1], mask[:1].repeat(1,3,1,1), fused[:1]], dim=-1)
                    save_image(panel, f"./samples_rgb/panel_e{epoch}_b{i}.png")

            pbar.set_postfix(loss=f"{running/(i+1):.4f}", ssim_m=f"{terms['ssim_m']:.3f}", grad=f"{terms['grad']:.3f}")

        scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"fusion_epoch{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "fusion_final_rgb.pth"))
    print("Training complete. Checkpoints saved to:", args.save_dir)


if __name__ == "__main__":
    main()
