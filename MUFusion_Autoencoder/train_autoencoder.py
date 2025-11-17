import os
import glob
import random
import math
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

# Reuse your MUFusion-style encoder/decoder (input_nc=2 => [IR, VIS] stacked)
from net import TwoFusion_net  # :contentReference[oaicite:2]{index=2}


# -----------------------------
# Dataset: pairs IR/VIS by name
# -----------------------------
class PairFolder(Dataset):
    """
    Expects:
      root/IR/*.png (or jpg)
      root/VIS/*.png (or jpg)
    Matching is by stem (filename without extension). Only files present in BOTH are kept.
    """
    def __init__(self, root: str, size: int = 128, exts=("*.png","*.jpg","*.jpeg"), augment=True):
        self.ir_dir  = os.path.join(root, "mri")
        self.vis_dir = os.path.join(root, "pet")
        ir_files  = []
        vis_files = []
        for p in exts:
            ir_files  += glob.glob(os.path.join(self.ir_dir,  p))
            vis_files += glob.glob(os.path.join(self.vis_dir, p))

        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        ir_map  = {stem(p): p for p in ir_files}
        vis_map = {stem(p): p for p in vis_files}
        common  = sorted(set(ir_map.keys()) & set(vis_map.keys()))
        self.pairs = [(ir_map[s], vis_map[s]) for s in common]
        if len(self.pairs) == 0:
            raise FileNotFoundError(f"No matching IR/VIS pairs found under {root}/IR and {root}/VIS")

        self.size = size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()  # scales to [0,1]
        self.rand_hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.rand_vflip = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self):
        return len(self.pairs)

    def _load_gray(self, path: str) -> Image.Image:
        # force grayscale, consistent with your original training code
        img = Image.open(path).convert("L")
        return img

    def _rand_crop_pair(self, ir: Image.Image, vis: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # random resized crop to size×size (keeps variability, simple & robust)
        W, H = ir.size
        size = self.size
        if W < size or H < size:
            # upsample the shorter side before cropping to avoid tiny images
            scale = max(size / W, size / H)
            newW, newH = int(round(W * scale)), int(round(H * scale))
            ir  = ir.resize((newW, newH), Image.BICUBIC)
            vis = vis.resize((newW, newH), Image.BICUBIC)
            W, H = newW, newH
        # pick same crop box for both to keep alignment
        x = random.randint(0, W - size)
        y = random.randint(0, H - size)
        box = (x, y, x + size, y + size)
        return ir.crop(box), vis.crop(box)

    def __getitem__(self, idx: int):
        ir_path, vis_path = self.pairs[idx]
        ir  = self._load_gray(ir_path)
        vis = self._load_gray(vis_path)

        # aligned random crop
        ir, vis = self._rand_crop_pair(ir, vis)

        # light augmentation (flips only), identical for both
        if self.augment:
            if random.random() < 0.5:
                ir  = self.rand_hflip(ir)
                vis = self.rand_hflip(vis)
            if random.random() < 0.5:
                ir  = self.rand_vflip(ir)
                vis = self.rand_vflip(vis)

        ir_t  = self.to_tensor(ir)   # [1,H,W], 0..1
        vis_t = self.to_tensor(vis)  # [1,H,W], 0..1

        # network expects 2 channels concatenated
        x = torch.cat([ir_t, vis_t], dim=0)  # [2,H,W]
        return x, ir_t, vis_t


# -----------------------------
# Simple Loss: L1 + SSIM
# -----------------------------
class FusionLoss(nn.Module):
    """
    Total loss = α * (1 - SSIM(out, IR)) + α * (1 - SSIM(out, VIS)) + β * (L1(out, IR) + L1(out, VIS))
    Using symmetric supervision to both sources encourages the fused output
    to preserve structure and intensity from each input without feature nets.
    """
    def __init__(self, alpha=5.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, out: torch.Tensor, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        # ssim() expects [N,1,H,W] in [0,1]
        ssim_ir  = ssim_fn(out, ir, data_range=1.0)
        ssim_vis = ssim_fn(out, vis, data_range=1.0)
        l1_ir  = self.l1(out, ir)
        l1_vis = self.l1(out, vis)
        loss = self.alpha * ((1.0 - ssim_ir) + (1.0 - ssim_vis)) + self.beta * (l1_ir + l1_vis)
        return loss


# -----------------------------
# Training
# -----------------------------
def seed_everything(seed=1234):
    import numpy as np, random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    import argparse
    parser = argparse.ArgumentParser("Autoencoder-style fusion trainer (simple losses)")
    parser.add_argument("--data_root", type=str, required=False, default='/data6/yasir/data/train', help="Folder containing IR/ and VIS/")
    parser.add_argument("--save_dir",  type=str, default="./checkpoints_simple")
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--size",      type=int, default=128, help="crop size")
    parser.add_argument("--alpha",     type=float, default=5.0, help="weight for SSIM term")
    parser.add_argument("--beta",      type=float, default=1.0, help="weight for L1 term")
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--amp",       action="store_true", help="use mixed precision")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(42)

    # Data
    train_set = PairFolder(args.data_root, size=args.size, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # Model (same topology as your MUFusion net)
    model = TwoFusion_net(input_nc=2, output_nc=1).to(args.device)  # :contentReference[oaicite:3]{index=3}
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = FusionLoss(alpha=args.alpha, beta=args.beta)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Train
    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        running = 0.0
        for x, ir, vis in pbar:
            x   = x.to(args.device)      # [B,2,H,W]
            ir  = ir.to(args.device)     # [B,1,H,W]
            vis = vis.to(args.device)    # [B,1,H,W]
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # Encoder/decoder forward
                feats = model.encoder(x)           # list of feature maps
                out   = model.decoder(feats)[0]    # fused image, [B,1,H,W]
                out   = torch.clamp(out, 0.0, 1.0) # keep in valid image range
                loss  = loss_fn(out, ir, vis)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{running / (global_step or 1):.4f}")

        # Save checkpoint each epoch
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        # torch.save(ckpt, os.path.join(args.save_dir, f"fusion_autoencoder_e{epoch:03d}.pt"))

    # Final
    torch.save(model.state_dict(), os.path.join(args.save_dir, "fusion_autoencoder_final.pth"))
    print("Training complete. Checkpoints saved to:", args.save_dir)


if __name__ == "__main__":
    main()
