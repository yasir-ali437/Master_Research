import os, glob, random
from typing import Tuple, List

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
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
      root/MRI/*.png|jpg
      root/PET/*.png|jpg
      root/GM/*.png|jpg   (binary cortex masks; same stem)
    """
    def __init__(self, root: str, size: int = 128, exts=("*.png","*.jpg","*.jpeg"), augment=True, bin_thresh=0.5):
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

    def _load_gray(self, path: str, mode: str) -> Image.Image:
        return Image.open(path).convert(mode)

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
        mri = self._load_gray(mri_path, "L")
        pet = self._load_gray(pet_path , "RGB")
        mask = self._load_gray(mask_path, "L")

        # same size + min size
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
        mask_t = (mask_t > self.bin_thresh).float()  # binarize
        
        x = torch.cat([mri_t, pet_t], dim=0)  # [2,H,W]
        return x, mri_t, pet_t, mask_t


# -----------------------------
# Losses: MS-SSIM/L1 + gradient + aux + TV + (ONE) outside-cortex penalty
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

def rgb_to_grayscale_batch(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of RGB image tensors to grayscale.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, 3, H, W] (RGB images).

    Returns:
        torch.Tensor: Grayscale image tensor of shape [B, 1, H, W].
    """
    # Ensure the input tensor has 3 channels (RGB)
    if tensor.shape[1] != 3:
        raise ValueError("Input tensor must have 3 channels (RGB).")

    # Define the weights for RGB to grayscale conversion
    r_weight = 0.2989
    g_weight = 0.5870
    b_weight = 0.1140

    # Apply the weighted sum to convert to grayscale for each image in the batch
    grayscale_tensor = (r_weight * tensor[:, 0, :, :] +
                        g_weight * tensor[:, 1, :, :] +
                        b_weight * tensor[:, 2, :, :]).unsqueeze(1)  # Keep batch dimension and add a single channel

    return grayscale_tensor

class EdgeGuidedFusionLoss(nn.Module):
    """
    Adds a single cortex constraint:
      L_outside = mean( ReLU(y) * (1 - mask) )
    """
    def __init__(self,
                 w_ssim_mri=2.5, w_ssim_pet=2.0,
                 w_l1_mri=1.0,  w_l1_pet=1.0,
                 w_grad=2.0, w_aux=1.0, w_tv=0.0, w_pet_sal=3.0,
                 w_outside=0.15):
        super().__init__()
        self.w_ssim_mri = w_ssim_mri
        self.w_ssim_pet = w_ssim_pet
        self.w_l1_mri   = w_l1_mri
        self.w_l1_pet   = w_l1_pet
        self.w_grad     = w_grad
        self.w_aux      = w_aux
        self.w_tv       = w_tv
        self.w_pet_sal  = w_pet_sal
        self.w_outside  = w_outside

        self.l1 = nn.L1Loss()
        self.sobel = SobelGrad()

    def multi_scale_grad(self, y, mri):
        def one_scale(a):
            return self.sobel(a)
        g_y1, g_m1 = one_scale(y), one_scale(mri)
        y2, m2 = F.avg_pool2d(y, 2, 2), F.avg_pool2d(mri, 2, 2)
        g_y2, g_m2 = one_scale(y2), one_scale(m2)
        return charbonnier(g_y1 - g_m1).mean() + charbonnier(g_y2 - g_m2).mean()

    def forward(self, y, mri, pet, mask, ehat):
        # --- Classical fusion terms ---
        ssim_m = 1.0 - ssim_fn(rgb_to_grayscale_batch(y), mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet, data_range=1.0)
        l1_m   = self.l1(rgb_to_grayscale_batch(y), mri)
        l1_p   = self.l1(y, pet)

        # PET saliency-weighted L1 (kept from your code)
        pet_n = (pet - pet.amin(dim=(2,3), keepdim=True)) / \
                (pet.amax(dim=(2,3), keepdim=True) - pet.amin(dim=(2,3), keepdim=True) + 1e-6)
        S_pet = torch.sigmoid(6.0 * (pet_n - 0.4))
        l1_pet_sal = (S_pet * (y - pet).abs()).mean()

        g_cons   = self.multi_scale_grad(rgb_to_grayscale_batch(y), mri)
        target_e = self.sobel(mri).detach()
        aux      = self.l1(ehat, target_e)
        tv       = tv_loss(y)

        # --- Single extra term: penalize fused intensity outside cortex ---
        # y_pos = torch.clamp(y, min=0.0)  # be safe if model outputs slightly <0
        # outside_penalty = (y_pos * (1.0 - mask)).mean()
        outside_penalty = (torch.clamp(y, min=0.0) * (1.0 - mask) * (pet.detach() + 1e-6)).mean()
        
        # --- Single new term: scale-invariant leakage fraction ---
        y_pos = torch.clamp(y, min=0.0)
        num = (y_pos * (1.0 - mask)).sum(dim=(1,2,3))
        den = (y_pos.sum(dim=(1,2,3)) + 1e-8)
        leak_fraction = (num / den).mean()
        

        # Weighted sum
        loss = (
            self.w_ssim_mri * ssim_m +
            self.w_ssim_pet * ssim_p +
            self.w_l1_mri   * l1_m   +
            self.w_l1_pet   * l1_p   +
            self.w_grad     * g_cons +
            self.w_aux      * aux    +
            self.w_tv       * tv     +
            self.w_pet_sal  * l1_pet_sal +
            self.w_outside  * leak_fraction
        )

        log_terms = {
            "ssim_m": ssim_m.detach().mean().item(),
            "ssim_p": ssim_p.detach().mean().item(),
            "l1_m":   l1_m.detach().mean().item(),
            "l1_p":   l1_p.detach().mean().item(),
            "grad":   g_cons.detach().mean().item(),
            "aux":    aux.detach().mean().item(),
            "tv":     tv.detach().mean().item(),
            "pet_sal":l1_pet_sal.detach().mean().item(),
            "outside":leak_fraction.detach().mean().item(),
        }
        # print(log_terms)
        return loss, log_terms



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
    parser = argparse.ArgumentParser("Edge-guided MRI-PET fusion trainer (single outside-cortex penalty)")
    parser.add_argument("--data_root", type=str, default="/mnt/yasir/PET_Dataset_Processed/Separated_Centered_PET_MRI_Updated_Dataset_Splitted/train", help="contains MRI/, PET/, GM/")
    parser.add_argument("--save_dir",  type=str, default="./checkpoints_edgeguided")
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--size",      type=int, default=128)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--amp",       action="store_true", help="use mixed precision")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # classical weights
    parser.add_argument("--w_ssim_mri", type=float, default=2.5)
    parser.add_argument("--w_ssim_pet", type=float, default=2.3)
    parser.add_argument("--w_l1_mri",   type=float, default=1.0)
    parser.add_argument("--w_l1_pet",   type=float, default=1.0)
    parser.add_argument("--w_grad",     type=float, default=2.0)
    parser.add_argument("--w_aux",      type=float, default=1.0)
    parser.add_argument("--w_tv",       type=float, default=0.0)  # your note: TV=0 improved numeric results
    parser.add_argument("--w_pet_sal",  type=float, default=3.0)

    # single new weight
    parser.add_argument("--w_outside",  type=float, default=2, help="weight for penalizing fused intensity outside cortex")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(42)

    # Data
    train_set = PairFolder(args.data_root, size=args.size, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # Model
    model = Fusion_net(input_nc=4, output_nc=3).to(args.device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = EdgeGuidedFusionLoss(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad=args.w_grad, w_aux=args.w_aux, w_tv=args.w_tv, w_pet_sal=args.w_pet_sal,
        w_outside=args.w_outside
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
            x    = x.to(args.device)      # [B,2,H,W]
            mri  = mri.to(args.device)    # [B,1,H,W]
            pet  = pet.to(args.device)    # [B,1,H,W]
            mask = mask.to(args.device)   # [B,1,H,W]
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # Forward: encoder returns (feats, internal_edge); decoder returns (fused, aux_edge_pred)
                feats, edge_in = model.encode_with_edge(x)
                fused, edge_pred = model.decode_from_feats(feats)  # fused ∈ [0,1]

                # IMPORTANT: pass (y, mri, pet, mask, edge_pred) in this order
                loss, terms = loss_fn(fused, mri, pet, mask, edge_pred)

                if counter % 50 == 0:
                    with torch.no_grad():
                        mri_rgb = mri[:1].repeat(1, 3, 1, 1)  # Repeat the grayscale MRI to match the 3 channels
                        mask_rgb = mask[:1].repeat(1, 3, 1, 1)  # Repeat the mask to match the 3 channels if necessary
                        # print("Shape:",mri[:1].shape, pet[:1].shape, mask[:1].shape, fused[:1].shape)
                        panel = torch.cat([pet[:1], fused[:1]], dim=-1)
                        save_image(panel, os.path.join(out_dir, f"panel_{epoch}_{random.randint(1000, 9999)}.png"))
                counter += 1
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{running / (global_step or 1):.4f}",
                             ssim_m=f"{terms['ssim_m']:.3f}",
                             grad=f"{terms['grad']:.3f}",
                             aux=f"{terms['aux']:.3f}",
                             outside=f"{terms['outside']:.3f}")

        # Save checkpoint each epoch (optional)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        # torch.save(ckpt, os.path.join(args.save_dir,_

        # Final
    torch.save(model.state_dict(), os.path.join(args.save_dir, "edgeguided_final.pth"))
    print("Training complete. Checkpoints saved to:", args.save_dir)


if __name__ == "__main__":
    main()