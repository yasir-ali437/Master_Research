import os, glob, random
from typing import Tuple, List

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from math import pi
import numpy as np
from typing import Optional, Sequence
from scipy import ndimage as ndi
from geomloss import SamplesLoss
from piq import ssim
from scipy.ndimage import gaussian_filter

try:
    from pytorch_msssim import ssim as ssim_fn  # pip install pytorch-msssim
except Exception as e:
    raise RuntimeError("Please install pytorch-msssim: pip install pytorch-msssim") from e

from model import Fusion_net

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="yasir517390-bilkent-university",
    # Set the wandb project where this run will be logged.
    project="PET-MRI Fusion",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "|Custom| MRI-PET",
        "epochs": 20,
    },
)
# ================================================
# Dataset
# ================================================
class PairFolder(Dataset):
    """Dataset with MRI, PET, and GM cortex mask triplets."""
    def __init__(self, root: str, size: int = 128, exts=("*.png","*.jpg","*.jpeg"),
                 augment=True, bin_thresh=0.5):
        self.mri_dir = os.path.join(root, "MRI")
        self.pet_dir = os.path.join(root, "PET_Deskulled")
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
        mri, pet, mask = self._load_gray(mri_path), self._load_gray(pet_path), self._load_gray(mask_path)
        mri, pet, mask = self._ensure_min_size([mri, pet, mask])
        mri, pet, mask = self._rand_crop_triplet(mri, pet, mask)

        if self.augment:
            if random.random() < 0.5:
                mri = self.rand_hflip(mri); pet = self.rand_hflip(pet); mask = self.rand_hflip(mask)
            if random.random() < 0.5:
                mri = self.rand_vflip(mri); pet = self.rand_vflip(pet); mask = self.rand_vflip(mask)

        mri_t = self.to_tensor(mri)
        pet_t = self.to_tensor(pet)
        mask_t = (self.to_tensor(mask) > self.bin_thresh).float()

        x = torch.cat([mri_t, pet_t, mask_t], dim=0)
        return x, mri_t, pet_t, mask_t


# ================================================
# Loss Helpers
# ================================================
class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]],dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1., 2., 1.],[0., 0., 0.],[-1., -2., -1.]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
        mag = mag / (mag.amax(dim=(2,3), keepdim=True).clamp_min(1e-6))
        return mag.clamp(0,1)

def charbonnier(x, eps=1e-3): return torch.sqrt(x*x + eps*eps)

def tv_loss(img):
    dy = img[:,:,1:,:] - img[:,:,:-1,:]
    dx = img[:,:,:,1:] - img[:,:,:,:-1]
    return (dy.abs().mean() + dx.abs().mean())


# ================================================
# Cortex transport: stable mask-projection loss
# ================================================
def mask_projection_loss(y, mask_cortex, alpha=0.05):
    y = F.softplus(y)  # positive & smooth
    y_norm = y / (y.amax(dim=(2,3), keepdim=True) + 1e-8)
    outside = ((1 - mask_cortex) * y_norm).mean()
    keep_alive = alpha * (y_norm**2).mean()
    return outside + keep_alive

def mask_projection_loss(y, mask_cortex, alpha_out=1.0, alpha_in=0.5):
    """
    Penalize intensity outside cortex (push down)
    and encourage activity inside cortex (push up).
    """
    y_pos = F.softplus(y)  # ensures positivity & gradients
    y_norm = y_pos / (y_pos.amax(dim=(2,3), keepdim=True) + 1e-8)

    outside_penalty = ((1 - mask_cortex) * y_norm).mean()
    inside_penalty  = (mask_cortex * (1 - y_norm)).mean()
    loss = alpha_out * outside_penalty + alpha_in * inside_penalty

    # small keep-alive term to avoid collapse
    loss += 0.01 * (y_norm**2).mean()
    return loss

import torch
import torch.nn.functional as F

def kl_transport_loss(pet, mask_cortex, mode="prob", beta=2.0, eps=1e-8):
    """
    KL divergence transport loss: aligns PET distribution to cortex prior.
    
    Args:
        pet:          [B,1,H,W] fused PET activation
        mask_cortex:  [B,1,H,W] binary or soft GM mask
        mode:         'uniform' | 'prob' | 'anat' (default: 'prob')
        beta:         exponent for probabilistic prior (beta>1 -> more focused)
        eps:          small constant for numerical stability
    Returns:
        Scalar loss (mean over batch)
    """

    # Clamp and normalize PET to probability map P(x)
    P = F.relu(pet)
    P = P / (P.sum(dim=(2,3), keepdim=True) + eps)

    # Construct target Q(x) depending on mode
    M = mask_cortex.clamp(0, 1)
    if mode == "uniform":
        Q = M
    elif mode == "prob":
        Q = M ** beta
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    Q = Q / (Q.sum(dim=(2,3), keepdim=True) + eps)

    # Compute KL divergence
    loss = (P * (torch.log(P + eps) - torch.log(Q + eps))).sum(dim=(2,3)).mean()
    return loss

# def distance_map_outside(mask01: np.ndarray,
#                          sampling: Optional[Sequence[float]] = None) -> np.ndarray:
#     """
#     OUTSIDE-ONLY distance: D >= 0 outside cortex, D = 0 inside cortex.
#     `mask01` is {0,1}, with 1=cortex. 2D or 3D.
#     `sampling` is e.g. (dy, dx) for 2D or (dz, dy, dx) for 3D, in mm.
#     """
#     mask01 = (mask01 > 0).astype(np.uint8)
#     outside = (mask01 == 0)
#     D = ndi.distance_transform_edt(outside, sampling=sampling).astype(np.float32)
#     D[mask01 == 1] = 0.0
#     return D



def distance_map_outside(mask: torch.Tensor, spacing=(1.0, 1.0)):
    """
    Compute OUTSIDE-ONLY distance maps from a mask tensor.
    Input:  mask [B,1,H,W] (or [B,1,D,H,W]); values in {0,1} or [0,1].
    Output: D    same shape as mask (float32) on mask.device.
    """
    assert mask.ndim in (4, 5) and mask.shape[1] == 1, "mask must be [B,1,H,W] or [B,1,D,H,W]"
    B = mask.shape[0]
    is3d = (mask.ndim == 5)
    out = torch.empty_like(mask, dtype=torch.float32)

    for b in range(B):
        mb = (mask[b, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)  # 1=cortex
        if is3d:
            # spacing = (dz, dy, dx) for 3D
            D = ndi.distance_transform_edt(mb == 0, sampling=spacing).astype(np.float32)
            D[mb == 1] = 0.0
        else:
            # spacing = (dy, dx) for 2D
            D = ndi.distance_transform_edt(mb == 0, sampling=spacing).astype(np.float32)
            D[mb == 1] = 0.0

        out[b, 0] = torch.from_numpy(D).to(mask.device)

    return out


def outside_cortex_penalty(Y, M, D=None, alpha=1.0, eps=1e-8):
    """
    Penalize fused intensity outside cortex, stronger the farther from cortex.

    Y : [B,1,H,W(,D)] fused output (any real) -> internally clamped >=0
    M : [B,1,H,W(,D)] cortex prior in [0,1]
    D : [B,1,H,W(,D)] nonnegative distance-to-cortex map (optional)
        If None, uses uniform outside penalty (no distance weighting).
    alpha : distance strength; typical 0.5–3
    """
    Yp = torch.clamp(Y, min=0.0)
    if D is None:
        # simple version: no distance weighting
        return (Yp * (1.0 - M)).mean()

    # Normalize D per-sample so alpha has a consistent meaning
    spatial_dims = tuple(range(2, Y.ndim))
    Dmax = D.amax(dim=spatial_dims, keepdim=True).clamp_min(eps)
    Dn = D / Dmax  # now in ~[0,1]
    return (Yp * (1.0 - M) * (1.0 + alpha * Dn)).mean()

def outside_cortex_penalty_balanced(Y, M, D=None, alpha=1.0, beta=1, eps=1e-8):
    """
    Penalize intensity outside cortex, reward intensity inside cortex slightly.
    Prevents collapse to zero output.
    """
    Yp = torch.clamp(Y, 0.0, 1.0)
    spatial_dims = tuple(range(2, Y.ndim))

    if D is not None:
        Dmax = D.amax(dim=spatial_dims, keepdim=True).clamp_min(eps)
        Dn = D / Dmax
    else:
        Dn = 0.0

    outside_term = (Yp * (1.0 - M) * (1.0 + alpha * Dn)).mean()
    inside_term  = ((1.0 - Yp) * M).mean()  # encourage non-zero inside mask
    loss = outside_term + beta * inside_term
    return loss

def cortex_transport_preserve_loss(Y, PET, mask, D=None, alpha=1.0, beta=0.5, gamma_tv=0.02, eps=1e-8):
    """
    Drives PET energy into cortex region, while preserving PET intensity patterns inside mask.

    Args:
        Y:    fused output [B,1,H,W]
        PET:  PET reference [B,1,H,W]
        mask: cortex mask [B,1,H,W]
        D:    optional distance map for outside weighting
        alpha: outside penalty strength
        beta: inside preservation weight
        gamma_tv: total variation regularization for smoothness
    """
    Yp = torch.sigmoid(Y)  # or torch.clamp(Y, 0, 1)
    spatial_dims = tuple(range(2, Y.ndim))

    # --- 1. Penalize intensity outside cortex ---
    if D is not None:
        Dmax = D.amax(dim=spatial_dims, keepdim=True).clamp_min(eps)
        Dn = D / Dmax
        outside_term = (Yp * (1.0 - mask) * (1.0 + alpha * Dn)).mean()
    else:
        outside_term = (Yp * (1.0 - mask)).mean()

    # --- 2. Preserve PET pattern inside cortex ---
    pet_in = PET * mask
    pet_in = pet_in / (pet_in.amax(dim=spatial_dims, keepdim=True) + eps)
    Y_in = Yp * mask
    inside_term = F.l1_loss(Y_in, pet_in)  # pattern-preserving match

    # --- 3. Optional smoothness regularizer ---
    dy = Yp[:,:,1:,:] - Yp[:,:,:-1,:]
    dx = Yp[:,:,:,1:] - Yp[:,:,:,:-1]
    tv = (dy.abs().mean() + dx.abs().mean())

    # --- Total loss ---
    loss = alpha * outside_term + beta * inside_term + gamma_tv * tv
    return loss

def intensity_preservation_loss(y, pet, mask, alpha=1.0):
    """Loss to preserve the original PET intensities inside the cortex."""
    pet_in_cortex = pet * mask  # PET values inside cortex
    y_in_cortex = y * mask  # Fused PET values inside cortex
    return alpha * F.mse_loss(y_in_cortex, pet_in_cortex)

def gaussian_smooth(x, kernel_size=3, sigma=1.0):
    """
    Applies Gaussian smoothing to the input tensor `x` manually.
    
    Args:
        x (tensor): Input tensor (B, C, H, W).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for the Gaussian kernel.
    
    Returns:
        tensor: Smoothed tensor.
    """
    # Convert the input tensor to numpy for processing
    x_np = x.detach().cpu().numpy()
    
    # Create a Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)  # Normalize the kernel

    # Apply Gaussian filter to each image in the batch
    smoothed = []
    for i in range(x_np.shape[0]):  # Iterate over the batch
        smoothed_img = np.zeros_like(x_np[i])
        for c in range(x_np.shape[1]):  # Iterate over channels
            smoothed_img[c] = gaussian_filter(x_np[i, c], sigma=sigma)
        smoothed.append(smoothed_img)

    # Convert the smoothed numpy back to tensor
    smoothed_tensor = torch.tensor(np.array(smoothed)).to(x.device)
    return smoothed_tensor
# ================================================
# Full Loss
# ================================================
class EdgeGuidedFusionLoss(nn.Module):
    def __init__(self, **weights):
        super().__init__()
        for k,v in weights.items(): setattr(self,k,v)
        self.l1 = nn.L1Loss()
        self.sobel = SobelGrad()
        self.sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)
        self.alpha = 0.8
        self.beta = 0.3
        self.gamma = 0.5
        
        
    def multi_scale_grad(self, y, mri):
        g_y1, g_m1 = self.sobel(y), self.sobel(mri)
        y2, m2 = F.avg_pool2d(y,2,2), F.avg_pool2d(mri,2,2)
        g_y2, g_m2 = self.sobel(y2), self.sobel(m2)
        return charbonnier(g_y1-g_m1).mean() + charbonnier(g_y2-g_m2).mean()

    def forward(self, y, mri, pet, mask, ehat):
        ssim_m = 1.0 - ssim_fn(y, mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet, data_range=1.0)
        l1_m, l1_p = self.l1(y,mri), self.l1(y,pet)

        pet_n = (pet - pet.amin(dim=(2,3),keepdim=True)) / \
                (pet.amax(dim=(2,3),keepdim=True) - pet.amin(dim=(2,3),keepdim=True) + 1e-6)
        S_pet = torch.sigmoid(6.0 * (pet_n - 0.4))
        l1_pet_sal = (S_pet * (y - pet).abs()).mean()

        g_cons = self.multi_scale_grad(y,mri)
        target_e = self.sobel(mri).detach()
        aux = self.l1(ehat, target_e)
        tv  = tv_loss(y)
        
        ##############
        ############## outside_cortex_penalty_balanced
        ##############
        # --- NEW: outside-cortex penalty ---
        # dist = distance_map_outside(mask)
        # L_out = outside_cortex_penalty_balanced(y, mask)
        # loss_total = self.w_outside * L_out
        
        ##############
        ############## cortex_transport_preserve_loss
        ##############
        # Yp = y / (y.amax(dim=(2,3), keepdim=True) + 1e-8)
        # loss_transport = cortex_transport_preserve_loss(Yp, pet, mask, D=dist)
        # loss_total = self.w_outside * L_out
    

        ##############
        ############## Mask Project and KL Loss
        ##############
        
        # y = y.float()
        # loss_transport = mask_projection_loss(y, mask)
        # loss_transport = kl_transport_loss(y, mask)
        
        # loss_total = self.w_outside * loss_transport
        
        ##############
        ############## IT was CAUSING MEMORY ISSUESSS
        ##############
        
        # B = pet.size(0)
        # flatten = lambda t: t.view(B, -1, 1)
        # pet_masked = flatten(pet * mask)
        # y_masked = flatten(y * mask)
        # L_transport = self.sinkhorn(y_masked, pet_masked).mean()
        # L_outside = torch.mean((y * (1 - mask)) ** 2)
        # L_ssim = 1 - ssim(y * mask, mri * mask, data_range=1.0)
        # loss_total = self.alpha * L_transport + self.beta * L_outside + self.gamma * L_ssim
        # return total, L_transport, L_outside, L_ssim

        ##############
        ############## Best Working Till Now
        ##############
        
        B = pet.size(0)
        L_transport_total, L_out_total, L_ssim_total = 0.0, 0.0, 0.0

        for i in range(B):
            # Extract one sample to avoid multiscale batching issues
            pet_i = (pet[i] * mask[i]).view(-1, 1)
            y_i = (y[i] * mask[i]).view(-1, 1)
            # Filter valid pixels
            valid = mask[i].flatten() > 0.1
            if valid.sum() < 32:
                # skip transport if almost no cortex pixels
                continue
            # Randomly downsample pixels to reduce cost further
            if pet_i.shape[0] > 4096:
                idx = torch.randperm(pet_i.shape[0])[:4096]
                pet_i, y_i = pet_i[idx], y_i[idx]

            L_transport_total += self.sinkhorn(y_i, pet_i)

            L_out_total += torch.mean((y[i] * (1 - mask[i])) ** 2)
            L_ssim_total += 1 - ssim(y[i:i+1] * mask[i:i+1], mri[i:i+1] * mask[i:i+1], data_range=1.0)

        L_transport_total /= B
        L_out_total /= B
        L_ssim_total /= B
        loss_total = self.alpha * L_transport_total + self.beta * L_out_total + self.gamma * L_ssim_total

        L_intensity = intensity_preservation_loss(y, pet, mask, alpha=2.5)
        print("LOSS of Trasnport:", loss_total.item(), " | Intensity Preservation Loss:", L_intensity.item())
        smoothed_fused = gaussian_smooth(y)
        L_smooth = F.mse_loss(smoothed_fused, y)
        print("SMOOTHNESS LOSS:", L_smooth.item())
        # loss_total += L_smooth
        loss_total += L_intensity
        
        loss = (
                self.w_ssim_mri*ssim_m + 
                # self.w_ssim_pet*ssim_p +
                # self.w_l1_mri*l1_m + 
                # self.w_l1_pet*l1_p +
                # self.w_grad*g_cons + self.w_aux*aux +
                # self.w_tv*tv + self.w_pet_sal*l1_pet_sal + 
                loss_total)
        
        # Log metrics to wandb.
        run.log({"loss": loss})
        
        # print(f"\nOutside-cortex penalty: {loss_total.item():.4f} | Total loss: {loss.item():.4f}")
        return loss, {
            "ssim_m": ssim_m.mean().item(),
            "ssim_p": ssim_p.mean().item(),
            "grad": g_cons.mean().item(),
            "aux": aux.mean().item()
        }


# ================================================
# Training
# ================================================
def seed_everything(seed=1234):
    import numpy as np, random as pyrand
    pyrand.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    import argparse
    parser = argparse.ArgumentParser("MRI-PET fusion with cortex constraint")
    parser.add_argument("--data_root", type=str, default="/mnt/yasir/PET_Dataset_Processed/Separated_Centered_PET_MRI_Updated_Dataset_Splitted/train")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_edgeguided")
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
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = EdgeGuidedFusionLoss(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad=args.w_grad, w_aux=args.w_aux,
        w_tv=args.w_tv, w_pet_sal=args.w_pet_sal,
        w_outside=args.w_outside
    ).to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    os.makedirs("./samples", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        for i,(x,mri,pet,mask) in enumerate(pbar):
            x,mri,pet,mask = x.to(args.device), mri.to(args.device), pet.to(args.device), mask.to(args.device)
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
                    panel = torch.cat([mri[:1], pet[:1], mask[:1], fused[:1]], dim=-1)
                    save_image(panel, f"./samples/panel_e{epoch}_b{i}.png")

            pbar.set_postfix(loss=f"{running/(i+1):.4f}", ssim_m=f"{terms['ssim_m']:.3f}", grad=f"{terms['grad']:.3f}")

        scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"edgeguided_epoch{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "edgeguided_final.pth"))
    print("Training complete. Checkpoints saved to:", args.save_dir)


if __name__ == "__main__":
    main()
