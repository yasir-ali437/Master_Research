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

import torch
import torch.nn.functional as F
from math import pi

# ---------- shared helpers ----------
def _clamp_pos(x, eps=1e-8):
    return torch.clamp(x, min=0.0) + eps

def normalize_to_prob(x, eps=1e-8):
    """Normalize nonnegative map to a per-sample probability map (sum=1)."""
    x = _clamp_pos(x, eps)
    s = x.sum(dim=(-2, -1), keepdim=True)
    return x / (s + eps)

def make_cortex_target_prior(M, mode="uniform", beta=1.0, anat_weight=None):
    """
    Build Q over the cortex ribbon.
      mode="uniform": Q ∝ M
      mode="prob":    Q ∝ M^beta  (beta>1 concentrates on high-prob GM)
      mode="anat":    Q ∝ M * anat_weight  (e.g., thickness/gyri prior)
    Shapes: M,[B,1,H,W] in [0,1]; anat_weight optional [B,1,H,W] (>=0).
    Returns Q normalized to sum=1 per sample.
    """
    if mode == "uniform":
        Q = torch.clamp(M, 0, 1)
    elif mode == "prob":
        Q = torch.pow(torch.clamp(M, 0, 1), beta)
    elif mode == "anat":
        if anat_weight is None:
            raise ValueError("anat_weight must be provided when mode='anat'.")
        Q = torch.clamp(M, 0, 1) * torch.clamp(anat_weight, min=0.0)
    else:
        raise ValueError(f"Unknown prior mode: {mode}")
    return normalize_to_prob(Q)

def _coords_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1)  # [H,W,2]
    return coords.view(1, 1, H*W, 2)        # [1,1,HW,2]

# ---------- Option 1: Sliced-Wasserstein to cortex (fast) ----------
def sliced_wasserstein_to_cortex(P_src, Q_tgt, n_proj=64, seed=None):
    """
    Approximate W1 with projections.
    P_src, Q_tgt: probability maps [B,1,H,W], sum=1 per sample.
    Returns scalar (mean over batch).
    """
    B, _, H, W = P_src.shape
    device = P_src.device
    if seed is not None:
        torch.manual_seed(seed)

    coords = _coords_grid(H, W, device)      # [1,1,HW,2]
    P = P_src.view(B, 1, H*W)                 # [B,1,HW]
    Q = Q_tgt.view(B, 1, H*W)                 # [B,1,HW]

    theta = torch.rand(n_proj, device=device) * 2 * pi
    dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # [n,2]

    swd = 0.0
    for d in dirs:
        d = d.view(1, 1, 1, 2)
        proj = (coords * d).sum(-1)                     # [1,1,HW]
        idx = torch.argsort(proj.expand(B, 1, -1), -1)  # [B,1,HW]
        proj_sorted = torch.gather(proj.expand(B,1,-1), -1, idx)
        P_sorted    = torch.gather(P, -1, idx)
        Q_sorted    = torch.gather(Q, -1, idx)
        cP = torch.cumsum(P_sorted, dim=-1)
        cQ = torch.cumsum(Q_sorted, dim=-1)
        dt = torch.diff(proj_sorted, dim=-1, prepend=proj_sorted[..., :1])
        swd += torch.sum(torch.abs(cP - cQ) * dt, dim=-1)  # [B,1]

    swd = swd / n_proj
    return swd.mean()

# ---------- Option 2: Entropic Sinkhorn (balanced / unbalanced) ----------
def _sample_support(P, K, coords):
    """
    Randomly sample K positions according to P (no grad through sampling).
    P: [B,1,H,W] prob map (sum=1), coords: [1,1,HW,2]
    Returns (x:[B,K,2], w:[B,K]) normalized.
    """
    B, _, H, W = P.shape
    HW = H * W
    device = P.device
    p = P.view(B, HW)                       # [B,HW]
    idx = torch.multinomial(p, num_samples=K, replacement=True)  # [B,K]
    flat_coords = coords.view(1, HW, 2).expand(B, HW, 2)         # [B,HW,2]
    x = torch.gather(flat_coords, 1, idx.unsqueeze(-1).expand(B, K, 2))  # [B,K,2]
    w = torch.gather(p, 1, idx)                                    # [B,K]
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)                    # renorm
    return x, w

def _pairwise_sq_dists(x, y):
    """
    x: [B, Nx, 2], y:[B, Ny, 2] -> C: [B, Nx, Ny] with ||x-y||^2
    """
    x2 = (x**2).sum(-1, keepdim=True)        # [B,Nx,1]
    y2 = (y**2).sum(-1).unsqueeze(1)         # [B,1,Ny]
    C = x2 + y2 - 2.0 * x @ y.transpose(1, 2)
    return torch.clamp(C, min=0.0)

def sinkhorn_ot_to_cortex(
    P_src, Q_tgt,               # [B,1,H,W], sums=1
    n_samples=512, epsilon=0.05, n_iters=50,
    balanced=True,              # False -> unbalanced
    kl_strength_src=10.0,       # lambda (ρ) for unbalanced KL on source
    kl_strength_tgt=10.0
):
    """
    Entropic-regularized OT (balanced or unbalanced) via stabilized Sinkhorn.
    Samples supports from P and Q to keep it fast.
    Returns scalar regularized transport cost <π, C>.
    """
    B, _, H, W = P_src.shape
    device = P_src.device
    coords = _coords_grid(H, W, device)  # [1,1,HW,2]

    # sample support
    Xs, a = _sample_support(P_src, n_samples, coords)   # [B,K,2], [B,K]
    Xt, b = _sample_support(Q_tgt, n_samples, coords)   # [B,K,2], [B,K]

    C = _pairwise_sq_dists(Xs, Xt)    # [B,K,K]
    K = torch.exp(-C / epsilon).clamp_min(1e-12)  # Gibbs kernel

    # log-domain stabilized updates
    log_K = -C / epsilon                 # [B,K,K]
    log_a = torch.log(a + 1e-12)         # [B,K]
    log_b = torch.log(b + 1e-12)         # [B,K]
    log_u = torch.zeros_like(log_a)      # initialize u,v ~ 1
    log_v = torch.zeros_like(log_b)

    if balanced:
        for _ in range(n_iters):
            # log_u = log_a - logsumexp(log_K + log_v, axis=target)
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=-1)
            # log_v = log_b - logsumexp(log_K^T + log_u, axis=source)
            log_v = log_b - torch.logsumexp(log_K.transpose(1,2) + log_u.unsqueeze(2), dim=1)
    else:
        # Unbalanced: set tau = rho / (rho + epsilon)
        tau_a = kl_strength_src / (kl_strength_src + epsilon)
        tau_b = kl_strength_tgt / (kl_strength_tgt + epsilon)
        for _ in range(n_iters):
            log_u = tau_a * (log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=-1))
            log_v = tau_b * (log_b - torch.logsumexp(log_K.transpose(1,2) + log_u.unsqueeze(2), dim=1))

    # Expected transport cost: ⟨π, C⟩ with π = diag(u) K diag(v)
    # Use log form: sum_{ij} exp(log_u_i + log_K_ij + log_v_j) * C_ij
    log_pi = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)  # [B,K,K]
    # For numerical stability, mask tiny masses:
    log_pi = torch.clamp(log_pi, max=30.0)  # avoids inf when exp
    piC = torch.exp(log_pi) * C
    cost = piC.sum(dim=(1,2)).mean()  # mean over batch
    return cost

class EdgeGuidedFusionLoss(nn.Module):
    """
    Classical fusion loss (+ optional cortex-restricted OT terms).
    Add either/both:
      - Sliced-Wasserstein to cortex prior (fast): weight w_sot
      - Entropic Sinkhorn OT to cortex prior (balanced/unbalanced): weight w_sink
    """
    def __init__(self,
                 # classical
                 w_ssim_mri=2.5, w_ssim_pet=2.0,
                 w_l1_mri=1.0,  w_l1_pet=1.0,
                 w_grad=2.0, w_aux=1.0, w_tv=0.0, w_pet_sal=3.0,
                 # OT options
                 w_sot=2,            # set >0 to enable Sliced-Wasserstein
                 w_sink=2,           # set >0 to enable Sinkhorn
                 prior_mode="uniform", # 'uniform' | 'prob' | 'anat'
                 beta_prior=1.5,       # used when prior_mode='prob'
                 n_proj=64,            # Sliced-Wasserstein projections
                 sink_samples=512,     # #points sampled from each measure
                 sink_epsilon=0.05,    # entropic reg ε
                 sink_iters=50,        # #iterations
                 sink_balanced=True,   # False => unbalanced with KL
                 sink_kl_src=10.0,     # λ_source (ρ) for unbalanced
                 sink_kl_tgt=10.0):
        super().__init__()
        # classical
        self.w_ssim_mri = w_ssim_mri
        self.w_ssim_pet = w_ssim_pet
        self.w_l1_mri   = w_l1_mri
        self.w_l1_pet   = w_l1_pet
        self.w_grad     = w_grad
        self.w_aux      = w_aux
        self.w_tv       = w_tv
        self.w_pet_sal  = w_pet_sal

        # OT knobs
        self.w_sot      = w_sot
        self.w_sink     = w_sink
        self.prior_mode = prior_mode
        self.beta_prior = beta_prior
        self.n_proj     = n_proj

        self.sink_samples = sink_samples
        self.sink_epsilon = sink_epsilon
        self.sink_iters   = sink_iters
        self.sink_balanced= sink_balanced
        self.sink_kl_src  = sink_kl_src
        self.sink_kl_tgt  = sink_kl_tgt

        self.l1 = nn.L1Loss()
        self.sobel = SobelGrad()

    def multi_scale_grad(self, y, mri):
        def one_scale(a):
            return self.sobel(a)
        g_y1, g_m1 = one_scale(y), one_scale(mri)
        y2, m2 = F.avg_pool2d(y, 2, 2), F.avg_pool2d(mri, 2, 2)
        g_y2, g_m2 = one_scale(y2), one_scale(m2)
        return charbonnier(g_y1 - g_m1).mean() + charbonnier(g_y2 - g_m2).mean()

    def forward(self, y, mri, pet, mask, ehat, anat_weight=None):
        # --- classical terms ---
        ssim_m = 1.0 - ssim_fn(y, mri, data_range=1.0)
        ssim_p = 1.0 - ssim_fn(y, pet, data_range=1.0)
        l1_m   = self.l1(y, mri)
        l1_p   = self.l1(y, pet)

        pet_n = (pet - pet.amin(dim=(2,3), keepdim=True)) / \
                (pet.amax(dim=(2,3), keepdim=True) - pet.amin(dim=(2,3), keepdim=True) + 1e-6)
        S_pet = torch.sigmoid(6.0 * (pet_n - 0.4))
        l1_pet_sal = (S_pet * (y - pet).abs()).mean()

        g_cons   = self.multi_scale_grad(y, mri)
        target_e = self.sobel(mri).detach()
        aux      = self.l1(ehat, target_e)
        tv       = tv_loss(y)

        # --- OT terms (optional) ---
        # Source distribution = fused output density on image
        P_Y = normalize_to_prob(torch.clamp(y, min=0.0))
        # Target cortex prior Q
        Q = make_cortex_target_prior(mask, mode=self.prior_mode,
                                     beta=self.beta_prior, anat_weight=anat_weight)

        L_sot  = torch.tensor(0.0, device=y.device)
        L_sink = torch.tensor(0.0, device=y.device)

        if self.w_sot > 0.0:
            L_sot = sliced_wasserstein_to_cortex(P_Y, Q, n_proj=self.n_proj)

        if self.w_sink > 0.0:
            L_sink = sinkhorn_ot_to_cortex(
                P_Y, Q,
                n_samples=self.sink_samples,
                epsilon=self.sink_epsilon,
                n_iters=self.sink_iters,
                balanced=self.sink_balanced,
                kl_strength_src=self.sink_kl_src,
                kl_strength_tgt=self.sink_kl_tgt
            )

        # total
        loss = (
            self.w_ssim_mri * ssim_m +
            self.w_ssim_pet * ssim_p +
            self.w_l1_mri   * l1_m   +
            self.w_l1_pet   * l1_p   +
            self.w_grad     * g_cons +
            self.w_aux      * aux    +
            self.w_tv       * tv     +
            self.w_pet_sal  * l1_pet_sal +
            self.w_sot      * L_sot  +
            self.w_sink     * L_sink
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
            "sot":    L_sot.detach().mean().item() if self.w_sot>0 else 0.0,
            "sink":   L_sink.detach().mean().item() if self.w_sink>0 else 0.0,
        }
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
    model = Fusion_net(input_nc=2, output_nc=1).to(args.device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = EdgeGuidedFusionLoss(
        w_ssim_mri=args.w_ssim_mri, w_ssim_pet=args.w_ssim_pet,
        w_l1_mri=args.w_l1_mri, w_l1_pet=args.w_l1_pet,
        w_grad=args.w_grad, w_aux=args.w_aux, w_tv=args.w_tv, w_pet_sal=args.w_pet_sal
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
                        panel = torch.cat([mri[:1], pet[:1], mask[:1], fused[:1]], dim=-1)
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
                             aux=f"{terms['aux']:.3f}")

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