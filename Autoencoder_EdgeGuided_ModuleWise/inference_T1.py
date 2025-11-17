# ================================================================
# Inference for the new Dual-Branch Edge+Transformer Fusion model
# - Works with grayscale PET or RGB PET
# - If PET is RGB, reconstructs color by Y_fused ⊕ (Cb,Cr)_PET
# - Auto-pads to multiples of 8, then crops back
# - Saves fused (Y or RGB) and optional edge prediction
#
# Usage:
#   python infer_new.py \
#       --data_root /path/to/data            # contains mri/ & pet/ (or MRI/ & PET/)
#       --ckpt ./checkpoints_fusion_v2/fusion_best.pth
#       --out_dir ./fused_out
#       --pet_color                          # set if PET images are RGB
#       --save_edges                         # also save predicted edges
#       --amp                                # (optional) mixed precision on GPU
#
# NOTE: Ensure the FusionNet class used for training is importable here.
#       If it's defined in a different file/module, adjust the import below.
# ================================================================

import os, glob, argparse
from typing import Tuple, List
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# ==== IMPORT YOUR TRAINED MODEL CLASS ====
# If your model is in "train_T1.py" or another module, change the import:
# from train_T1 import FusionNet
from train_T1 import FusionNet  # <-- update to your actual module if needed

# ---------- I/O helpers ----------
to_tensor = transforms.ToTensor()      # [0,1], shape [C,H,W]
to_pil    = transforms.ToPILImage()    # expects [C,H,W] in [0,1]

def load_gray(path: str) -> torch.Tensor:
    """Load single-channel image as float tensor [1,H,W] in [0,1]."""
    img = Image.open(path).convert("L")
    return to_tensor(img)

def load_rgb(path: str) -> torch.Tensor:
    """Load RGB image as float tensor [3,H,W] in [0,1]."""
    img = Image.open(path).convert("RGB")
    return to_tensor(img)

# ---------- YCbCr helpers (numpy / PIL friendly) ----------
def rgb_pil_to_ycbcr_np(image_pil: Image.Image):
    """PIL RGB -> (Y,Cb,Cr) as float64 numpy arrays in the same HxW space (Y clamped to [0,255])."""
    rgb = np.asarray(image_pil.convert("RGB"), dtype=np.float64)  # [H,W,3] 0..255
    # BT.601-ish transform
    M = np.array([[ 0.299,   0.587,   0.114 ],
                  [-0.168736,-0.331264,0.5   ],
                  [ 0.5,    -0.418688,-0.081312]])
    ycbcr = rgb @ M.T
    y  = np.clip(ycbcr[...,0], 0, 255)
    cb = ycbcr[...,1] + 128.0
    cr = ycbcr[...,2] + 128.0
    return y, cb, cr

def ycbcr_np_to_rgb_pil(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> Image.Image:
    """(Y,Cb,Cr) arrays (0..255-ish) -> PIL RGB image."""
    ycbcr = np.stack([y, cb-128.0, cr-128.0], axis=-1)
    # inverse (approx)
    M_inv = np.array([[1.0,  0.0,       1.402   ],
                      [1.0, -0.344136, -0.714136],
                      [1.0,  1.772,     0.0     ]])
    rgb = ycbcr @ M_inv.T
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")

def save_gray(t: torch.Tensor, path: str):
    """Save [1,H,W] or [H,W] in [0,1] to path as 8-bit PNG."""
    t = t.detach()
    if t.dim() == 4:  # [B,1,H,W] -> first
        t = t[0]
    if t.dim() == 2:
        t = t.unsqueeze(0)
    img = to_pil(t.cpu().clamp(0,1).float())
    img.save(path)

# ---------- padding ----------
def pad_to_multiple(x: torch.Tensor, mul: int = 8) -> Tuple[torch.Tensor, Tuple[int,int]]:
    """Pad [B,C,H,W] to multiple-of-`mul` spatial dims (reflect). Return x_pad and (H0,W0)."""
    B,C,H,W = x.shape
    Ht = (H + mul - 1)//mul*mul
    Wt = (W + mul - 1)//mul*mul
    pad_h, pad_w = Ht - H, Wt - W
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (H, W)

def crop_to_size(x: torch.Tensor, orig_hw: Tuple[int,int]) -> torch.Tensor:
    H0, W0 = orig_hw
    return x[..., :H0, :W0]

# ---------- pair collection ----------
def collect_pairs(root: str, exts=("*.png","*.jpg","*.jpeg")) -> List[Tuple[str,str,str]]:
    """
    Return list of (stem, mri_path, pet_path).
    Accepts either 'mri'/'pet' or 'MRI'/'PET' subfolders.
    """
    cand_mri = [os.path.join(root, "mri"), os.path.join(root, "MRI")]
    cand_pet = [os.path.join(root, "pet"), os.path.join(root, "/data1/yasir/test/PET_Transported")]
    mri_dir = next((d for d in cand_mri if os.path.isdir(d)), None)
    pet_dir = next((d for d in cand_pet if os.path.isdir(d)), None)
    if mri_dir is None or pet_dir is None:
        raise FileNotFoundError(f"Could not find 'mri'/'pet' (or 'MRI'/'PET') under {root}")

    mri_files, pet_files = [], []
    for p in exts:
        mri_files += glob.glob(os.path.join(mri_dir, p))
        pet_files += glob.glob(os.path.join(pet_dir, p))
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    m_map = {stem(p): p for p in mri_files}
    p_map = {stem(p): p for p in pet_files}
    common = sorted(set(m_map.keys()) & set(p_map.keys()))
    return [(s, m_map[s], p_map[s]) for s in common]

# ---------- checkpoint loader ----------
def load_model(ckpt_path: str, device: str = "cuda",
               base:int=32, growth:int=32, cstf_heads:int=2, cstf_depth:int=1):
    """
    Load FusionNet with the same hyperparams as training.
    """
    model = FusionNet(input_nc=2, out_nc=1, base=base, growth=growth,
                      cstf_heads=cstf_heads, cstf_depth=cstf_depth).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # accept either pure state_dict or {"model": state_dict, ...}
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ---------- core fuse ----------
@torch.no_grad()
def fuse_pair(model: torch.nn.Module,
              mri_1ch: torch.Tensor,
              pet: torch.Tensor,
              device="cuda",
              amp: bool = True,
              pet_color: bool = False):
    """
    mri_1ch: [1,H,W] in [0,1]  (torch)
    pet:     [1,H,W] (grayscale) OR [3,H,W] (RGB) in [0,1] (torch)
    Returns:
      fused_y [1,H,W] in [0,1]
      edge_pred [1,H,W] in [0,1]
    """
    mri = mri_1ch.unsqueeze(0).to(device)           # [1,1,H,W]

    if pet_color and pet.shape[0] == 3:
        # Convert RGB PET -> Y on-the-fly in torch
        r,g,b = pet[0:1], pet[1:2], pet[2:3]
        pet_y = (0.299*r + 0.587*g + 0.114*b).unsqueeze(0).to(device)
    else:
        pet_y = pet.unsqueeze(0).to(device)         # [1,1,H,W]

    x = torch.cat([mri, pet_y], dim=1)              # [1,2,H,W]
    x_pad, hw0 = pad_to_multiple(x, 8)

    with torch.cuda.amp.autocast(enabled=amp and device.startswith("cuda")):
        fused_pad, edge_pred_pad, _ = model(x_pad)  # [1,1,Ht,Wt], [1,1,Ht,Wt], edge_in

    fused = crop_to_size(fused_pad, hw0).clamp(0,1)       # [1,1,H,W]
    edge  = crop_to_size(edge_pred_pad, hw0).clamp(0,1)   # use decoder edge prediction
    return fused[0].cpu(), edge[0].cpu()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Dual-branch PET–MRI fusion inference")
    ap.add_argument("--data_root", type=str, default="/data1/yasir/test",
                    help="Folder containing mri/ & pet/ (or MRI/ & PET/) subfolders")
    ap.add_argument("--ckpt", type=str, default="checkpoints_fusion_v2/fusion_best.pth",
                    help="Path to trained checkpoint (.pth or .pt)")
    ap.add_argument("--out_dir", type=str, default="./fused_out", help="Where to save fused images")
    ap.add_argument("--exts", type=str, nargs="+", default=["*.png","*.jpg","*.jpeg"], help="File patterns to match")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    ap.add_argument("--pet_color", action="store_true", help="Treat PET as RGB and reconstruct color")
    ap.add_argument("--save_edges", action="store_true", help="Also save predicted edge maps")
    # if you changed these during training, set the same values here:
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--growth", type=int, default=32)
    ap.add_argument("--cstf_heads", type=int, default=2)
    ap.add_argument("--cstf_depth", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_edges:
        os.makedirs(os.path.join(args.out_dir, "edges"), exist_ok=True)

    pairs = collect_pairs(args.data_root, exts=tuple(args.exts))
    if not pairs:
        raise FileNotFoundError(f"No matching pairs found under {args.data_root}")
    print(f"Found {len(pairs)} pairs. Loading model from {args.ckpt} ...")

    model = load_model(args.ckpt, device=args.device,
                       base=args.base, growth=args.growth,
                       cstf_heads=args.cstf_heads, cstf_depth=args.cstf_depth)

    for stem, mri_path, pet_path in pairs:
        # Load MRI
        mri = load_gray(mri_path)  # [1,H,W]

        # Load PET (grayscale or RGB)
        if args.pet_color:
            pet_rgb_pil = Image.open(pet_path).convert("RGB")
            pet = to_tensor(pet_rgb_pil)  # [3,H,W] in [0,1] torch
        else:
            pet = load_gray(pet_path)     # [1,H,W] in [0,1] torch

        # Fuse
        fused_y, edge_pred = fuse_pair(model, mri, pet, device=args.device, amp=args.amp, pet_color=args.pet_color)
        # Save outputs
        if args.pet_color:
            # Build color fused: (Y_fused ⊕ CbCr_pet_original)
            y_np = (fused_y.squeeze(0).numpy() * 400.0).round().astype(np.uint8)  # [H,W] 0..255
            pet_img = Image.open(pet_path).convert("RGB")
            _, cb_np, cr_np = rgb_pil_to_ycbcr_np(pet_img)  # cb/cr as float
            rgb_fused = ycbcr_np_to_rgb_pil(y_np.astype(np.float64), cb_np, cr_np)
            out_path = os.path.join(args.out_dir, f"{stem}_fusedRGB.png")
            rgb_fused.save(out_path)
        else:
            out_path = os.path.join(args.out_dir, f"{stem}_fusedY.png")
            save_gray(fused_y, out_path)

        if args.save_edges:
            edge_path = os.path.join(args.out_dir, "edges", f"{stem}_edge.png")
            save_gray(edge_pred, edge_path)

        print(f"Saved: {out_path}" + (" (edge saved)" if args.save_edges else ""))

    print("Done.")

if __name__ == "__main__":
    main()
