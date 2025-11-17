import os
import glob
import math
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

# Your MUFusion-style network (same as used in training)
from net import TwoFusion_net


# -----------------------------
# Utilities
# -----------------------------
def load_rgb(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")  # force RGB for PET images
    return img

def load_gray(path: str) -> Image.Image:
    img = Image.open(path).convert("L")  # force grayscale for MRI images
    return img

def to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None, :, :]  # [1,H,W]
    elif arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)  # [C,H,W]
    else:
        raise ValueError("Expected grayscale or RGB image.")
    return torch.from_numpy(arr)  # [C,H,W], 0..1

def to_image(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1).detach().cpu().numpy()
    t = (t * 255.0 + 0.5).astype(np.uint8)
    t = t.transpose(1, 2, 0)  # [H, W, C]
    return Image.fromarray(t)

def rgb2ycbcr(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    return Y, Cb, Cr

def ycbcr2rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    return np.stack([R, G, B], axis=-1)

def pair_by_stem(ir_dir: str, vis_dir: str, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> List[Tuple[str,str,str]]:
    def collect(d):
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(d, f"*{e}")))
        return {os.path.splitext(os.path.basename(p))[0]: p for p in files}

    ir_map  = collect(ir_dir)
    vis_map = collect(vis_dir)
    common = sorted(set(ir_map.keys()) & set(vis_map.keys()))
    if not common:
        raise FileNotFoundError(f"No matching stems between {ir_dir} and {vis_dir}.")
    return [(stem, ir_map[stem], vis_map[stem]) for stem in common]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_weights(model: torch.nn.Module, weights_path: str, device: str):
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[warn] Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    B, C, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, (0, pad_w, 0, pad_h)

def unpad(x: torch.Tensor, pads: Tuple[int,int,int,int]) -> torch.Tensor:
    l, r, t, b = pads
    H = x.shape[-2] - t - b
    W = x.shape[-1] - l - r
    return x[..., :H, :W]

def tile_coords(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int,int,int,int]]:
    ys = list(range(0, max(1, H - tile + 1), tile - overlap))
    xs = list(range(0, max(1, W - tile + 1), tile - overlap))
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    return [(y, y + tile, x, x + tile) for y in ys for x in xs]

# -----------------------------
# Inference (single pass)
# -----------------------------
@torch.no_grad()
def fuse_pair(model, ir_img: Image.Image, vis_img: Image.Image, device="cuda", amp=False,
              pad_mult: int = 16, tile: Optional[int] = None, overlap: int = 32) -> Image.Image:
    # Convert RGB PET image to YCbCr
    ir_np = np.array(ir_img)
    Y_ir, Cb_ir, Cr_ir = rgb2ycbcr(ir_np)
    
    # Convert grayscale MRI image to tensor
    vis_t = to_tensor01(vis_img).to(device)  # Move MRI tensor to the same device as the model

    # Fuse only Y channel with MRI
    Y_ir_t = torch.from_numpy(Y_ir).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions and move to the same device
    print("Shape: ", vis_t.shape, Y_ir_t.shape)
    fused_Y = model.encoder(torch.cat([Y_ir_t, vis_t], dim=0).unsqueeze(0).to(device))  # Apply fusion

    # After fusion, merge the Cb and Cr channels back with the fused Y channel
    fused_Y = fused_Y.squeeze().cpu().numpy()  # Get the fused Y channel back as numpy array
    fused_img = ycbcr2rgb(fused_Y, Cb_ir, Cr_ir)

    return Image.fromarray(fused_img.astype(np.uint8))


# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser("Autoencoder fusion inference")
    parser.add_argument("--weights", required=True, help="Path to .pt or .pth (model or checkpoint dict)")
    parser.add_argument("--ir", help="Path to IR image (single pair mode)")
    parser.add_argument("--vis", help="Path to VIS image (single pair mode)")
    parser.add_argument("--ir_dir", help="Folder with IR images (batch mode)")
    parser.add_argument("--vis_dir", help="Folder with VIS images (batch mode)")
    parser.add_argument("--out_dir", default="./fused_out_new", help="Where to write outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--tile", type=int, default=None, help="Tile size (e.g., 512). If omitted, runs full-frame.")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap (pixels) between tiles")
    parser.add_argument("--pad_mult", type=int, default=16, help="Pad H/W to multiple of this")
    args = parser.parse_args()

    # Model
    device = args.device
    model = TwoFusion_net(input_nc=2, output_nc=1).to(device)
    load_weights(model, args.weights, device)
    model.eval()

    ensure_dir(args.out_dir)

    # Single pair
    if args.ir and args.vis:
        ir_img  = load_rgb(args.ir)  # RGB PET image
        vis_img = load_gray(args.vis)  # Grayscale MRI image
        if ir_img.size != vis_img.size:
            vis_img = vis_img.resize(ir_img.size, Image.BICUBIC)
        fused = fuse_pair(model, ir_img, vis_img, device=device, amp=args.amp,
                          pad_mult=args.pad_mult, tile=args.tile, overlap=args.overlap)
        stem = os.path.splitext(os.path.basename(args.ir))[0]
        out_path = os.path.join(args.out_dir, f"{stem}_FUSED.png")
        fused.save(out_path)
        print("Saved:", out_path)
        return

    # Batch folder mode
    if args.ir_dir and args.vis_dir:
        pairs = pair_by_stem(args.ir_dir, args.vis_dir)
        for stem, ir_p, vis_p in tqdm(pairs, desc="Fusing"):
            ir_img  = load_rgb(ir_p)  # RGB PET image
            vis_img = load_gray(vis_p)  # Grayscale MRI image
            if ir_img.size != vis_img.size:
                vis_img = vis_img.resize(ir_img.size, Image.BICUBIC)
            fused = fuse_pair(model, ir_img, vis_img, device=device, amp=args.amp,
                              pad_mult=args.pad_mult, tile=args.tile, overlap=args.overlap)
            out_path = os.path.join(args.out_dir, f"{stem}_FUSED.png")
            fused.save(out_path)
        print(f"Done. Wrote {len(pairs)} files to {args.out_dir}")
        return

    raise SystemExit("Specify either --ir & --vis (single pair) OR --ir_dir & --vis_dir (batch).")


if __name__ == "__main__":
    main()
