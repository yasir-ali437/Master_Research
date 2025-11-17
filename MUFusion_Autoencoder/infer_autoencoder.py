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
def load_gray(path: str) -> Image.Image:
    img = Image.open(path).convert("L")  # force grayscale
    return img

def to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None, :, :]  # [1,H,W]
    else:
        raise ValueError("Expected grayscale image.")
    return torch.from_numpy(arr)  # [1,H,W], 0..1

def to_image(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1).detach().cpu().numpy()
    t = (t * 255.0 + 0.5).astype(np.uint8)
    t = t[0]  # [H,W]
    return Image.fromarray(t, mode="L")

def pair_by_stem(ir_dir: str, vis_dir: str, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")) -> List[Tuple[str,str,str]]:
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
    """Pad 4D tensor [B,C,H,W] to multiples of `multiple` (common for UNet-like nets)."""
    B, C, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    # F.pad uses (left, right, top, bottom)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, (0, pad_w, 0, pad_h)

def unpad(x: torch.Tensor, pads: Tuple[int,int,int,int]) -> torch.Tensor:
    l, r, t, b = pads
    H = x.shape[-2] - t - b
    W = x.shape[-1] - l - r
    return x[..., :H, :W]

def tile_coords(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int,int,int,int]]:
    """Return list of (y0,y1,x0,x1) windows covering HxW with overlap."""
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
    """
    If `tile` is set (e.g., 512), the image is processed in overlapping tiles to save memory.
    """
    ir_t  = to_tensor01(ir_img)
    vis_t = to_tensor01(vis_img)
    assert ir_t.shape[-2:] == vis_t.shape[-2:], "IR and VIS must be same size"
    H, W = ir_t.shape[-2], vis_t.shape[-1]

    # stack to [1,2,H,W]
    x = torch.cat([ir_t, vis_t], dim=0).unsqueeze(0).to(device)

    if tile is None:
        # Pad to network-friendly size
        x_pad, pads = pad_to_multiple(x, pad_mult)
        with torch.autocast(device_type=("cuda" if device.startswith("cuda") else "cpu"), enabled=amp):
            feats = model.encoder(x_pad)
            out   = model.decoder(feats)[0]  # [1,1,H',W']
            out   = torch.clamp(out, 0.0, 1.0)
        out = unpad(out, pads)
        out = out[..., :H, :W]
        return to_image(out[0])
    else:
        # Tiled path
        out_full = torch.zeros((1, 1, H, W), device=device)
        weight   = torch.zeros((1, 1, H, W), device=device)

        coords = tile_coords(H, W, tile, overlap)
        for (y0, y1, x0, x1) in coords:
            x_crop = x[..., y0:y1, x0:x1]
            # pad each tile to multiple
            x_crop_pad, pads = pad_to_multiple(x_crop, pad_mult)
            with torch.autocast(device_type=("cuda" if device.startswith("cuda") else "cpu"), enabled=amp):
                feats = model.encoder(x_crop_pad)
                out   = model.decoder(feats)[0]
                out   = torch.clamp(out, 0.0, 1.0)
            out = unpad(out, pads)
            out_full[..., y0:y0+out.shape[-2], x0:x0+out.shape[-1]] += out
            weight  [..., y0:y0+out.shape[-2], x0:x0+out.shape[-1]] += 1.0

        out_full = out_full / torch.clamp_min(weight, 1e-6)
        return to_image(out_full[0])


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
    parser.add_argument("--out_dir", default="./fused_out", help="Where to write outputs")
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
        ir_img  = load_gray(args.ir)
        vis_img = load_gray(args.vis)
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
            ir_img  = load_gray(ir_p)
            vis_img = load_gray(vis_p)
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
