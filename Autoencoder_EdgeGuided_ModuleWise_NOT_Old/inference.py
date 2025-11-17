import os, glob, argparse
from typing import Tuple
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from model import Fusion_net  # your edge-guided net
import numpy as np
import imageio

# ---------- I/O helpers ----------
to_tensor = transforms.ToTensor()      # [0,1], [1,H,W]
to_pil    = transforms.ToPILImage()    # expects [1,H,W] or [H,W]

def load_gray(path: str) -> torch.Tensor:
    """Load single-channel image as float tensor [1,H,W] in [0,1]."""
    img = Image.open(path).convert("L")
    return to_tensor(img)

#PIL Image type "RGB mode"    
def rgb_to_ycbcr(image):
    rgb_array = np.array(image)

    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])

    ycbcr_array = np.dot(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, 0]
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]
    
    y_channel = np.clip(y_channel, 0, 255)
    return y_channel, cb_channel, cr_channel

def ycbcr_to_rgb(y, cb, cr):
    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    transform_matrix = np.array([[1, 0, 1.402],
                                 [1, -0.344136, -0.714136],
                                 [1, 1.772, 0]])

    rgb_array = np.dot(ycbcr_array, transform_matrix.T)

    rgb_array = np.clip(rgb_array, 0, 255)

    rgb_array = np.round(rgb_array).astype(np.uint8)

    rgb_image = Image.fromarray(rgb_array, mode='RGB')

    return rgb_image

def save_gray(t: torch.Tensor, path: str):
    """Save [1,H,W] or [H,W] in [0,1] to path as 8-bit PNG."""
    # Ensure CPU float32 before clamping/ToPIL
    t = t.detach()
    if t.dim() == 4:  # [B,1,H,W] -> take first
        t = t[0]
    if t.dim() == 2:
        t = t.unsqueeze(0)  # -> [1,H,W]
    # Cast to float32 on CPU before clamp/ToPIL
    t = t.to(device="cpu", dtype=torch.float32).clamp(0, 1)
    img = to_pil(t)
    img.save(path)

def pad_to_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """Pad [B,C,H,W] so H,W are even. Returns padded tensor and pad tuple (l,r,t,b)."""
    _, _, H, W = x.shape
    pad_r = (2 - W % 2) % 2
    pad_b = (2 - H % 2) % 2
    pad = (0, pad_r, 0, pad_b)  # (left,right,top,bottom) but F.pad uses (l,r,t,b)
    if pad_r or pad_b:
        x = F.pad(x, pad, mode="reflect")
    return x, pad

def unpad(x: torch.Tensor, pad: Tuple[int,int,int,int]) -> torch.Tensor:
    l, r, t, b = 0, pad[1], 0, pad[3]
    if r > 0:
        x = x[:, :, :, :-r]
    if b > 0:
        x = x[:, :, :-b, :]
    return x

# ---------- checkpoint loader ----------
def load_model(ckpt_path: str, device: str = "cuda"):
    model = Fusion_net(input_nc=3, output_nc=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # accept either pure state_dict or {"model": state_dict, ...}
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ---------- core fuse ----------
@torch.no_grad()
def fuse_pair(model: torch.nn.Module, mri: torch.Tensor, pet: torch.Tensor, mask: torch.Tensor, device="cuda", amp=True):
    """
    mri, pet: [1,H,W] in [0,1] CPU tensors
    returns fused [1,H,W] and aux edge pred [1,H,W]
    """
    # stack to [1,2,H,W]
    x = torch.cat([mri, pet, mask], dim=0).unsqueeze(0).to(device)  # [B=1,2,H,W]
    # pad to even so the stride-2 path is happy
    x, pad = pad_to_even(x)

    with torch.cuda.amp.autocast(enabled=amp and (device.startswith("cuda"))):
        feats, edge_in = model.encode_with_edge(x)            # edge_in not used here
        fused, edge_pred = model.decode_from_feats(feats)  # both in [0,1]

    fused = unpad(fused, pad).clamp(0, 1)
    edge_pred = unpad(edge_in, pad).clamp(0, 1)

    return fused[0], edge_pred[0]  # [1,H,W] each

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Edge-guided MRI+PET fusion inference")
    ap.add_argument("--data_root", type=str, default="/data1/yasir/test", required=False,
                    help="Folder containing mri/ and pet/ subfolders")
    ap.add_argument("--ckpt", type=str, default="checkpoints_edgeguided/edgeguided_final.pth", required=False,
                    help="Path to trained checkpoint (.pth or .pt)")
    ap.add_argument("--out_dir", type=str, default="./fused_out1",
                    help="Where to save fused images")
    ap.add_argument("--save_edges", action="store_true",
                    help="Also save predicted edge maps for inspection")
    ap.add_argument("--exts", type=str, nargs="+", default=["*.png","*.jpg","*.jpeg"],
                    help="File patterns to match")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_edges:
        os.makedirs(os.path.join(args.out_dir, "edges"), exist_ok=True)

    # collect matching pairs by filename stem
    mri_dir = os.path.join(args.data_root, "MRI")
    pet_dir = os.path.join(args.data_root, "PET_Coloured")
    mask_dir = os.path.join(args.data_root, "GM")
    mri_files, pet_files, mask_files = [], [], []
    
    for p in args.exts:
        mri_files += glob.glob(os.path.join(mri_dir, p))
        pet_files += glob.glob(os.path.join(pet_dir, p))
        mask_files += glob.glob(os.path.join(mask_dir, p))

    # mri_files = os.listdir(mri_dir)
    # pet_files = os.listdir(pet_dir)
    print(f"Found {len(mri_files)} MRI and {len(pet_files)} PET files.",mri_files)
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    m_map = {stem(p): p for p in mri_files}
    p_map = {stem(p): p for p in pet_files}
    mask_map = {stem(p): p for p in mask_files}
    common = sorted(set(m_map.keys()) & set(p_map.keys()) & set(mask_map.keys()))
    if not common:
        raise FileNotFoundError(f"No matching stems under {mri_dir} and {pet_dir}")

    print(f"Found {len(common)} pairs. Loading model from {args.ckpt} ...")
    model = load_model(args.ckpt, device=args.device)

    # run
    for s in common:
        mri_path, pet_path, mask_path = m_map[s], p_map[s], mask_map[s]
        mri = load_gray(mri_path)
        pet = load_gray(pet_path)
        mask = load_gray(mask_path)
        
        pet_img_rgb = Image.open(pet_path).convert("RGB");
        pet_img_y, pet_img_cb, pet_img_cr = rgb_to_ycbcr(pet_img_rgb);   
        pet_img_y = to_tensor(pet_img_y).float()
        
        # pet_img_y = pet_img_y/255.0
        # mri = mri/255.0
        
        fused, edge = fuse_pair(model, mri, pet, mask, device=args.device, amp=args.amp)
        out_path = os.path.join(args.out_dir, f"{s}_fused.png")
        
        # # save_gray(fused, out_path)
        fuseImage = fused*400;   #400 for edge guided, 500 for edge dominant
        fuseImage = fuseImage.squeeze().detach().cpu().numpy()
        # # print(f"Fusing {fuseImage.shape} and {pet_img_cb.shape} and {pet_img_cr.shape}...")
        
        fuseImage = ycbcr_to_rgb(fuseImage, pet_img_cb, pet_img_cr);
        
        # # file_name = 'fuse'+str(index) + '.png'
        output_path = out_path 

        # save_gray(fused, out_path)
        imageio.imwrite(output_path, fuseImage)

        if args.save_edges:
            edge_path = os.path.join(args.out_dir, "edges", f"{s}_edge.png")
            save_gray(edge, edge_path)

        print(f"Saved: {out_path}" + (" (with edge map)" if args.save_edges else ""))

    print("Done.")


# if __name__ == "__main__":
    
main()
