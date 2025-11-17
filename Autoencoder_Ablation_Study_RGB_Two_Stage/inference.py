import os, glob
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import argparse
from model_transport import Transport_Net  # must output 3-channel fused RGB
from model_fusion import Fusion_Net 
from typing import Tuple
import shutil
import numpy as np
import torch.nn.functional as F
import imageio
# ---------- I/O helpers ----------
to_tensor = transforms.ToTensor()      # [0,1], [1,H,W]
to_pil    = transforms.ToPILImage()    # expects [1,H,W] or [H,W]

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

def pad_to_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """Pad [B,C,H,W] so H,W are even. Returns padded tensor and pad tuple (l,r,t,b)."""
    print("Shape before padding:",x.shape)
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

# ---------- core fuse ----------
@torch.no_grad()
def fuse_pair(model: torch.nn.Module, mri: torch.Tensor, pet: torch.Tensor, device="cuda", amp=True):
    """
    mri, pet: [1,H,W] in [0,1] CPU tensors
    returns fused [1,H,W] and aux edge pred [1,H,W]
    """
    # print("Shapes before stacking:", mri.shape, pet.shape)

    # print("Shapes After stacking:", mri.shape, pet.shape)
    x = torch.cat([mri, pet], dim=0).unsqueeze(0).to(device)  # [B=1,2,H,W]
    # pad to even so the stride-2 path is happy
    x, pad = pad_to_even(x)

    with torch.cuda.amp.autocast(enabled=amp and (device.startswith("cuda"))):
        feats, edge_in = model.encode_with_edge(x)            # edge_in not used here
        fused, edge_pred = model.decode_from_feats(feats)  # both in [0,1]

    fused = unpad(fused, pad).clamp(0, 1)
    edge_pred = unpad(edge_in, pad).clamp(0, 1)

    return fused[0], edge_pred[0]  # [1,H,W] each

# ---------- checkpoint loader ----------
def load_model(ckpt_path: str, device: str = "cuda"):
    model = Fusion_Net(input_nc=2, output_nc=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # accept either pure state_dict or {"model": state_dict, ...}
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def load_gray(path: str) -> torch.Tensor:
    """Load single-channel image as float tensor [1,H,W] in [0,1]."""
    img = Image.open(path).convert("L")
    return to_tensor(img)
# ---------------------------------------------------------
# Dataset Loader for Inference (no cropping, no augmentation)
# ---------------------------------------------------------
# PET_Coloured
# PET_Deskulled_Colored  ----- for test dataset
class PairFolderInference(torch.utils.data.Dataset):
    def __init__(self, root: str, size: int = 128, exts=("*.png", "*.jpg", "*.jpeg")):
        self.mri_dir = os.path.join(root, "MRI")
        self.pet_dir = os.path.join(root, "PET_Coloured")
        self.mask_dir = os.path.join(root, "GM")
        self.transport_dir = os.path.join(root, "Transported")  # optional output folder

        def collect(d):
            files = []
            for p in exts:
                files += glob.glob(os.path.join(d, p))
            return files

        mri_files = collect(self.mri_dir)
        pet_files = collect(self.pet_dir)
        mask_files = collect(self.mask_dir)
        transport_files = collect(self.transport_dir)

        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        mri_map = {stem(p): p for p in mri_files}
        pet_map = {stem(p): p for p in pet_files}
        mask_map = {stem(p): p for p in mask_files}
        transport_map = {stem(p): p for p in transport_files}
        
        common = sorted(set(mri_map.keys()) & set(pet_map.keys()) & set(mask_map.keys()))
        self.triplets = [(s, mri_map[s], pet_map[s], mask_map[s]) for s in common]
            
        print(f"Found {len(self.triplets)} triplets in {root} for inference.",transport_files, common)
        if len(transport_files) > 0:
            common = sorted(set(mri_map.keys()) & set(pet_map.keys()) & set(mask_map.keys()) & set(transport_map.keys()))
            self.triplets = [(s, mri_map[s], pet_map[s], mask_map[s], transport_map[s]) for s in common]
        
        print(f"Common, {common} ")
        if not common:
            raise FileNotFoundError(f"No matching triplets in {root}")

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((size, size), interpolation=Image.BICUBIC)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if len(self.triplets[0]) == 5:
            name, mri_path, pet_path, mask_path, transport_path = self.triplets[idx]
        else:
            name, mri_path, pet_path, mask_path = self.triplets[idx]

        # MRI → grayscale, PET → RGB, mask → grayscale
        mri = Image.open(mri_path).convert("L")
        pet = Image.open(pet_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        mri_t = self.to_tensor(mri)          # [1,H,W]
        pet_t = self.to_tensor(pet)          # [3,H,W]
        mask_t = (self.to_tensor(mask) > 0.5).float()  # [1,H,W]      
            
        if len(self.triplets[0]) == 5:
            transport = Image.open(transport_path).convert("RGB")
            # transport_t = self.to_tensor(transport)  # [3,H,W]: 
            transport_t = np.array(transport)  # [3,H,W]:
            return name, mri_t, pet_t, mask_t, transport_t
        else:  
            return name, mri_t, pet_t, mask_t


folder_path = '/data1/yasir/test/Transported'
# Check if the folder exists
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    # Delete the folder and its contents
    shutil.rmtree(folder_path)
    print(f"The folder '{folder_path}' has been deleted.")
# ---------------------------------------------------------
# Inference Function
# ---------------------------------------------------------
@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fused_output_dir, exist_ok=True)

    # Load dataset
    dataset = PairFolderInference(args.data_root, size=args.size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = Transport_Net().to(device)
    ckpt = torch.load(args.checkpoint_transport, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint_transport}")

    # Run inference
    for name, mri, pet, mask in tqdm(loader, desc="Inference"):
        mri, pet, mask = mri.to(device), pet.to(device), mask.to(device)

        # Forward pass
        fused, edge_pred = model(pet, mask)
        fused = torch.clamp(fused, 0, 1)  # ensure valid range

        # Save fused output (3-channel RGB)
        save_path = os.path.join(args.output_dir, f"{name[0]}.png")
        save_image(fused, save_path)

        # Create visualization panel
        # mri_rgb = mri.repeat(1, 3, 1, 1)  # replicate grayscale MRI to 3-channel for side-by-side view
        # mask_rgb = mask.repeat(1, 3, 1, 1)
        # panel = torch.cat([mri_rgb, pet, mask_rgb, fused], dim=-1)
        # save_image(panel, os.path.join(args.output_dir, f"{name[0]}_panel.png"))

    # collect matching pairs by filename stem
    mri_dir = os.path.join(args.data_root, "MRI")
    pet_dir = os.path.join(args.data_root, "Transported")
    mri_files, pet_files = [], []
    for p in args.exts:
        mri_files += glob.glob(os.path.join(mri_dir, p))
        pet_files += glob.glob(os.path.join(pet_dir, p))

    # mri_files = os.listdir(mri_dir)
    # pet_files = os.listdir(pet_dir)
    print(f"Found {len(mri_files)} MRI and {len(pet_files)} PET files.",mri_files)
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    m_map = {stem(p): p for p in mri_files}
    p_map = {stem(p): p for p in pet_files}
    common = sorted(set(m_map.keys()) & set(p_map.keys()))
    if not common:
        raise FileNotFoundError(f"No matching stems under {mri_dir} and {pet_dir}")

    print(f"Found {len(common)} pairs. Loading model from {args.checkpoint_fusion} ...")
    model = load_model(args.checkpoint_fusion, device=args.device)

    # run
    for s in common:
        mri_path, pet_path = m_map[s], p_map[s]
        mri = load_gray(mri_path)
        pet = load_gray(pet_path)
        
        pet_img_rgb = Image.open(pet_path).convert("RGB")
        print("Shape of PET image RGB:",np.array(pet_img_rgb).shape)
        pet_img_y, pet_img_cb, pet_img_cr = rgb_to_ycbcr(pet_img_rgb);   
        pet_img_y = to_tensor(pet_img_y).float()
        
        # pet_img_y = pet_img_y/255.0
        # mri = mri/255.0
        
        fused, edge = fuse_pair(model, mri, pet , device=args.device, amp=args.amp)
        out_path = os.path.join(args.fused_output_dir, f"{s}.png")
        
        # # save_gray(fused, out_path)
        fuseImage = fused*450;   #400 for edge guided, 500 for edge dominant
        fuseImage = fuseImage.squeeze().detach().cpu().numpy()
        # # print(f"Fusing {fuseImage.shape} and {pet_img_cb.shape} and {pet_img_cr.shape}...")
        
        fuseImage = ycbcr_to_rgb(fuseImage, pet_img_cb, pet_img_cr)
        
        # # file_name = 'fuse'+str(index) + '.png'
        output_path = out_path 

        # save_gray(fused, out_path)
        
        # Normalize and convert to uint8
        # fuseImage_uint8 = (fuseImage - np.min(fuseImage)) / (np.max(fuseImage) - np.min(fuseImage))  # Normalize to 0-1
        # fuseImage_uint8 = (fuseImage_uint8 * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        imageio.imwrite(output_path, fuseImage)
    
    
    print(f"Inference complete. Results saved in {args.output_dir}")


# ---------------------------------------------------------
# Entry Point
# --------------------------------------------------------- 
# /data1/yasir/test
# /mnt/yasir/PET_Dataset_Processed/Separated_Centered_PET_MRI_Updated_Dataset_Splitted/test

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PET-MRI Fusion Inference (RGB PET, grayscale MRI)")
    parser.add_argument("--data_root", type=str, default="/data1/yasir/test",
                        help="Path to dataset folder with MRI, PET, and GM subfolders")
    parser.add_argument("--checkpoint_transport", type=str, default="checkpoints_rgbfusion/fusion_final_rgb.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--checkpoint_fusion", type=str, default="checkpoints_edgeguided/edgeguided_final.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="/data1/yasir/test/Transported",
                        help="Directory to save fused results")
    parser.add_argument("--fused_output_dir", type=str, default="/data1/yasir/test/Fused_Images",
                        help="Directory to save fused results")
    parser.add_argument("--size", type=int, default=128,
                        help="Optional resize (if model trained with fixed size)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--exts", type=str, nargs="+", default=["*.png","*.jpg","*.jpeg"],
                    help="File patterns to match")
    args = parser.parse_args()

    run_inference(args)
