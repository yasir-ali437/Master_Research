import os, glob
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import argparse
from model import Fusion_net  # must output 3-channel fused RGB

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

        self.triplets = [(s, mri_map[s], pet_map[s], mask_map[s]) for s in common]
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((size, size), interpolation=Image.BICUBIC)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        name, mri_path, pet_path, mask_path = self.triplets[idx]

        # MRI → grayscale, PET → RGB, mask → grayscale
        mri = Image.open(mri_path).convert("L")
        pet = Image.open(pet_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Optionally resize
        # mri, pet, mask = self.resize(mri), self.resize(pet), self.resize(mask)

        mri_t = self.to_tensor(mri)          # [1,H,W]
        pet_t = self.to_tensor(pet)          # [3,H,W]
        mask_t = (self.to_tensor(mask) > 0.5).float()  # [1,H,W]

        return name, mri_t, pet_t, mask_t


# ---------------------------------------------------------
# Inference Function
# ---------------------------------------------------------
@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = PairFolderInference(args.data_root, size=args.size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = Fusion_net().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint}")

    # Run inference
    for name, mri, pet, mask in tqdm(loader, desc="Inference"):
        mri, pet, mask = mri.to(device), pet.to(device), mask.to(device)

        # Forward pass
        fused, edge_pred = model(mri, pet, mask)
        fused = torch.clamp(fused, 0, 1)  # ensure valid range

        # Save fused output (3-channel RGB)
        save_path = os.path.join(args.output_dir, f"{name[0]}_fused.png")
        save_image(fused, save_path)

        # Create visualization panel
        mri_rgb = mri.repeat(1, 3, 1, 1)  # replicate grayscale MRI to 3-channel for side-by-side view
        mask_rgb = mask.repeat(1, 3, 1, 1)
        panel = torch.cat([mri_rgb, pet, mask_rgb, fused], dim=-1)
        save_image(panel, os.path.join(args.output_dir, f"{name[0]}_panel.png"))

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
    parser.add_argument("--checkpoint", type=str, default="checkpoints_rgbfusion/fusion_final_rgb.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./inference_results_rgb",
                        help="Directory to save fused results")
    parser.add_argument("--size", type=int, default=128,
                        help="Optional resize (if model trained with fixed size)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_inference(args)
