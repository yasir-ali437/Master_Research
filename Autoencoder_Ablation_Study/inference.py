import os, glob
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import argparse
from model import Fusion_net  # same model class used in training

# ---------------------------------------------------------
# Dataset Loader for inference (no cropping, no augmentation)
# ---------------------------------------------------------
class PairFolderInference(torch.utils.data.Dataset):
    def __init__(self, root: str, size: int = 128, exts=("*.png","*.jpg","*.jpeg")):
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
        if not common:
            raise FileNotFoundError(f"No matching triplets in {root}")

        self.triplets = [(s, mri_map[s], pet_map[s], mask_map[s]) for s in common]
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((size, size), interpolation=Image.BICUBIC)

    def __len__(self): return len(self.triplets)

    def __getitem__(self, idx):
        name, mri_path, pet_path, mask_path = self.triplets[idx]
        mri = Image.open(mri_path).convert("L")
        pet = Image.open(pet_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # mri = self.resize(mri)
        # pet = self.resize(pet)
        # mask = self.resize(mask)

        mri_t = self.to_tensor(mri)
        pet_t = self.to_tensor(pet)
        mask_t = (self.to_tensor(mask) > 0.5).float()

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
        fused, edge_pred = model(mri, pet, mask)
        fused = torch.clamp(fused, 0, 1)

        # Save fused PET->MRI output
        save_path = os.path.join(args.output_dir, f"{name[0]}_fused.png")
        save_image(fused, save_path)

        # Optional: save visualization panel
        panel = torch.cat([mri, pet, mask, fused], dim=-1)
        save_image(panel, os.path.join(args.output_dir, f"{name[0]}_panel.png"))

    print(f"Inference complete. Results saved in {args.output_dir}")


#/mnt/yasir/PET_Dataset_Processed/Separated_Centered_PET_MRI_Updated_Dataset_Splitted/test
# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("PET-MRI Fusion Inference")
    parser.add_argument("--data_root", type=str, default="/data1/yasir/test",
                        help="Path to test dataset folder containing MRI, PET_Deskulled, and GM subfolders")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_edgeguided/edgeguided_final.pth",
                        help="Path to trained model checkpoint (e.g., edgeguided_final.pth)")
    parser.add_argument("--output_dir", type=str, default="./inference_results_shrinked",
                        help="Directory to save fused outputs")
    parser.add_argument("--size", type=int, default=128, help="Resize images to this size before inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_inference(args)
