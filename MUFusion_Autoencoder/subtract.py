import os
import cv2
import numpy as np

# Set your folder paths
folder1 = "/data1/yasir/test/pet_gray"
folder2 = "/home/yasir/Summer_Research/MUFusion/medical/output_folder_original_mri"
output_folder = "/home/yasir/Summer_Research/MUFusion/medical/output_folder_original_mri_subtracted"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of binary mask files
mask_filenames = os.listdir(folder1)

# Go through each mask
for mask_filename in mask_filenames:
    mask_name, _ = os.path.splitext(mask_filename)
    
    # Extract prefix/subpart (e.g., P01_Pair1)
    parts = mask_name.split('_')
    if len(parts) < 1:
        print(f"Skipping {mask_filename}, cannot extract prefix.")
        continue
    subpart = '_'.join(parts[:2])

    # Find matching MRI file in folder2
    matching_mri_file = None
    for mri_file in os.listdir(folder2):
        if subpart in mri_file:
            matching_mri_file = mri_file
            break

    if not matching_mri_file:
        print(f"No match found in MRI folder for {mask_filename} (subpart: {subpart})")
        continue

    path1 = os.path.join(folder1, mask_filename)
    path2 = os.path.join(folder2, matching_mri_file)

    # Read images in grayscale
    mask = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    mri = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    if mask is None or mri is None:
        print(f"Error reading images for {mask_filename}")
        continue

    # Apply mask: retain MRI pixels where mask is non-zero
    masked = np.where(mask != 0, mri, 0).astype(np.uint8)

    # Save the result with the same name as mask (or use MRI name if preferred)
    output_path = os.path.join(output_folder, mask_filename)
    cv2.imwrite(output_path, masked)

print("Masked MRI image generation complete.")
