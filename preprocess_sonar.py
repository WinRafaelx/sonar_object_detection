import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import shutil
import yaml
from glob import glob
from torch.utils.data import Dataset, DataLoader

class SonarAugmentDataset(Dataset):
    """Feeds images for sonar-specific augmentation (Speckle Noise & Shadows)."""
    def __init__(self, file_paths):
        self.paths = file_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return torch.zeros((1, 640, 640)), path, 640, 640
        h_orig, w_orig = img.shape
        img_resized = cv2.resize(img, (640, 640))
        tensor = torch.from_numpy(img_resized).unsqueeze(0).float() / 255.0
        return tensor, path, h_orig, w_orig

def inject_speckle_noise(tensors, intensity=0.1):
    """Adds multiplicative speckle noise: I = I + I * noise."""
    noise = torch.randn_like(tensors) * intensity
    return torch.clamp(tensors + tensors * noise, 0, 1)

def inject_acoustic_shadows(tensors, num_shadows=2):
    """Simulates sonar shadows by darkening random rectangular regions."""
    b, c, h, w = tensors.shape
    for i in range(b):
        for _ in range(num_shadows):
            # Random shadow size and position
            sh = torch.randint(20, 100, (1,)).item()
            sw = torch.randint(50, 200, (1,)).item()
            sy = torch.randint(0, h - sh, (1,)).item()
            sx = torch.randint(0, w - sw, (1,)).item()
            
            # Apply darkening factor (shadow is never pitch black in sonar)
            factor = torch.rand(1).item() * 0.5 + 0.1 # 0.1 to 0.6 brightness
            tensors[i, :, sy:sy+sh, sx:sx+sw] *= factor
    return tensors

def setup_sonar_augmented_dataset(src_dir, intensity=0.15, shadows=2):
    """Creates a sonar-enhanced version of the dataset on GPU."""
    dest_dir = src_dir + "_sonar_aug"
    if os.path.exists(dest_dir):
        print(f"✓ Skipping Sonar Augmentation, '{dest_dir}' already exists.")
        return dest_dir

    print(f"Generating Sonar-Enhanced Dataset (Speckle + Shadows) at {dest_dir}...")
    shutil.copytree(src_dir, dest_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get all images in train folder ONLY (we don't augment val/test to keep evaluation clean)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    paths = []
    train_path = os.path.join(dest_dir, 'train', 'images')
    if os.path.exists(train_path):
        for ext in extensions:
            paths.extend(glob(os.path.join(train_path, ext)))

    if not paths:
        print(f"⚠ Warning: No training images found in {train_path}.")
        return dest_dir

    dataset = SonarAugmentDataset(paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    for batch_tensors, batch_paths, heights, widths in dataloader:
        batch_tensors = batch_tensors.to(device)
        
        with torch.no_grad():
            # Apply augmentations
            aug_tensors = inject_speckle_noise(batch_tensors, intensity=intensity)
            aug_tensors = inject_acoustic_shadows(aug_tensors, num_shadows=shadows)
            
            final_arrays = (aug_tensors.squeeze(1).cpu().numpy() * 255.0).astype(np.uint8)
            
        for i, path in enumerate(batch_paths):
            h, w = heights[i].item(), widths[i].item()
            final_img = cv2.resize(final_arrays[i], (w, h))
            cv2.imwrite(path, final_img)

    # Patch data.yaml
    yaml_path = os.path.join(dest_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        data_config['path'] = os.path.abspath(dest_dir)
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        
    print(f"✓ Sonar-Augmented dataset ready at: {dest_dir}")
    return dest_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    args = parser.parse_args()
    setup_sonar_augmented_dataset(args.src)
