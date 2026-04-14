import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import shutil
import yaml
from glob import glob
from torch.utils.data import Dataset, DataLoader
from pytorch_wavelets import DWTForward

class SonarDataset(Dataset):
    """Feeds the GPU images in batches to stop it from waiting on the hard drive."""
    def __init__(self, file_paths):
        self.paths = file_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (640, 640))
        tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
        return tensor, path, img.shape[0], img.shape[1]

def setup_dwt_dataset(src_dir):
    """Max performant GPU-accelerated DWT caching with DataLoader batching."""
    dest_dir = src_dir + "_dwt"
    if os.path.exists(dest_dir):
        print(f"✓ Skipping DWT, '{dest_dir}' already exists.")
        return dest_dir

    print(f"Applying Batched GPU-accelerated 2D-DWT to {dest_dir}...")
    shutil.copytree(src_dir, dest_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)
    
    # Get all images in train, val, test folders
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    paths = []
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dest_dir, split, 'images')
        if os.path.exists(split_path):
            for ext in extensions:
                paths.extend(glob(os.path.join(split_path, ext)))

    if not paths:
        print(f"⚠ Warning: No images found in standard YOLO structure in {dest_dir}. Searching recursively...")
        for ext in extensions:
            paths.extend(glob(os.path.join(dest_dir, "**", ext), recursive=True))

    dataset = SonarDataset(paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    for batch_tensors, batch_paths, heights, widths in dataloader:
        batch_tensors = batch_tensors.to(device)
        
        with torch.no_grad():
            Yl, _ = dwt(batch_tensors)
            LL = Yl.squeeze(1) 
            
            # Fast GPU normalization across batch
            LL_min = LL.amin(dim=(1, 2), keepdim=True)
            LL_max = LL.amax(dim=(1, 2), keepdim=True)
            LL_norm = (LL - LL_min) / (LL_max - LL_min + 1e-8) * 255.0
            
            # Resize on GPU
            final_tensors = F.interpolate(
                LL_norm.unsqueeze(1), 
                size=(640, 640), 
                mode='bilinear', 
                align_corners=False
            )
            final_arrays = final_tensors.squeeze(1).cpu().numpy().astype(np.uint8)
            
        for i, path in enumerate(batch_paths):
            h, w = heights[i].item(), widths[i].item()
            final_img = cv2.resize(final_arrays[i], (w, h))
            cv2.imwrite(path, final_img)

    # Patch the new data.yaml with the new absolute path
    yaml_path = os.path.join(dest_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        data_config['path'] = os.path.abspath(dest_dir)
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        
    print(f"✓ DWT dataset ready at: {dest_dir}")
    return dest_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to source dataset")
    args = parser.parse_args()
    setup_dwt_dataset(args.src)
