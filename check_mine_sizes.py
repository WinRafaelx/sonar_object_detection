import os
import yaml
from glob import glob

def main():
    label_path = "data/Combined_Dataset/val/labels/*.txt"
    mine_class = 2
    
    label_files = glob(label_path)
    if not label_files:
        print("No labels found.")
        return
        
    mine_widths = []
    mine_heights = []
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if int(parts[0]) == mine_class:
                    w = float(parts[3])
                    h = float(parts[4])
                    mine_widths.append(w)
                    mine_heights.append(h)
    
    if not mine_widths:
        print("No mines found in the labels.")
        return
        
    avg_w = sum(mine_widths) / len(mine_widths)
    avg_h = sum(mine_heights) / len(mine_heights)
    min_w = min(mine_widths)
    max_w = max(mine_widths)
    
    print(f"Mines found: {len(mine_widths)}")
    print(f"Average size (normalized): {avg_w:.4f} x {avg_h:.4f}")
    print(f"Min size (normalized): {min_w:.4f}")
    print(f"Max size (normalized): {max_w:.4f}")
    
    # In 640x640 resolution
    print(f"Average size in pixels (640x640): {avg_w*640:.1f} x {avg_h*640:.1f}")
    print(f"Min width in pixels (640x640): {min_w*640:.1f}")

if __name__ == "__main__":
    main()
