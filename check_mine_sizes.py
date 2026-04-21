import os
import yaml
from glob import glob

def main():
    label_path = "data/combined_data/valid/labels/*.txt"
    # mine_class = 2
    
    label_files = glob(label_path)
    if not label_files:
        print(f"No labels found in {label_path}")
        # Try another common path
        label_path = "data/combined_data/labels/valid/*.txt"
        label_files = glob(label_path)
        if not label_files:
            print(f"No labels found in {label_path}")
            return
    
    print(f"Found {len(label_files)} label files.")
    
    widths = []
    heights = []
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    w = float(parts[3])
                    h = float(parts[4])
                    widths.append(w)
                    heights.append(h)
    
    if not widths:
        print("No objects found in the labels.")
        return
        
    avg_w = sum(widths) / len(widths)
    avg_h = sum(heights) / len(heights)
    min_w = min(widths)
    max_w = max(widths)
    
    print(f"Objects found: {len(widths)}")
    print(f"Average size (normalized): {avg_w:.4f} x {avg_h:.4f}")
    print(f"Min size (normalized): {min_w:.4f}")
    print(f"Max size (normalized): {max_w:.4f}")
    
    # In 640x640 resolution
    print(f"Average size in pixels (640x640): {avg_w*640:.1f} x {avg_h*640:.1f}")
    print(f"Min width in pixels (640x640): {min_w*640:.1f}")
    print(f"Small objects (< 32px): {len([w for w in widths if w*640 < 32])} ({len([w for w in widths if w*640 < 32])/len(widths)*100:.1f}%)")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
