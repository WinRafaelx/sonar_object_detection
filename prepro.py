import os
import cv2
import glob
import random
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow
batch_size = 16

# ==========================================
# 1. CLEANING PIPELINE (Static - Runs Once)
# ==========================================
def apply_cleaning_pipeline(image_path):
    """
    Overwrites the image with a Cleaned version (Speckle + CLAHE + TVG).
    We do this ONCE so the model doesn't have to calculate it 500 times.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    # 1. Denoise (Speckle Reduction)
    img = cv2.medianBlur(img, 3)

    # 2. Contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 3. TVG (Linear Gain)
    rows, cols = img.shape
    center_x = cols // 2
    x_indices = np.arange(cols)
    dist = np.abs(x_indices - center_x) / (cols / 2)
    gain_map = 1.0 + (dist * 0.2)
    gain_matrix = np.tile(gain_map, (rows, 1))

    img = img.astype(np.float32) * gain_matrix
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Overwrite file
    cv2.imwrite(image_path, img)

def clean_dataset(dataset_path):
    print(f"🚀 Cleaning images in {dataset_path}...")
    # Find all images (Train, Valid, Test)
    images = glob.glob(os.path.join(dataset_path, "**", "images", "*.*"), recursive=True)

    count = 0
    for img_file in images:
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            apply_cleaning_pipeline(img_file)
            count += 1
            if count % 100 == 0: print(f"   Cleaned {count} images...")

    print(f"✅ Finished cleaning {count} images.")

# ==========================================
# NEW: 1.5 OFFLINE AUGMENTATION (Dataset Multiplier)
# ==========================================
def offline_augment(dataset_path):
    print("🧬 Multiplying training data...")
    # Only augment the TRAINING data, leave valid/test alone!
    train_images = glob.glob(os.path.join(dataset_path, "train", "images", "*.*"))

    # Setup our remix pipeline (Horizontal Flip + Brightness/Contrast)
    # BboxParams keeps the YOLO labels glued to the object when we move it
    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.GaussNoise(p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

    count = 0
    for img_path in train_images:
        # Find matching label file
        label_path = img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
        if not os.path.exists(label_path): continue

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations likes RGB

        # Read YOLO labels (class, x_center, y_center, width, height)
        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])

        # Apply transformation
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except ValueError:
            continue # Skip if a bounding box goes out of bounds

        # Save the new remixed image
        new_img_path = img_path.rsplit('.', 1)[0] + "_aug1.jpg"
        cv2.imwrite(new_img_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))

        # Save the new remixed label
        new_label_path = label_path.rsplit('.', 1)[0] + "_aug1.txt"
        with open(new_label_path, 'w') as f:
            for bbox, cls in zip(transformed['bboxes'], transformed['class_labels']):
                f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

        count += 1

    print(f"✅ Created {count} new augmented images!")

# ==========================================
# NEW: 1.6 VISUALIZATION (The Vibe Check)
# ==========================================
def visualize_samples(dataset_path, num_samples=4):
    print("👀 Running a vibe check on the dataset...")
    images = glob.glob(os.path.join(dataset_path, "train", "images", "*.*"))
    samples = random.sample(images, min(num_samples, len(images)))

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        label_path = img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # Math to convert YOLO format back to pixel coordinates for drawing
                    cls, x_c, y_c, bw, bh = map(float, line.strip().split())
                    x_min = int((x_c - bw / 2) * w)
                    y_min = int((y_c - bh / 2) * h)
                    x_max = int((x_c + bw / 2) * w)
                    y_max = int((y_c + bh / 2) * h)
                    # Draw a bright green box
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(img_path)[:15])
    plt.show()

# ==========================================
# 2. MAIN EXECUTION (Updated)
# ==========================================
def main():
    # 1. Download Data
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")) 
    project = rf.workspace("object-detect-ury2h").project("sonar_detect")
    version = project.version(1)
    dataset = version.download("yolov11")

    # 2. Clean Data (One-time physics correction)
    clean_dataset(dataset.location) # (Assuming you still have this from before)

    # 3. Multiply Data (Offline Augmentation)
    offline_augment(dataset.location)

    # 4. Vibe Check (Look at the images before training)
    # visualize_samples(dataset.location)

    # 6. Train with Native Augmentation
    model = YOLO("yolo11m.pt")
    results = model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        imgsz=640,
        epochs=500,
        patience=50,
        batch=batch_size,
        project="runs/train",
        name="yolo11m_sonar_native_aug",
        verbose=True,
        plots=True,

        # --- NATIVE AUGMENTATION SETTINGS ---
        # These happen in memory during training
        degrees=10.0,      # Rotate image +/- 10 degrees randomly
        fliplr=0.5,        # Flip horizontal 50% of the time
        # mosaic=1.0,        # Mosaic (stitch 4 images) 100% of the time (Very good for sonar)
        mixup=0.1,         # Mixup (blend 2 images) 10% chance
        scale=0.5,         # Zoom in/out by +/- 50%

        # Turn off unrelated augs
        flipud=0.0,        # Don't flip upside down (violates shadow physics)
        hsv_h=0.0,         # Don't change Hue (it's grayscale anyway)
        hsv_s=0.0,         # Don't change Saturation
        hsv_v=0.4          # Random brightness (Value) changes are okay
    )

if __name__ == '__main__':
    main()