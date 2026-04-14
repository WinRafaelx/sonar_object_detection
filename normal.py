import os
import sys
import torch
import torch.nn as nn
import yaml
import inspect
import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from constant import batch_size as default_batch_size
import argparse

# Import custom modules
from custom_modules import SonarSPDConv, CoordAtt, CBAM, EMA, BiFPN_Concat2
from architecture_configs import SPDCONV_YOLO_YAML, ATTN_SPDCONV_YOLO_YAML
from preprocess_dwt import setup_dwt_dataset
from preprocess_sonar import setup_sonar_augmented_dataset

# 1. LOAD ENV & KAGGLE AUTH
load_dotenv()
if 'KAGGLE_API_TOKEN' in os.environ and 'KAGGLE_KEY' not in os.environ:
    os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']
import kaggle

# --- 2. MONKEY-PATCHING ---
def patch_yolo_parser():
    """Register custom modules and patch the YOLO model parser."""
    setattr(tasks, 'SonarSPDConv', SonarSPDConv)
    setattr(tasks, 'CoordAtt', CoordAtt)
    setattr(tasks, 'CBAM', CBAM)
    setattr(tasks, 'EMA', EMA)
    setattr(tasks, 'BiFPN_Concat2', BiFPN_Concat2)
    
    try:
        source = inspect.getsource(tasks.parse_model)
        if "SonarSPDConv" not in source:
            # 1. Add single-input modules that follow (c1, c2, n) signature to base_modules AND repeat_modules
            # We try multiple anchors to be robust across different Ultralytics versions
            patched = False
            for anchor in ["SPDConv,", "C2fCIB,", "C2f,", "Conv,"]:
                if anchor in source:
                    # Replace the first 2 occurrences (base_modules and repeat_modules)
                    source = source.replace(anchor, f"{anchor} SonarSPDConv, CoordAtt, CBAM,", 2)
                    patched = True
                    break
            
            if not patched:
                print("⚠ Warning: Could not find a suitable anchor in parse_model to patch custom modules.")
            
            # 2. Add BiFPN_Concat2 branch (handles multiple inputs)
            concat_branch = "elif m is Concat:\n            c2 = sum(ch[x] for x in f)"
            if concat_branch in source:
                bifpn_branch = concat_branch + "\n        elif m is BiFPN_Concat2:\n            c2 = args[0]\n            if c2 != nc:\n                c2 = make_divisible(min(c2, max_channels) * width, 8)\n            args = [[ch[x] for x in f], c2]"
                source = source.replace(concat_branch, bifpn_branch)
            
            # 3. Handle EMA specifically because its signature is (channels, factor)
            ema_branch = "\n        elif m is EMA:\n            c2 = ch[f]\n            args = [c2, *args]"
            source = source.replace("else:\n            c2 = ch[f]", ema_branch + "\n        else:\n            c2 = ch[f]")

            exec_globals = tasks.__dict__.copy()
            exec_globals.update({
                'SonarSPDConv': SonarSPDConv,
                'CoordAtt': CoordAtt,
                'CBAM': CBAM,
                'EMA': EMA,
                'BiFPN_Concat2': BiFPN_Concat2,
                'torch': torch,
                'nn': nn,
                'inspect': inspect,
                'ast': __import__('ast'),
                'contextlib': __import__('contextlib'),
            })
            exec(source, exec_globals)
            tasks.parse_model = exec_globals['parse_model']
            print("✓ Successfully patched YOLO model parser with BiFPN and Attention support.")
    except Exception as e:
        print(f"⚠ Warning: Could not patch model parser: {e}")

patch_yolo_parser()

# --- 3. HELPERS ---
def patch_data_yaml(yaml_path, dataset_dir):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    data['path'] = os.path.abspath(dataset_dir)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return os.path.abspath(yaml_path)

def get_metrics(results):
    box = results.box
    return {
        "mAP50": box.map50,
        "mAP50-95": box.map,
        "Precision": box.mp,
        "Recall": box.mr
    }

# --- 4. MAIN EXPERIMENTATION ENGINE ---
def run_experiment(args):
    mode = args.mode
    use_dwt = args.dwt
    print(f"\n{'='*20} STARTING EXPERIMENT: {mode.upper()} {' (DWT=' + str(use_dwt) + ')'} {'='*20}")
    
    # Load hyperparams
    with open('best_hyperparameters.yaml', 'r') as f:
        best_hyp = yaml.safe_load(f)

    # Dataset Setup
    base_data_dir = os.path.abspath('./data')
    sss_dir = os.path.join(base_data_dir, 'sampled_yolo_dataset')
    
    if not os.path.exists(sss_dir):
        print("Downloading SSS dataset...")
        kaggle.api.dataset_download_files('mawins/sample-sss-img', path=base_data_dir, unzip=True)
    
    # Apply DWT if requested
    if use_dwt:
        sss_dir = setup_dwt_dataset(sss_dir)
    elif args.sonar_aug:
        sss_dir = setup_sonar_augmented_dataset(sss_dir)

    sss_yaml = patch_data_yaml(os.path.join(sss_dir, 'data.yaml'), sss_dir)
    
    # Architecture & Weights Setup
    model_yaml_path = f"tmp_{mode}.yaml"
    pretrained_weights = "yolo11n.pt"
    
    with open(sss_yaml, 'r') as f:
        nc = yaml.safe_load(f)['nc']

    if mode == "standard":
        model = YOLO(pretrained_weights)
    elif mode == "spdconv":
        with open(model_yaml_path, "w") as f:
            f.write(f"nc: {nc}\n" + SPDCONV_YOLO_YAML)
        model = YOLO(model_yaml_path).load(pretrained_weights)
    elif mode == "hybrid":
        with open(model_yaml_path, "w") as f:
            f.write(f"nc: {nc}\n" + ATTN_SPDCONV_YOLO_YAML)
        model = YOLO(model_yaml_path).load(pretrained_weights)
    else:
        raise ValueError("Invalid mode")

    # Training
    results = model.train(
        data=sss_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        optimizer=args.optimizer,
        device=args.device,
        project="runs/experiments",
        name=f"exp_{mode}_{'dwt' if use_dwt else ('sonar_aug' if args.sonar_aug else 'normal')}",
        verbose=True,
        **best_hyp
    )
    
    metrics = get_metrics(results)
    metrics["mode"] = mode
    metrics["use_dwt"] = use_dwt
    metrics["sonar_aug"] = args.sonar_aug
    return metrics

if __name__ == "__main__":
    # Determine the most sensible default device
    default_device = "0" if torch.cuda.is_available() else "cpu"
    
    parser = argparse.ArgumentParser()
    # High-level Experiment Config
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "spdconv", "hybrid"])
    parser.add_argument("--dwt", action="store_true", help="Use DWT preprocessed dataset")
    parser.add_argument("--sonar_aug", action="store_true", help="Use Sonar-Augmented (Speckle/Shadows) dataset")
    
    # Training Parameters
    parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--batch", type=int, default=default_batch_size, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer to use (SGD, Adam, AdamW, etc.)")
    parser.add_argument("--device", type=str, default=default_device, help="CUDA device index (e.g. 0) or 'cpu'")
    
    args = parser.parse_args()

    metrics = run_experiment(args)
    
    # Save results summary
    summary_file = "experiment_results.csv"
    df_new = pd.DataFrame([metrics])
    if os.path.exists(summary_file):
        df_old = pd.read_csv(summary_file)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    
    df_final.to_csv(summary_file, index=False)
    print(f"\n✓ Experiment {args.mode} complete. Results appended to {summary_file}")
    print(df_new)
