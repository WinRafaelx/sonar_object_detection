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
import contextlib
import shutil

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
            patched = False
            for anchor in ["SPDConv,", "C2fCIB,", "C2f,", "Conv,"]:
                if anchor in source:
                    source = source.replace(anchor, f"{anchor} SonarSPDConv, CoordAtt, CBAM,", 2)
                    patched = True
                    break
            
            concat_branch = "elif m is Concat:\n            c2 = sum(ch[x] for x in f)"
            if concat_branch in source:
                bifpn_branch = concat_branch + "\n        elif m is BiFPN_Concat2:\n            c2 = args[0]\n            if c2 != nc:\n                c2 = make_divisible(min(c2, max_channels) * width, 8)\n            args = [[ch[x] for x in f], c2]"
                source = source.replace(concat_branch, bifpn_branch)
            
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
                'contextlib': contextlib,
            })
            exec(source, exec_globals)
            tasks.parse_model = exec_globals['parse_model']
            print("✓ Successfully patched YOLO model parser.")
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
    
    with open('best_hyperparameters.yaml', 'r') as f:
        best_hyp = yaml.safe_load(f)
    
    if args.box is not None: best_hyp['box'] = args.box
    if args.cls is not None: best_hyp['cls'] = args.cls

    # Dataset Setup
    base_data_dir = os.path.abspath('./data')
    sss_dir = os.path.join(base_data_dir, 'sampled_yolo_dataset')
    
    if not os.path.exists(sss_dir):
        kaggle.api.dataset_download_files('mawins/sample-sss-img', path=base_data_dir, unzip=True)
    
    if use_dwt:
        sss_dir = setup_dwt_dataset(sss_dir)
    elif args.sonar_aug:
        # Pass intensity to augmentation script
        sss_dir = setup_sonar_augmented_dataset(sss_dir, intensity=args.aug_intensity)

    sss_yaml = patch_data_yaml(os.path.join(sss_dir, 'data.yaml'), sss_dir)
    model_yaml_path = f"tmp_{mode}.yaml"
    pretrained_weights = "yolo11n.pt"
    
    with open(sss_yaml, 'r') as f:
        nc = yaml.safe_load(f)['nc']

    # Build Model
    if mode == "standard":
        model = YOLO(pretrained_weights)
    elif mode == "spdconv":
        with open(model_yaml_path, "w") as f: f.write(f"nc: {nc}\n" + SPDCONV_YOLO_YAML)
        model = YOLO(model_yaml_path).load(pretrained_weights)
    elif mode == "hybrid":
        with open(model_yaml_path, "w") as f: f.write(f"nc: {nc}\n" + ATTN_SPDCONV_YOLO_YAML)
        model = YOLO(model_yaml_path).load(pretrained_weights)

    exp_name = f"exp_{mode}_{'dwt' if use_dwt else ('aug' if args.sonar_aug else 'norm')}_f{args.freeze}_b{best_hyp['box']}_c{best_hyp['cls']}"

    # Execution
    project_dir = os.path.abspath("runs/experiments")
    
    if args.two_stage:
        stage1_epochs = min(30, max(1, args.epochs // 2))
        stage2_epochs = max(1, args.epochs - stage1_epochs)
        print(f">>> Stage 1: Training Frozen ({stage1_epochs} Epochs)...")
        
        # Use absolute path for project to ensure predictable location
        model.train(
            data=sss_yaml, imgsz=args.imgsz, epochs=stage1_epochs, batch=args.batch,
            freeze=10, project=project_dir, name=exp_name, exist_ok=True, **best_hyp
        )
        
        print(">>> Stage 1 Complete. Searching for weights...")
        # Check standard path and the 'detect' subfolder path YOLO sometimes creates
        possible_paths = [
            os.path.join(project_dir, exp_name, "weights", "best.pt"),
            os.path.join(project_dir, exp_name, "weights", "last.pt"),
            os.path.join("runs", "detect", exp_name, "weights", "best.pt")
        ]
        
        best_pt = None
        for p in possible_paths:
            if os.path.exists(p):
                best_pt = p
                break
        
        if not best_pt:
            raise FileNotFoundError(f"Could not find Stage 1 weights in any of: {possible_paths}")

        print(f">>> Loading weights from: {best_pt}")
        model = YOLO(best_pt)
        print(f">>> Stage 2: Unfreezing and Finishing ({stage2_epochs} Epochs)...")
        results = model.train(
            data=sss_yaml, imgsz=args.imgsz, epochs=stage2_epochs, batch=args.batch,
            freeze=0, project=project_dir, name=exp_name + "_final", exist_ok=True, **best_hyp
        )
    else:
        results = model.train(
            data=sss_yaml, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
            freeze=args.freeze, project=project_dir, name=exp_name, exist_ok=True, **best_hyp
        )
    
    metrics = get_metrics(results)
    metrics.update({"mode": mode, "use_dwt": use_dwt, "sonar_aug": args.sonar_aug, 
                    "freeze": args.freeze, "box_gain": best_hyp['box'], "cls_gain": best_hyp['cls'],
                    "two_stage": args.two_stage, "intensity": args.aug_intensity})
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "spdconv", "hybrid"])
    parser.add_argument("--dwt", action="store_true")
    parser.add_argument("--sonar_aug", action="store_true")
    parser.add_argument("--aug_intensity", type=float, default=0.02)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--two_stage", action="store_true")
    parser.add_argument("--box", type=float)
    parser.add_argument("--cls", type=float)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=default_batch_size)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--device", type=str, default="0")
    
    args = parser.parse_args()
    metrics = run_experiment(args)
    
    summary_file = "experiment_results.csv"
    df_new = pd.DataFrame([metrics])
    if os.path.exists(summary_file):
        df_old = pd.read_csv(summary_file)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(summary_file, index=False)
    print(df_new)
