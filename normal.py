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
from constant import batch_size
import argparse

# Import custom modules
from custom_modules import SonarSPDConv, CoordAtt, CBAM, EMA, BiFPN_Concat2
from architecture_configs import SPDCONV_YOLO_YAML, ATTN_SPDCONV_YOLO_YAML
from preprocess_dwt import setup_dwt_dataset

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
        custom_mods = "SonarSPDConv, CoordAtt, CBAM, EMA, BiFPN_Concat2,"
        if "SonarSPDConv" not in source:
            new_source = source.replace("A2C2f,", f"A2C2f, {custom_mods}", 1)
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
            })
            exec(new_source, exec_globals)
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
def run_experiment(mode, epochs=100, use_dwt=False):
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

    sss_yaml = patch_data_yaml(os.path.join(sss_dir, 'data.yaml'), sss_dir)
    
    # Architecture & Weights Setup
    model_yaml_path = f"tmp_{mode}.yaml"
    pretrained_weights = "yolo11m.pt"
    
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
    elif mode == "full_staged":
        # Stage 1: Pre-train on URPC2020
        print("\n--- STAGE 1: Pre-training on URPC2020 ---")
        urpc_dir = os.path.join(base_data_dir, 'URPC2020')
        if not os.path.exists(urpc_dir):
            kaggle.api.dataset_download_files('lywang777/urpc2020', path=base_data_dir, unzip=True)
        urpc_yaml = patch_data_yaml(os.path.join(urpc_dir, 'data.yaml'), urpc_dir)
        
        with open(model_yaml_path, "w") as f:
            f.write(f"nc: {yaml.safe_load(open(urpc_yaml))['nc']}\n" + ATTN_SPDCONV_YOLO_YAML)
        
        stage1_model = YOLO(model_yaml_path).load(pretrained_weights)
        stage1_model.train(data=urpc_yaml, epochs=50, batch=batch_size, imgsz=640, name=f"stage1_{mode}")
        
        # Stage 2: Fine-tune on SSS
        print("\n--- STAGE 2: Fine-tuning on SSS ---")
        stage1_weights = f"runs/detect/stage1_{mode}/weights/best.pt"
        model = YOLO(stage1_weights)
    else:
        raise ValueError("Invalid mode")

    # Training
    results = model.train(
        data=sss_yaml,
        imgsz=640,
        epochs=epochs,
        patience=50,
        batch=batch_size,
        project="runs/experiments",
        name=f"exp_{mode}_{'dwt' if use_dwt else 'normal'}",
        verbose=True,
        **best_hyp
    )
    
    metrics = get_metrics(results)
    metrics["mode"] = mode
    metrics["use_dwt"] = use_dwt
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "spdconv", "hybrid", "full_staged"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dwt", action="store_true", help="Use DWT preprocessed dataset")
    args = parser.parse_args()

    metrics = run_experiment(args.mode, args.epochs, args.dwt)
    
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
