import subprocess
import sys
import os
import argparse
import time

# List of training scripts to run
SCRIPTS = [
    "train.py",
    "normal.py",
    "soca.py",
    "BES_yolo.py",
    "golden_combination.py"
]

def run_script(script_path, test_mode=False, device="0"):
    """Runs a single training script and handles failures."""
    epochs = 5 if test_mode else 500
    batch = 16
    patience = 50
    
    cmd = [sys.executable, script_path]
    
    # Scripts that use argparse
    if script_path in ["normal.py", "golden_combination.py"]:
        cmd.extend(["--epochs", str(epochs), "--batch", str(batch), "--device", str(device)])
        if script_path == "normal.py":
            cmd.extend(["--patience", str(patience), "--mode", "standard"])
    else:
        # For scripts we'll update to support --device or that use train_args
        cmd.extend(["--device", str(device)])
        if test_mode:
            cmd.extend(["--epochs", "1"])

    print(f"\n{'='*60}")
    print(f"STARTING: {script_path} ({'TEST' if test_mode else 'PROD'} | DEVICE: {device})")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd)
        duration = (time.time() - start_time) / 60
        if result.returncode == 0:
            print(f"\n>>> SUCCESS: {script_path} completed in {duration:.2f}m")
        else:
            print(f"\n>>> FAILED: {script_path} exited with code {result.returncode}")
    except Exception as e:
        print(f"\n>>> ERROR: Could not execute {script_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run all sonar detection experiments.")
    parser.add_argument("--test", action="store_true", help="Run 1 epoch for each model.")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID (e.g. 0, 1) or 'cpu'.")
    args = parser.parse_args()

    print("="*60)
    print("SONAR OBJECT DETECTION: EXPERIMENT PIPELINE")
    print(f"Mode: {'TEST' if args.test else 'PRODUCTION'} | GPU: {args.device}")
    print("="*60)

    for script in SCRIPTS:
        if not os.path.exists(script):
            print(f"⚠ Warning: Script {script} not found. Skipping.")
            continue
        run_script(script, test_mode=args.test, device=args.device)

    print("\n" + "="*60)
    print("ALL PLANNED EXPERIMENTS PROCESSED")
    print("="*60)

if __name__ == "__main__":
    main()
