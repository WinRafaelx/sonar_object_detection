import subprocess
import sys
import os
import pandas as pd
import time

# List of experiments to run: (mode, use_dwt, use_sonar_aug)
EXPERIMENTS = [
    ("hybrid", False, True),  # Full Architecture + Sonar Augmentation (Speckle/Shadows)
]

# Standard epochs for most runs
EPOCHS = 100
# High-intensity training for the final model
ULTIMATE_EPOCHS = 300

def run_cmd(mode, use_dwt, use_sonar_aug):
    epochs = ULTIMATE_EPOCHS if use_sonar_aug else EPOCHS
    cmd = [
        sys.executable, "normal.py",
        "--mode", mode,
        "--epochs", str(epochs)
    ]
    if use_dwt:
        cmd.append("--dwt")
    if use_sonar_aug:
        cmd.append("--sonar_aug")
    
    print(f"\n>>> RUNNING: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        # We use subprocess.run to ensure each training session starts with a fresh memory/CUDA state
        subprocess.run(cmd, check=True)
        duration = (time.time() - start_time) / 60
        print(f">>> SUCCESS: {mode} (DWT={use_dwt}, SonarAug={use_sonar_aug}) completed in {duration:.2f} minutes.")
    except subprocess.CalledProcessError as e:
        print(f">>> ERROR: Experiment {mode} (DWT={use_dwt}, SonarAug={use_sonar_aug}) failed with error: {e}")

def main():
    print("="*60)
    print("SONAR OBJECT DETECTION: AUTOMATED EXPERIMENT PIPELINE")
    print("="*60)
    print(f"Total experiments planned: {len(EXPERIMENTS)}")
    print("="*60)

    for mode, use_dwt, use_sonar_aug in EXPERIMENTS:
        run_cmd(mode, use_dwt, use_sonar_aug)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)

    # Display final results table
    if os.path.exists("experiment_results.csv"):
        df = pd.read_csv("experiment_results.csv")
        # Sort by mAP50 to see the best performing model at the top
        df = df.sort_values(by="mAP50", ascending=False)
        print("\nFINAL RANKING (By mAP50):")
        print(df.to_string(index=False))
        
        # Save a timestamped copy of the final report
        report_name = f"final_report_{int(time.time())}.csv"
        df.to_csv(report_name, index=False)
        print(f"\n✓ Final report saved as: {report_name}")
    else:
        print("⚠ Error: No results file found. Check logs for training failures.")

if __name__ == "__main__":
    main()
