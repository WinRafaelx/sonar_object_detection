import subprocess
import sys
import os
import pandas as pd
import time

# Matrix of 20 experiments: (mode, dwt, sonar_aug, freeze, box_gain, epochs)
# This is designed to run for ~8 hours on an A40
EXPERIMENTS = [
    # --- BASELINES (100 Epochs) ---
    ("standard", False, False, 0, 7.5, 100),
    ("spdconv", False, False, 0, 7.5, 100),
    ("hybrid", False, False, 0, 7.5, 100),
    
    # --- DWT SERIES ---
    ("standard", True, False, 0, 7.5, 100),
    ("spdconv", True, False, 0, 7.5, 100),
    ("hybrid", True, False, 0, 7.5, 100),

    # --- SONAR AUG SERIES (SMART NOISE) ---
    ("standard", False, True, 0, 7.5, 100),
    ("spdconv", False, True, 0, 7.5, 100),
    ("hybrid", False, True, 0, 7.5, 100),
    
    # --- FREEZING STRATEGY (f10 vs f0) ---
    ("spdconv", False, False, 10, 7.5, 100),
    ("hybrid", False, False, 10, 7.5, 100),
    ("hybrid", False, True, 10, 7.5, 100),
    
    # --- RECALL OPTIMIZATION (High Box Gain) ---
    ("hybrid", False, False, 10, 10.0, 100),
    ("hybrid", False, True, 10, 10.0, 100),
    
    # --- LONG RUNS (300 Epochs) ---
    ("standard", False, False, 0, 7.5, 300),
    ("spdconv", False, False, 0, 7.5, 300),
    ("hybrid", False, False, 10, 7.5, 300),
    ("hybrid", False, True, 10, 7.5, 300),
    ("hybrid", False, True, 0, 10.0, 300),
    ("hybrid", True, False, 0, 7.5, 300),
]

def run_cmd(mode, use_dwt, use_sonar_aug, freeze, box_gain, epochs):
    cmd = [
        sys.executable, "normal.py",
        "--mode", mode,
        "--epochs", str(epochs),
        "--freeze", str(freeze),
        "--box", str(box_gain)
    ]
    if use_dwt:
        cmd.append("--dwt")
    if use_sonar_aug:
        cmd.append("--sonar_aug")
    
    print(f"\n>>> RUNNING: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        subprocess.run(cmd, check=True)
        duration = (time.time() - start_time) / 60
        print(f">>> SUCCESS: {mode} (DWT={use_dwt}, SonarAug={use_sonar_aug}, F={freeze}, B={box_gain}) in {duration:.2f}m")
    except subprocess.CalledProcessError as e:
        print(f">>> ERROR: Experiment {mode} failed: {e}")

def main():
    print("="*60)
    print("SONAR OBJECT DETECTION: 8-HOUR HYPER-EXPERIMENT MATRIX")
    print("="*60)
    print(f"Total experiments planned: {len(EXPERIMENTS)}")
    print("="*60)

    for mode, use_dwt, use_sonar_aug, freeze, box_gain, epochs in EXPERIMENTS:
        run_cmd(mode, use_dwt, use_sonar_aug, freeze, box_gain, epochs)

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
