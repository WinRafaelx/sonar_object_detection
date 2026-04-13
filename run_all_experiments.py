import subprocess
import sys
import os
import pandas as pd
import time

# List of experiments to run: (mode, use_dwt)
EXPERIMENTS = [
    ("standard", False),      # Baseline
    ("spdconv", False),       # Backbone optimization (SOCA)
    ("hybrid", False),        # Full Architecture (SPDConv + Attention + BiFPN)
    ("standard", True),       # DWT Pre-processing only
    ("hybrid", True),         # Full Architecture + DWT (The "Ultimate" model)
]

# You can reduce this for a "quick check" or keep at 100+ for final results
EPOCHS = 100

def run_cmd(mode, use_dwt):
    cmd = [
        sys.executable, "normal.py",
        "--mode", mode,
        "--epochs", str(EPOCHS)
    ]
    if use_dwt:
        cmd.append("--dwt")
    
    print(f"\n>>> RUNNING: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        # We use subprocess.run to ensure each training session starts with a fresh memory/CUDA state
        subprocess.run(cmd, check=True)
        duration = (time.time() - start_time) / 60
        print(f">>> SUCCESS: {mode} (DWT={use_dwt}) completed in {duration:.2f} minutes.")
    except subprocess.CalledProcessError as e:
        print(f">>> ERROR: Experiment {mode} (DWT={use_dwt}) failed with error: {e}")

def main():
    print("="*60)
    print("SONAR OBJECT DETECTION: AUTOMATED EXPERIMENT PIPELINE")
    print("="*60)
    print(f"Total experiments planned: {len(EXPERIMENTS)}")
    print(f"Epochs per run: {EPOCHS}")
    print("="*60)

    for mode, use_dwt in EXPERIMENTS:
        run_cmd(mode, use_dwt)

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
