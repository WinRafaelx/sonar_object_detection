import subprocess
import sys
import os
import pandas as pd
import time

# Matrix of 16 targeted experiments for Phase 3: 
# (mode, dwt, sonar_aug, intensity, freeze, two_stage, box_gain, cls_gain, epochs)
EXPERIMENTS = [
    # --- GROUP A: RECALL AGGRESSION (Focus on finding everything) ---
    ("hybrid", False, False, 0.0, 0, False, 12.0, 1.0, 100),
    ("hybrid", False, False, 0.0, 0, False, 15.0, 1.0, 100),
    ("hybrid", False, False, 0.0, 0, False, 12.0, 1.5, 100),
    ("spdconv", False, False, 0.0, 0, False, 12.0, 1.0, 100),

    # --- GROUP B: TWO-STAGE "UNFREEZER" (Stability then Accuracy) ---
    ("hybrid", False, False, 0.0, 0, True, 10.0, 1.0, 300),
    ("hybrid", False, True, 0.02, 0, True, 10.0, 1.0, 300),
    ("spdconv", False, False, 0.0, 0, True, 10.0, 1.0, 300),

    # --- GROUP C: THE "SOFT TOUCH" (Subtle Noise for Precision) ---
    ("hybrid", False, True, 0.01, 0, False, 10.0, 1.0, 100),
    ("hybrid", False, True, 0.02, 0, False, 10.0, 1.0, 100),
    ("hybrid", False, True, 0.03, 0, False, 10.0, 1.0, 100),

    # --- GROUP D: THE FUSION (Best of all worlds) ---
    ("hybrid", False, True, 0.02, 0, True, 12.0, 1.5, 300),
    ("hybrid", False, True, 0.01, 0, True, 15.0, 1.2, 300),
    ("spdconv", False, True, 0.02, 0, True, 12.0, 1.0, 300),
    
    # --- GROUP E: DWT SECOND CHANCE (Longer training) ---
    ("hybrid", True, False, 0.0, 0, True, 10.0, 1.0, 300),
    ("standard", False, True, 0.02, 0, False, 10.0, 1.0, 100),
    ("standard", False, False, 0, 12.0, 1.5, 100),
]

def run_cmd(mode, dwt, aug, intensity, freeze, two_stage, box, cls, epochs):
    cmd = [
        sys.executable, "normal.py",
        "--mode", mode,
        "--epochs", str(epochs),
        "--freeze", str(freeze),
        "--box", str(box),
        "--cls", str(cls),
        "--aug_intensity", str(intensity)
    ]
    if dwt: cmd.append("--dwt")
    if aug: cmd.append("--sonar_aug")
    if two_stage: cmd.append("--two_stage")
    
    print(f"\n>>> RUNNING: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        subprocess.run(cmd, check=True)
        duration = (time.time() - start_time) / 60
        print(f">>> SUCCESS: {mode} (DWT={dwt}, Aug={aug}, B={box}, C={cls}, 2S={two_stage}) in {duration:.2f}m")
    except subprocess.CalledProcessError as e:
        print(f">>> ERROR: Experiment {mode} failed: {e}")

def main():
    print("="*60)
    print("SONAR DETECTION PHASE 3: THE RECALL & FUSION MATRIX")
    print("="*60)
    print(f"Total experiments planned: {len(EXPERIMENTS)}")
    print("="*60)

    for mode, dwt, aug, intensity, freeze, two_stage, box, cls, epochs in EXPERIMENTS:
        run_cmd(mode, dwt, aug, intensity, freeze, two_stage, box, cls, epochs)

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
