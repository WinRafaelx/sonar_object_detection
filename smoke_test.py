import subprocess
import sys
import os
import time

# Unique configurations from Phase 3 Matrix
# (mode, dwt, aug, intensity, freeze, two_stage, box, cls)
EXPERIMENTS = [
    ("hybrid", False, False, 0.0, 0, False, 12.0, 1.0),
    ("hybrid", False, False, 0.0, 0, False, 15.0, 1.0),
    ("hybrid", False, False, 0.0, 0, False, 12.0, 1.5),
    ("spdconv", False, False, 0.0, 0, False, 12.0, 1.0),
    ("hybrid", False, False, 0.0, 0, True, 10.0, 1.0),
    ("hybrid", False, True, 0.02, 0, True, 10.0, 1.0),
    ("spdconv", False, False, 0.0, 0, True, 10.0, 1.0),
    ("hybrid", False, True, 0.01, 0, False, 10.0, 1.0),
    ("hybrid", False, True, 0.02, 0, False, 10.0, 1.0),
    ("hybrid", False, True, 0.03, 0, False, 10.0, 1.0),
    ("hybrid", False, True, 0.02, 0, True, 12.0, 1.5),
    ("hybrid", False, True, 0.01, 0, True, 15.0, 1.2),
    ("spdconv", False, True, 0.02, 0, True, 12.0, 1.0),
    ("hybrid", True, False, 0.0, 0, True, 10.0, 1.0),
    ("standard", False, True, 0.02, 0, False, 10.0, 1.0),
    ("standard", False, False, 0.0, 0, False, 12.0, 1.5),
]

def verify_config(mode, dwt, aug, intensity, freeze, two_stage, box, cls):
    print(f"\n[ TESTING ] Mode: {mode}, DWT: {dwt}, Aug: {aug}, Int: {intensity}, 2S: {two_stage}, B: {box}, C: {cls}")
    
    # Smoke test: 1 epoch, small batch, 2 epochs total for 2-stage (1+1)
    epochs = "2" if two_stage else "1"
    cmd = [
        sys.executable, "normal.py",
        "--mode", mode,
        "--epochs", epochs,
        "--batch", "2",
        "--freeze", str(freeze),
        "--box", str(box),
        "--cls", str(cls),
        "--aug_intensity", str(intensity)
    ]
    if dwt: cmd.append("--dwt")
    if aug: cmd.append("--sonar_aug")
    if two_stage: cmd.append("--two_stage")
    
    try:
        subprocess.run(cmd, check=True, timeout=600) # Two-stage takes longer
        return True
    except Exception as e:
        print(f"!!! CONFIG FAILED: {e}")
        return False

def main():
    print("="*60)
    print("SONAR DETECTION: PHASE 3 CONFIGURATION SMOKE TEST")
    print("Testing 1 epoch (or 2 for Stage-Training) for all combinations...")
    print("="*60)
    
    results = []
    for exp in EXPERIMENTS:
        success = verify_config(*exp)
        name = f"{exp[0]}_dwt{exp[1]}_aug{exp[2]}_2s{exp[5]}_b{exp[6]}"
        results.append((name, "PASS" if success else "FAIL"))
        if not success:
            print("\nCRITICAL ERROR FOUND. Stopping smoke test.")
            break

    print("\n" + "="*60)
    print("PHASE 3 SMOKE TEST RESULTS")
    print("="*60)
    for name, status in results:
        print(f"{name:<45} | {status}")
    print("="*60)

if __name__ == "__main__":
    main()
