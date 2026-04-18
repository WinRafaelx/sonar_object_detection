import subprocess
import sys
import os
import time

# The exact same matrix from run_all_experiments.py
EXPERIMENTS = [
    ("standard", False, False, 0, 7.5),
    ("spdconv", False, False, 0, 7.5),
    ("hybrid", False, False, 0, 7.5),
    ("standard", True, False, 0, 7.5),
    ("spdconv", True, False, 0, 7.5),
    ("hybrid", True, False, 0, 7.5),
    ("standard", False, True, 0, 7.5),
    ("spdconv", False, True, 0, 7.5),
    ("hybrid", False, True, 0, 7.5),
    ("spdconv", False, False, 10, 7.5),
    ("hybrid", False, False, 10, 7.5),
    ("hybrid", False, True, 10, 7.5),
    ("hybrid", False, False, 10, 10.0),
    ("hybrid", False, True, 10, 10.0),
]

def verify_config(mode, dwt, sonar_aug, freeze, box):
    print(f"\n[ TESTING ] Mode: {mode}, DWT: {dwt}, SonarAug: {sonar_aug}, Freeze: {freeze}, Box: {box}")
    
    # We run for only 1 epoch and small batch to test initialization and first step
    cmd = [
        sys.executable, "normal.py",
        "--mode", mode,
        "--epochs", "1",
        "--batch", "2",
        "--freeze", str(freeze),
        "--box", str(box)
    ]
    if dwt: cmd.append("--dwt")
    if sonar_aug: cmd.append("--sonar_aug")
    
    try:
        # Use a short timeout per test (5 mins should be plenty for 1 epoch on A40)
        subprocess.run(cmd, check=True, timeout=300)
        return True
    except Exception as e:
        print(f"!!! CONFIG FAILED: {e}")
        return False

def main():
    print("="*60)
    print("SONAR DETECTION: CONFIGURATION SMOKE TEST")
    print("Testing 1 epoch for all unique combinations...")
    print("="*60)
    
    results = []
    for mode, dwt, sonar_aug, freeze, box in EXPERIMENTS:
        success = verify_config(mode, dwt, sonar_aug, freeze, box)
        results.append((f"{mode}_dwt{dwt}_aug{sonar_aug}_f{freeze}_b{box}", "PASS" if success else "FAIL"))
        if not success:
            print("\nCRITICAL ERROR FOUND. Stopping smoke test to save time.")
            break

    print("\n" + "="*60)
    print("SMOKE TEST RESULTS")
    print("="*60)
    for name, status in results:
        print(f"{name:<45} | {status}")
    print("="*60)

if __name__ == "__main__":
    main()
