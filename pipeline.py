"""
Main pipeline - Runs all steps in sequence
Skips if output already exists
"""

import os
import subprocess
import sys

steps = [
    ("clean.py", "datasets/cleaned_data.csv"),
    ("compare_models.py", "outputs/results/model_comparison.csv"),
    ("train_model.py", "outputs/models/final_model.pkl"),
    ("tune_model.py", "outputs/models/tuned_model.pkl"),
    ("statistical_tests.py", "documentation/statistical_validation.md"),
    ("visualization.py", "outputs/figures/confusion_matrix.png"),
]

print("Starting pipeline...")

for script, output in steps:
    if os.path.exists(output):
        print(f"Skipping {script} (output exists)")
    else:
        print(f"Running {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"Error in {script}")
            sys.exit(1)

print("Pipeline complete!")