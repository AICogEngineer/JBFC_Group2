import os
import subprocess
import shutil
import numpy as np
from dotenv import load_dotenv
import sys

# --- CONFIGURATION ---
load_dotenv()
SOURCE_DIR = "./dataset_c-main"           
PIPELINE_SCRIPT = "./integration/test_single_sprite_pipline.py" 
OUTPUT_DIR = "dataset_c_organized" 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"--- Debugging Organization of {SOURCE_DIR} ---")

files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(files)} images to process.")

for filename in files:
    full_file_path = os.path.join(SOURCE_DIR, filename)
    
    try:
        # Run the pipeline script
        result = subprocess.run(
            [sys.executable, PIPELINE_SCRIPT, full_file_path], # <--- CHANGE "python" to sys.executable
            capture_output=True,
            text=True
        )

        # DEBUG: Print exactly what the script returned
        raw_output = result.stdout.strip()
        error_output = result.stderr.strip()

        if result.returncode != 0:
            print(f"Error running script for {filename}:")
            print(f"   STDERR: {error_output}")
            continue

        # TRICK: Split by newlines and take the LAST line only
        # This ignores warnings printed before the label
        lines = raw_output.split('\n')
        predicted_label = lines[-1].strip() if lines else ""

        if not predicted_label:
            print(f"Skipping {filename}: Script returned empty output.")
            continue

        print(f"Label found: '{predicted_label}' for {filename}")

        # Create folder and move
        target_folder = os.path.join(OUTPUT_DIR, predicted_label)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        shutil.copy2(full_file_path, os.path.join(target_folder, filename))

    except Exception as e:
        print(f"Python Error on {filename}: {e}")

print("Done.")