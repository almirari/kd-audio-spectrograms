import os
import shutil
import pandas as pd

# Input paths
AUDIO_DIR = r".\data\raw\audio_and_txt_files"
CSV_PATH = r".\data\raw\patient_diagnosis.csv"

# Output path
OUT_DIR = r".\data\audio"

def main():
    # Read CSV
    df = pd.read_csv(CSV_PATH)
    # Expecting columns: ["patient_id", "diagnosis"]
    mapping = dict(zip(df["patient_id"].astype(str), df["diagnosis"]))

    os.makedirs(OUT_DIR, exist_ok=True)

    # Process all wav files
    for fname in os.listdir(AUDIO_DIR):
        if not fname.lower().endswith(".wav"):
            continue
        patient_id = fname.split("_")[0]  # e.g., "101_1b1..." → "101"
        diagnosis = mapping.get(patient_id)
        if diagnosis is None:
            print(f"Warning: no diagnosis found for patient {patient_id}, skipping {fname}")
            continue

        # Create diagnosis subfolder
        target_dir = os.path.join(OUT_DIR, diagnosis)
        os.makedirs(target_dir, exist_ok=True)

        # Copy file
        src = os.path.join(AUDIO_DIR, fname)
        dst = os.path.join(target_dir, fname)
        shutil.copy2(src, dst)

    print(f"Dataset prepared in: {OUT_DIR}")

if __name__ == "__main__":
    main()
