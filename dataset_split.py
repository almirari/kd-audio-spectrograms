import os
from pathlib import Path
import random
from math import floor

def make_balanced_splits(
    root_dir, out_dir,
    train_frac=0.7, val_frac=0.15, test_frac=0.15,
    random_seed=42, balance_mode="min"
):
    """
    Creates balanced train/val/test splits across classes.
    Each class contributes equally (based on smallest or capped class size).
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory {root_dir} does not exist.")
    
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 0.001:
        raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")
        
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if len(classes) == 0:
        raise ValueError(f"No class directories found in {root_dir}")
    
    print(f"Found {len(classes)} classes: {classes}")
    
    random.seed(random_seed)
    all_samples = {}

    # Collect samples per class
    for cls in classes:
        cls_path = root / cls
        samples = [
            str(cls_path / f)
            for f in os.listdir(cls_path)
            if f.lower().endswith((".wav", ".png", ".jpg", ".jpeg"))
        ]
        random.shuffle(samples)
        all_samples[cls] = samples

    # Determine balancing count
    class_counts = {cls: len(samples) for cls, samples in all_samples.items()}
    min_count = min(class_counts.values())
    max_cap = min(200, min_count)  # cap to reasonable size if needed

    print(f"Balancing using {max_cap} samples per class (min={min_count})")

    train_files, val_files, test_files = [], [], []

    for cls, samples in all_samples.items():
        if len(samples) < 3:
            print(f"⚠️ Class '{cls}' too small ({len(samples)} samples), skipping.")
            continue

        n_use = min(len(samples), max_cap)
        chosen = samples[:n_use]

        n_train = floor(n_use * train_frac)
        n_val = floor(n_use * val_frac)
        n_test = n_use - n_train - n_val

        train_samples = chosen[:n_train]
        val_samples = chosen[n_train:n_train + n_val]
        test_samples = chosen[n_train + n_val:]

        train_files.extend((s, cls) for s in train_samples)
        val_files.extend((s, cls) for s in val_samples)
        test_files.extend((s, cls) for s in test_samples)

        print(f"{cls}: {n_train} train, {n_val} val, {n_test} test")

    os.makedirs(out_dir, exist_ok=True)

    def save_split(split, filename):
        with open(os.path.join(out_dir, filename), "w") as f:
            for path, cls in split:
                rel_path = os.path.relpath(path, root_dir)
                f.write(f"{rel_path}\n")

    save_split(train_files, "train.txt")
    save_split(val_files, "val.txt")
    save_split(test_files, "test.txt")

    print("\nFinal class distributions :")
    for cls in classes:
        t = sum(1 for _, c in train_files if c == cls)
        v = sum(1 for _, c in val_files if c == cls)
        e = sum(1 for _, c in test_files if c == cls)
        if t + v + e > 0:
            print(f"  {cls}: {t}T + {v}V + {e}E = {t+v+e} total")

if __name__ == "__main__":
    make_balanced_splits(
        root_dir="./data/spectrograms/",
        out_dir="./data/splits/",
        random_seed=30
    )
