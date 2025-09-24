import os
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

def make_splits(root_dir, out_dir, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_seed=42):
    """
    root_dir: path to root folder containing class subfolders
    out_dir: path where to save train.txt, val.txt, test.txt
    Fractions should sum to 1.0 (or very close).
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"❌ Root directory {root_dir} does not exist.")
        
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    all_samples = []
    for cls in classes:
        cls_path = root / cls
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".wav", ".png", ".jpg")):
                full = cls_path / fname
                all_samples.append((str(full), cls))

    if len(all_samples) == 0:
        raise ValueError(f"❌ No samples found in {root_dir}. Check folder names and file extensions.")

    # Check class counts
    labels = [lbl for _, lbl in all_samples]
    label_counts = Counter(labels)
    use_stratify = all(count >= 2 for count in label_counts.values())
    if not use_stratify:
        print("Some classes have < 2 samples. Disabling stratified split.")
        stratify_labels = None
    else:
        stratify_labels = labels

    # First split: train+val vs test
    trainval, test = train_test_split(
        all_samples,
        test_size=test_frac,
        stratify=stratify_labels,
        random_state=random_seed,
    )

    # For second split, recompute stratify labels if possible
    if use_stratify:
        trainval_labels = [lbl for _, lbl in trainval]
        trainval_counts = Counter(trainval_labels)
        use_stratify_trainval = all(count >= 2 for count in trainval_counts.values())
        stratify_trainval = trainval_labels if use_stratify_trainval else None
        if stratify_trainval is None:
            print("Some classes in train/val subset have < 2 samples. Disabling stratified split for val.")
    else:
        stratify_trainval = None

    # Second split: train vs val
    val_relative = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        trainval,
        test_size=val_relative,
        stratify=stratify_trainval,
        random_state=random_seed,
    )

    # Save splits
    os.makedirs(out_dir, exist_ok=True)

    def save_split(split, filename):
        with open(os.path.join(out_dir, filename), "w") as f:
            for (path, cls) in split:
                rel_path = os.path.relpath(path, root_dir)
                f.write(f"{rel_path}\n")

    save_split(train, "train.txt")
    save_split(val, "val.txt")
    save_split(test, "test.txt")

    print(f"Split complete.")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

if __name__ == "__main__":
    make_splits(
        root_dir="./data_spectrograms/", 
        out_dir="./data_splits",
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15
    )