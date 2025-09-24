import os
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

def make_splits(root_dir, out_dir, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_seed=42):
    """
    root_dir: path to root folder containing class subfolders
    out_dir: path where to save train.txt, val.txt, test.txt
    Fractions should sum to 1.0 (or very close).
    
    This function ensures ALL classes appear in each split (train, val, test).
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory {root_dir} does not exist.")
    
    # Verify fractions sum to 1
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 0.001:
        raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")
        
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if len(classes) == 0:
        raise ValueError(f"No class directories found in {root_dir}")
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Collect samples by class
    class_samples = {}
    for cls in classes:
        cls_path = root / cls
        samples = []
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".wav", ".png", ".jpg", ".jpeg")):
                full = cls_path / fname
                samples.append(str(full))
        class_samples[cls] = samples
        print(f"  {cls}: {len(samples)} samples")
    
    # Check if any class is empty
    empty_classes = [cls for cls, samples in class_samples.items() if len(samples) == 0]
    if empty_classes:
        raise ValueError(f"Empty classes found: {empty_classes}")
    
    # Check minimum samples per class for splitting
    min_samples = min(len(samples) for samples in class_samples.values())
    if min_samples < 3:
        print(f"WARNING: Some classes have only {min_samples} samples.")
        print("   This may result in uneven splits. Consider having at least 3 samples per class.")
    
    # Split each class individually to ensure all classes in each split
    train_files, val_files, test_files = [], [], []
    
    for cls, samples in class_samples.items():
        n_samples = len(samples)
        
        if n_samples == 1:
            # Only 1 sample - put in train, warn user
            train_files.extend([(sample, cls) for sample in samples])
            print(f"Class '{cls}' has only 1 sample - assigned to train set")
            continue
        elif n_samples == 2:
            # Only 2 samples - 1 for train, 1 for test
            train_files.append((samples[0], cls))
            test_files.append((samples[1], cls))
            print(f"Class '{cls}' has only 2 samples - 1 in train, 1 in test, 0 in val")
            continue
        
        # Calculate split sizes for this class
        n_test = max(1, int(n_samples * test_frac))
        n_val = max(1, int(n_samples * val_frac))
        n_train = n_samples - n_test - n_val
        
        # Ensure at least 1 sample per split
        if n_train < 1:
            n_train = 1
            n_val = max(1, n_samples - n_train - n_test)
            n_test = n_samples - n_train - n_val
        
        print(f"  {cls}: {n_train} train, {n_val} val, {n_test} test")
        
        # Split samples for this class
        class_train, temp = train_test_split(
            samples, 
            train_size=n_train, 
            random_state=random_seed + hash(cls) % 1000  # Different seed per class
        )
        
        if len(temp) >= 2:
            class_val, class_test = train_test_split(
                temp, 
                train_size=n_val, 
                random_state=random_seed + hash(cls) % 1000 + 1
            )
        else:
            # If only 1 sample left, assign to val or test
            class_val = temp if n_val > 0 else []
            class_test = temp if n_val == 0 else []
        
        # Add to final lists
        train_files.extend([(sample, cls) for sample in class_train])
        val_files.extend([(sample, cls) for sample in class_val])
        test_files.extend([(sample, cls) for sample in class_test])
    
    # Shuffle the final lists
    import random
    random.seed(random_seed)
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    
    # Verify all classes are in each split (where possible)
    train_classes = set(cls for _, cls in train_files)
    val_classes = set(cls for _, cls in val_files)
    test_classes = set(cls for _, cls in test_files)
    
    print(f"\n📊 SPLIT VERIFICATION:")
    print(f"Train set: {len(train_files)} samples, {len(train_classes)} classes")
    print(f"Val set:   {len(val_files)} samples, {len(val_classes)} classes")
    print(f"Test set:  {len(test_files)} samples, {len(test_classes)} classes")
    
    missing_from_train = set(classes) - train_classes
    missing_from_val = set(classes) - val_classes  
    missing_from_test = set(classes) - test_classes
    
    if missing_from_train:
        print(f"Classes missing from TRAIN: {missing_from_train}")
    if missing_from_val:
        print(f"Classes missing from VAL: {missing_from_val}")
    if missing_from_test:
        print(f"Classes missing from TEST: {missing_from_test}")
    
    if not (missing_from_train or missing_from_val or missing_from_test):
        print("✅ All classes present in all splits!")
    
    # Save splits
    os.makedirs(out_dir, exist_ok=True)

    def save_split(split, filename):
        with open(os.path.join(out_dir, filename), "w") as f:
            for (path, cls) in split:
                rel_path = os.path.relpath(path, root_dir)
                f.write(f"{rel_path}\n")

    save_split(train_files, "train.txt")
    save_split(val_files, "val.txt")
    save_split(test_files, "test.txt")

    print(f"\nSplit files saved to {out_dir}")
    
    # Final summary by class
    print(f"\n📈 FINAL DISTRIBUTION BY CLASS:")
    for cls in classes:
        train_count = sum(1 for _, c in train_files if c == cls)
        val_count = sum(1 for _, c in val_files if c == cls)
        test_count = sum(1 for _, c in test_files if c == cls)
        total = train_count + val_count + test_count
        print(f"  {cls}: {train_count}T + {val_count}V + {test_count}E = {total} total")

if __name__ == "__main__":
    make_splits(
        root_dir="./data_spectrograms/", 
        out_dir="./data_splits",
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        random_seed=42
    )