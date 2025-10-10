import os
import hashlib
import librosa
import numpy as np

# --- Step 1: Hash-based duplicate check (bit-for-bit identical) ---
def file_hash(filepath, algo="md5", chunk_size=8192):
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

# --- Step 2: Content-based duplicate check (compare audio data) ---
def load_audio_fingerprint(filepath, sr=22050):
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        # Normalize amplitude
        y = librosa.util.normalize(y)
        return y
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compare_audio(y1, y2, tol=1e-5):
    if y1 is None or y2 is None:
        return False
    if len(y1) != len(y2):
        # pad shorter one
        max_len = max(len(y1), len(y2))
        y1 = np.pad(y1, (0, max_len - len(y1)))
        y2 = np.pad(y2, (0, max_len - len(y2)))
    return np.allclose(y1, y2, atol=tol)

# --- Main duplicate finder & deleter ---
def remove_duplicates(root_folder, delete=True):
    hashes = {}
    audio_cache = {}
    duplicates = []

    # Collect all .wav files (recursively)
    files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(dirpath, f))

    print(f"Scanning {len(files)} .wav files across subfolders...")

    # Check exact duplicates by hash
    for path in files:
        h = file_hash(path)
        if h in hashes:
            print(f"[Exact duplicate] {path} == {hashes[h]}")
            duplicates.append(path)
        else:
            hashes[h] = path

    # Check content duplicates (slower)
    for i in range(len(files)):
        f1 = files[i]
        if f1 not in audio_cache:
            audio_cache[f1] = load_audio_fingerprint(f1)

        for j in range(i+1, len(files)):
            f2 = files[j]
            if f2 not in audio_cache:
                audio_cache[f2] = load_audio_fingerprint(f2)

            if compare_audio(audio_cache[f1], audio_cache[f2]):
                if f2 not in duplicates:  # only delete one of them
                    print(f"[Content duplicate] {f2} ≈ {f1}")
                    duplicates.append(f2)

    # Delete duplicates if requested
    if delete:
        for dup in duplicates:
            try:
                os.remove(dup)
                print(f"Deleted: {dup}")
            except Exception as e:
                print(f"Could not delete {dup}: {e}")
    else:
        print("\nDuplicates found (not deleted):")
        for dup in duplicates:
            print(dup)

if __name__ == "__main__":
    root_folder = "./data/audio/"
    remove_duplicates(root_folder, delete=False) 
