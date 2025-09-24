import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "./data_audio"               
OUT_DIR = "./data_spectrograms"
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 128
FMAX = 2000

def save_spectrogram(path, out_path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, res_type="kaiser_fast")
    # Pad if too short
    if len(y) < int(SAMPLE_RATE * DURATION):
        y = np.pad(y, (0, int(SAMPLE_RATE*DURATION) - len(y)))
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Save as PNG
    plt.figure(figsize=(2, 2))
    librosa.display.specshow(log_mel, sr=sr, fmax=FMAX, y_axis=None, x_axis=None, cmap="magma")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# Loop over dataset and save spectrograms
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    out_label_dir = os.path.join(OUT_DIR, label)
    os.makedirs(out_label_dir, exist_ok=True)

    for fname in os.listdir(label_dir):
        if not fname.endswith(".wav"):
            continue
        in_path = os.path.join(label_dir, fname)
        out_path = os.path.join(out_label_dir, fname.replace(".wav", ".png"))
        save_spectrogram(in_path, out_path)

print("All audio converted to spectrogram images in:", OUT_DIR)
