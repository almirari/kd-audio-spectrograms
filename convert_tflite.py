import torch
import torchvision.models as models
import tensorflow as tf
from pytorch2keras.converter import pytorch_to_keras

# -----------------------------
# 1. LOAD MOBILENETV2 WITH YOUR CHECKPOINT
# -----------------------------
num_classes = 6  # adjust if your dataset has different classes
model = models.mobilenet_v2(num_classes=num_classes)

state_dict = torch.load("student_seed3.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# 2. CONVERT TO KERAS
# -----------------------------
dummy_input = torch.randn(1, 3, 224, 224)  # MobileNetV2 default input size

k_model = pytorch_to_keras(
    model,
    dummy_input,
    [(3, 224, 224)],  # input shape without batch dim
    verbose=True,
    change_ordering=True
)

# Save as Keras model
k_model.save("student_tf.h5")

# -----------------------------
# 3. CONVERT TO TFLITE
# -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(k_model)

# (Optional) Enable optimizations for smaller size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("student.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete: student.tflite")
