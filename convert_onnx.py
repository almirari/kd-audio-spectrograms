import torch
import torchvision.models as models
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

# 1. Load MobileNetV2 (PyTorch)
num_classes = 4
model = models.mobilenet_v2(num_classes=num_classes)
state_dict = torch.load("student.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()

# 2. Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
onnx_filename = "student.onnx"
torch.onnx.export(
    model, dummy_input, onnx_filename,
    input_names=["input"], output_names=["output"],
    opset_version=11
)
print(f"Exported to {onnx_filename}")

# 3. Sanitize ONNX node names (remove invalid characters)
onnx_model = onnx.load(onnx_filename)

def sanitize_name(name: str):
    # Replace illegal characters with underscore
    return name.replace("/", "_").replace("\\", "_")

for node in onnx_model.graph.node:
    node.name = sanitize_name(node.name)
for inp in onnx_model.graph.input:
    inp.name = sanitize_name(inp.name)
for out in onnx_model.graph.output:
    out.name = sanitize_name(out.name)
for init in onnx_model.graph.initializer:
    init.name = sanitize_name(init.name)

clean_onnx_path = "student.onnx"
onnx.save(onnx_model, clean_onnx_path)
print(f"Saved cleaned ONNX model as {clean_onnx_path}")

# 4. Convert ONNX → TensorFlow SavedModel
onnx_model = onnx.load(clean_onnx_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("student_tf_model")
print("Converted to TensorFlow SavedModel (student_tf_model/)")

# 5. Convert TensorFlow → TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("student_tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_filename = "student.tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_model)

print(f"Conversion complete! Saved TFLite model as {tflite_filename}")