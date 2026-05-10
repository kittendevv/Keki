import numpy as np
import onnx
import onnxruntime as ort
import torch
from model import FoodCNN

device = torch.device("cpu")  # type: ignore

model = FoodCNN(num_classes=101).to(device)
model.load_state_dict(torch.load("checkpoints/foodcnn_best.pth", map_location=device))
model.eval()

dummy_input = torch.randn(1, 3, 128, 128)

torch.onnx.export(
    model,
    (dummy_input,),
    "onnx/foodcnn.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}},
    opset_version=17,
)
print("Exported to foodcnn.onnx")

onnx_model = onnx.load("foodcnn.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")

session = ort.InferenceSession("foodcnn.onnx")
outputs = session.run(["logits"], {"image": dummy_input.numpy()})
print(f"✓ Test inference passed — output shape: {outputs[0].shape}")  # type: ignore
