import torch
import torchvision.models as models
import numpy as np
import onnx
import onnxruntime as ort

model = models.resnet18(pretrained=True)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, dummy_input, "resnet18.onnx", 
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                  opset_version=11)

session = ort.InferenceSession("resnet18.onnx")
ort_input = {"input": dummy_input.numpy()}
ort_output = session.run(None, ort_input)

with torch.no_grad():
    torch_output = model(dummy_input)

print("PyTorch:", torch_output[0][:5])
print("ONNX Runtime:", ort_output[0][0][:5])
