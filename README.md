# ONNX 模型优化指南

本文档介绍如何对导出的 ONNX 模型进行优化，以提升推理性能和减少模型体积。

## 目录

- 环境准备
- ONNX 模型导出
- ONNX 模型优化
- 使用 ONNX Runtime 进行推理
- 参考资料

---

## 环境准备

请确保已安装以下 Python 包：
```bash
pip install onnx onnxruntime onnxoptimizer onnx-simplifier
```

---

## ONNX 模型导出

示例 PyTorch 模型导出：
```bash
import torch
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).to(device).eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    input_names=["input"], 
    output_names=["output"],
    opset_version=12,
)
```
---

## ONNX 模型优化

对导出的 ONNX 模型进行优化，主要包括：

1. 使用 onnxoptimizer 进行图优化
```bash
import onnx
from onnxoptimizer import optimize

model = onnx.load("model.onnx")
passes = ["eliminate_deadend", "eliminate_identity", "eliminate_nop_dropout", "eliminate_nop_transpose"]
optimized_model = optimize(model, passes)
onnx.save(optimized_model, "model_optimized.onnx")
```
2. 使用 onnx-simplifier 简化模型
```bash
python3 -m onnxsim model_optimized.onnx model_optimized_simplified.onnx
```
---

## 使用 ONNX Runtime 进行推理
```bash
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model_optimized_simplified.onnx")
input_name = session.get_inputs()[0].name

dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})

print(outputs)
```
---

## 参考资料

- ONNX 官方文档 https://onnx.ai/
- ONNX Runtime GitHub https://github.com/microsoft/onnxruntime
- onnx-simplifier https://github.com/daquexian/onnx-simplifier

---
