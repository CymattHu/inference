import torch
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)  # 这里以 ResNet18 为例
model = model.to(device)
model.eval()  # 切换到评估模式

dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(model, dummy_input, "model.onnx")