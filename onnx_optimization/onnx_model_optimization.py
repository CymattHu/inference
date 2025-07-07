import onnx
from onnxoptimizer import optimize

model = onnx.load("model.onnx")
passes = ["eliminate_deadend", "eliminate_identity", "eliminate_nop_dropout", "eliminate_nop_transpose"]
optimized_model = optimize(model, passes)
onnx.save(optimized_model, "model_optimized.onnx")