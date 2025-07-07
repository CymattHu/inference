import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model_optimized_simplified.onnx")
input_name = session.get_inputs()[0].name

dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})

print(outputs)