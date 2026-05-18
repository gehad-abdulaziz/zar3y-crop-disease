import os
os.environ['TFLITE_DISABLE_XNNPACK'] = '1'

import tensorflow as tf
from src.settings import MODELS_DIR

model_path = os.path.join(MODELS_DIR, "model_quantized.tflite")
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
else:
    interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[], num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input Details:")
    for detail in input_details:
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Dtype: {detail['dtype']}")
        print(f"  Quantization: {detail['quantization']}")
    
    print("\nOutput Details:")
    for detail in output_details:
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Dtype: {detail['dtype']}")
        print(f"  Quantization: {detail['quantization']}")
