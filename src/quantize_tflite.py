import os
import time
import tensorflow as tf
import numpy as np
from src.settings import MODELS_DIR, OUTPUTS_DIR

def quantize_and_benchmark(train_ds, model=None):
    """Convert to TFLite INT8 and benchmark size + latency."""
    model_path = os.path.join(MODELS_DIR, "best_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "best_model.h5")
        
    if model is None:
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model for quantization from {model_path}: {e}")
            raise e
    
    # INT8 Post-Training Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset (200 images)
    def representative_data_gen():
        for input_value, _ in train_ds.unbatch().take(200):
            yield [np.expand_dims(input_value.numpy(), axis=0)]
            
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    print("Converting to TFLite INT8...")
    tflite_model = converter.convert()
    tflite_path = os.path.join(MODELS_DIR, "model_quantized.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    # Benchmarking
    size_h5 = os.path.getsize(model_path) / (1024 * 1024)
    size_tflite = os.path.getsize(tflite_path) / (1024 * 1024)
    
    benchmark = {
        "model_size_mb": {
            "h5": round(size_h5, 2),
            "tflite_int8": round(size_tflite, 2),
            "compression_ratio": round(size_h5 / size_tflite, 1)
        }
    }
    
    print(f"Quantized model saved to {tflite_path}")
    return benchmark
