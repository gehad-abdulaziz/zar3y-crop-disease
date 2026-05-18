import os
import tensorflow as tf
import numpy as np
from src.settings import MODELS_DIR, DATA_DIR

def requantize():
    model_path = os.path.join(MODELS_DIR, "best_model.keras")
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Use real data for representative dataset
    train_data_path = os.path.join(DATA_DIR, "plant_village/train")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_path,
        image_size=(224, 224),
        batch_size=32
    )
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_data_gen():
        for input_value, _ in train_ds.unbatch().take(100):
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
    print(f"Fresh TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    requantize()
