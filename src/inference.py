import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Global fix for XNNPACK failure on some Windows environments
os.environ['TFLITE_DISABLE_XNNPACK'] = '1'

class Zar3yInference:
    def __init__(self, model_path):
        self.input_details = None
        self.output_details = None
        self.keras_model = None

        # Try loading TFLite first
        try:
            print("Attempting to load TFLite model...")
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                experimental_delegates=[],
                num_threads=1
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("Model loaded successfully (TFLite)")
        except Exception as e:
            print(f"TFLite load failed: {e}. Attempting Keras fallback...")
            try:
                # Try .keras first, then .h5
                keras_path = model_path.replace(".tflite", ".keras")
                if not os.path.exists(keras_path):
                    keras_path = model_path.replace(".tflite", ".h5")
                
                if not os.path.exists(keras_path):
                    keras_path = os.path.join(os.path.dirname(model_path), "best_model.keras")
                if not os.path.exists(keras_path):
                    keras_path = os.path.join(os.path.dirname(model_path), "best_model.h5")
                
                if os.path.exists(keras_path):
                    self.keras_model = tf.keras.models.load_model(keras_path)
                    print(f"Model loaded successfully (Keras: {keras_path})")
                else:
                    raise FileNotFoundError(f"Could not find Keras model (.keras or .h5)")
            except Exception as e2:
                print(f"CRITICAL: All model loading attempts failed: {e2}")
                raise e2
        
    def predict(self, image_bytes):
        """Run inference on image bytes."""
        # Preprocess
        img = Image.open(image_bytes).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        # Normalization is now handled by the Rescaling layer within the model itself.
        
        if self.keras_model:
            # Keras inference expects float32 [0, 1] and batch dimension
            img_batch = np.expand_dims(img_array.astype(np.float32), axis=0)
            output_data = self.keras_model.predict(img_batch, verbose=0)
            confidences = output_data[0]
        else:
            # TFLite inference
            input_dtype = self.input_details[0]['dtype']
            if input_dtype == np.uint8 or input_dtype == np.int8:
                # Get quantization parameters
                scale, zero_point = self.input_details[0]['quantization']
                if scale == 0: scale = 1.0 # Avoid division by zero
                
                # Quantize to [0, 255] or [-128, 127]
                if input_dtype == np.uint8:
                    img_array = (img_array / scale + zero_point).astype(np.uint8)
                else:
                    img_array = (img_array / scale + zero_point).astype(np.int8)
            
            img_batch = np.expand_dims(img_array, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], img_batch)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Handle quantized output
            output_dtype = self.output_details[0]['dtype']
            if output_dtype == np.uint8:
                # Dequantize from uint8
                scale, zero_point = self.output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - zero_point) * scale
            
            confidences = output_data[0]
        
        # IMPORTANT: Model already has softmax activation, do NOT apply it again.
        class_idx = np.argmax(confidences)
        
        return class_idx, float(confidences[class_idx])
