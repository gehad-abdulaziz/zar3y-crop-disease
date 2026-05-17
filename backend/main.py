import os
# CRITICAL: Disable XNNPACK before any other imports to avoid initialization crashes on this system
os.environ['TFLITE_DISABLE_XNNPACK'] = '1'

from fastapi import FastAPI, UploadFile, File
import uvicorn
import io
from src.inference import Zar3yInference
from src.settings import MODELS_DIR, LOCKED_CLASSES

import base64
from src.grad_cam import get_gradcam_heatmap, display_gradcam
import numpy as np
import cv2

app = FastAPI(title="Zar3y Backend")

# Load TFLite Model for fast inference
TFLITE_PATH = os.path.join(MODELS_DIR, "model_quantized.tflite")
engine = Zar3yInference(TFLITE_PATH) if os.path.exists(TFLITE_PATH) else None

# Load Keras Model for Grad-CAM (since TFLite doesn't support gradients)
KERAS_PATH = os.path.join(MODELS_DIR, "best_model.keras")
if not os.path.exists(KERAS_PATH):
    KERAS_PATH = os.path.join(MODELS_DIR, "best_model.h5")

keras_model = None
if os.path.exists(KERAS_PATH):
    try:
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(KERAS_PATH)
        print(f"Keras model loaded from {KERAS_PATH} for Grad-CAM")
    except Exception as e:
        print(f"Failed to load Keras model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if engine is None:
        return {"error": "Model not found. Run training script first."}
    
    contents = await file.read()
    class_idx, confidence = engine.predict(io.BytesIO(contents))
    
    class_name = LOCKED_CLASSES[class_idx]
    
    # Generate Grad-CAM if Keras model is available
    heatmap_b64 = None
    if keras_model:
        try:
            # Prepare image for Grad-CAM
            from PIL import Image
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            # Keras model has rescaling layer
            img_batch = np.expand_dims(img_array, axis=0)
            
            heatmap = get_gradcam_heatmap(img_batch, keras_model)
            gradcam_img = display_gradcam(img_array, heatmap)
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

    return {
        "class_name": class_name,
        "confidence": confidence,
        "heatmap": heatmap_b64,
        "explanation": get_explanation(class_name)
    }

def get_explanation(class_name):
    explanations = {
        "Tomato___Late_blight": "Serious fungal disease. Apply copper-based fungicides immediately and improve air circulation.",
        "Tomato___Early_blight": "Common fungus. Remove infected lower leaves and use mulch to prevent soil splashing.",
        "Tomato___healthy": "Your tomato plant looks healthy! Keep up the good work.",
        "Potato___Late_blight": "Highly contagious. Remove infected plants and apply fungicide to remaining ones.",
        "Corn___Common_rust": "Fungal infection. Usually doesn't kill the plant, but can reduce yield. Consider resistant varieties next season.",
    }
    return explanations.get(class_name, f"{class_name.replace('___', ' ')} detected. Monitor closely and ensure proper irrigation.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
