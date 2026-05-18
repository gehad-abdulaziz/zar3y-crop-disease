import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.settings import MODELS_DIR, DATA_DIR, IMG_SIZE, LOCKED_CLASSES
from src.grad_cam import get_gradcam_heatmap, display_gradcam

def gen_gradcam_examples():
    print("Generating Grad-CAM examples...")
    model_path = os.path.join(MODELS_DIR, "best_model.keras")
    model = tf.keras.models.load_model(model_path)
    
    out_dir = "outputs/grad_cam_examples"
    os.makedirs(out_dir, exist_ok=True)
    
    test_dir = os.path.join(DATA_DIR, "plant_village/test")
    
    # Pick a few classes
    classes_to_pick = ['Tomato___Late_blight', 'Corn___Common_rust', 'Potato___Early_blight', 'Pepper_bell___Bacterial_spot', 'Tomato___healthy']
    
    for i, cls in enumerate(classes_to_pick):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        images = os.listdir(cls_dir)
        if not images:
            continue
            
        img_path = os.path.join(cls_dir, images[0])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array_expanded, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_class = LOCKED_CLASSES[pred_idx]
        
        heatmap = get_gradcam_heatmap(img_array_expanded, model)
        overlay = display_gradcam(img_array, heatmap)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_array.astype('uint8'))
        plt.title(f"Original: {cls.split('___')[-1]}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f"Grad-CAM (Pred: {pred_class.split('___')[-1]})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"example_{i}_{cls}.png"))
        plt.close()
        
    print(f"Grad-CAM examples saved to {out_dir}")

if __name__ == "__main__":
    gen_gradcam_examples()
