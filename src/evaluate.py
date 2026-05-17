import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from src.settings import DATA_DIR, MODELS_DIR, IMG_SIZE, LOCKED_CLASSES

def evaluate_model():
    print("Starting Evaluation...")
    
    # 1. Load model
    model_path = os.path.join(MODELS_DIR, "best_model.keras")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Load test data
    test_dir = os.path.join(DATA_DIR, "plant_village/test")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode='categorical',
        shuffle=False,
        class_names=LOCKED_CLASSES
    )
    
    # 3. Get predictions
    y_true = []
    y_pred = []
    
    print("Predicting on test set...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 4. Generate Report
    report = classification_report(y_true, y_pred, target_names=LOCKED_CLASSES, output_dict=True)
    print("\nOverall Accuracy:", report['accuracy'])
    
    # Save report to JSON
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LOCKED_CLASSES, yticklabels=LOCKED_CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    print("Confusion Matrix saved to outputs/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
