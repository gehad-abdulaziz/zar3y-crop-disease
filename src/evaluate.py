import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from src.settings import DATA_DIR, MODELS_DIR, IMG_SIZE, LOCKED_CLASSES, BATCH_SIZE
from src.data_prep import build_dataset


def evaluate_model():
    print("Starting evaluation...")

    # ── Load model ───────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "best_model.keras")
    model = tf.keras.models.load_model(model_path)

    # ── Load test data ───────────────────────────────────────────────────────
    # build_dataset with augment=False returns raw [0,255] pixels, which is
    # exactly what the model's internal preprocess_input layer expects.
    # Never add a manual / 255 rescale here.
    test_dir = os.path.join(DATA_DIR, "plant_village/test")
    test_ds = build_dataset(
        test_dir,
        class_names=LOCKED_CLASSES,
        batch_size=BATCH_SIZE,
        augment=False       # no augmentation at eval time
    )

    # ── Predictions ─────────────────────────────────────────────────────────
    y_true, y_pred = [], []
    print("Predicting on test set...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Classification report ────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=LOCKED_CLASSES, output_dict=True
    )
    print(f"\nOverall Accuracy: {report['accuracy']:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Evaluation report saved to outputs/eval_report.json")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=LOCKED_CLASSES, yticklabels=LOCKED_CLASSES
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    print("Confusion matrix saved to outputs/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()