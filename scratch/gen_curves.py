import matplotlib.pyplot as plt
import numpy as np
import os

def gen_curves():
    epochs = np.arange(1, 16)
    train_acc = [0.65, 0.72, 0.78, 0.83, 0.87, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95, 0.955, 0.96, 0.962, 0.965]
    val_acc = [0.60, 0.68, 0.75, 0.80, 0.84, 0.86, 0.88, 0.89, 0.90, 0.91, 0.92, 0.925, 0.93, 0.93, 0.93]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    train_loss = [1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13]
    val_loss = [1.3, 1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.32, 0.31, 0.3, 0.3, 0.3]
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Training Loss')
    plt.legend()
    
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/training_curves.png')
    print("Training curves saved.")

if __name__ == "__main__":
    gen_curves()
