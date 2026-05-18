import os
import tensorflow as tf
import matplotlib.pyplot as plt
from src.data_prep import get_augmentation_pipeline
from src.settings import DATA_DIR, LOCKED_CLASSES

def generate_samples():
    print("Generating augmentation samples...")
    train_dir = os.path.join(DATA_DIR, "plant_village/train")
    # Load a small batch
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=1,
        shuffle=True,
        class_names=LOCKED_CLASSES
    )
    
    aug = get_augmentation_pipeline()
    
    plt.figure(figsize=(10, 10))
    for images, _ in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = aug(images)
            plt.imshow(augmented_image[0].numpy().astype("uint8"))
            plt.axis("off")
            
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/augmentation_samples.png")
    print("Augmentation samples saved to outputs/augmentation_samples.png")

if __name__ == "__main__":
    generate_samples()
