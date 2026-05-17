import os, shutil, json, zipfile
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.settings import SEED, LOCKED_CLASSES, DATA_DIR, IMG_SIZE

def prepare_data():
    """Download, filter, and split the PlantVillage dataset."""
    zip_path = os.path.join(DATA_DIR, "plantvillage-dataset.zip")
    raw_path = os.path.join(DATA_DIR, "raw")

    # Check if we need to download
    if not os.path.exists(zip_path) and not (os.path.exists(raw_path) and os.listdir(raw_path)):
        kaggle_config = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
        if not os.path.exists(kaggle_config):
            raise FileNotFoundError(
                f"Kaggle API credentials not found and no local dataset found. "
                "Please provide 'data/plantvillage-dataset.zip' manually."
            )
        print("Downloading dataset from Kaggle...")
        os.system(f"kaggle datasets download -d abdallahalidev/plantvillage-dataset -p {DATA_DIR} --force")
    
    zip_path = os.path.join(DATA_DIR, "plantvillage-dataset.zip")
    raw_path = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_path, exist_ok=True)

    if os.path.exists(zip_path):
        if not os.listdir(raw_path):
            print(f"Extracting dataset archive to {raw_path}...")
            with zipfile.ZipFile(zip_path, 'r') as archive:
                archive.extractall(raw_path)
        else:
            print(f"Raw data already extracted at {raw_path}")
    else:
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}. Please download it manually with Kaggle.")

    split_path = os.path.join(DATA_DIR, "plant_village")
    
    # Locate actual image directory
    potential_roots = [
        os.path.join(raw_path, "plantvillage dataset", "color"),
        os.path.join(raw_path, "plantvillage-dataset", "color"),
        os.path.join(raw_path, "plantvillage dataset"),
        os.path.join(raw_path, "plantvillage-dataset"),
        raw_path,
    ]
    actual_root = next((p for p in potential_roots if os.path.exists(p)), raw_path)
    if not os.path.exists(actual_root):
        raise FileNotFoundError(f"Could not locate extracted dataset in {raw_path}")

    counts = {}
    print("Filtering and splitting classes...")
    import re
    def clean(s): return re.sub(r'[^a-z0-9]', '', s.lower())

    found_classes = []
    for cls in LOCKED_CLASSES:
        cls_clean = clean(cls)
        # Flexible matching
        match = None
        for d in os.listdir(actual_root):
            d_clean = clean(d)
            # Match if one is a subset of the other or they share core words
            if cls_clean in d_clean or d_clean in cls_clean or \
               (clean(cls.split('___')[0]) in d_clean and clean(cls.split('___')[-1]) in d_clean):
                match = d
                break
                
        if not match: 
            print(f"Warning: Class {cls} not found in {actual_root}.")
            continue
        
        imgs = [os.path.join(actual_root, match, f) for f in os.listdir(os.path.join(actual_root, match)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs:
            print(f"Warning: No images found for {cls}")
            continue

        print(f"Processing {cls}: {len(imgs)} images")
        found_classes.append(cls)
        
        train, temp = train_test_split(imgs, test_size=0.30, random_state=SEED)
        val, test = train_test_split(temp, test_size=0.50, random_state=SEED)
        
        counts[cls] = len(imgs)
        for name, data in [('train', train), ('val', val), ('test', test)]:
            dst = os.path.join(split_path, name, cls)
            os.makedirs(dst, exist_ok=True)
            for f in data: shutil.copy2(f, dst)
    
    # Update LOCKED_CLASSES in memory for this run if needed, 
    # but we should really stick to the global one.
    # We will return the found classes to the training script.
    
    with open(os.path.join(DATA_DIR, "class_counts.json"), 'w') as f:
        json.dump(counts, f, indent=2)
    
    print("Data preparation complete.")
    return counts, found_classes

def get_augmentation_pipeline():
    """Build a tf.data augmentation pipeline."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.042), # ±15 degrees
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])

if __name__ == "__main__":
    prepare_data()
