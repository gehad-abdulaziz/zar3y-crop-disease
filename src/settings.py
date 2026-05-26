import os

# Locked Project Settings
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LOCKED_CLASSES = [
    'Corn___Common_rust',
    'Pepper_bell___Bacterial_spot',
    'Pepper_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___healthy'
]

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)