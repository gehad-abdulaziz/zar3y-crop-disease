import os
import tensorflow as tf
from src.settings import IMG_SIZE, MODELS_DIR, SEED, BATCH_SIZE
from src.data_prep import get_augmentation_pipeline

def build_model(num_classes):
    """MobileNetV2 based model with proper preprocessing for better stability."""
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Use the official MobileNetV2 preprocessing
    # This scales inputs to [-1, 1] which is what the pretrained weights expect
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Use MobileNetV2 as base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def train_zar3y(train_ds, val_ds, class_names=None, class_weights=None):
    """Single-phase training for simple CNN."""
    if class_names is not None:
        num_classes = len(class_names)
    else:
        num_classes = len(train_ds.class_names)
    
    model, base_model = build_model(num_classes)
    
    # Single phase training
    print("Training model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint_path = os.path.join(MODELS_DIR, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
    
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks, class_weight=class_weights)
    print(f"Model saved to {checkpoint_path}")
    return model
