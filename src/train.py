import os
import tensorflow as tf
from src.settings import IMG_SIZE, MODELS_DIR, SEED, BATCH_SIZE
from src.data_prep import build_dataset


def build_model(num_classes):
    """
    MobileNetV2-based classifier.

    Input contract:
      - Expects raw pixel values in [0, 255]  (NOT pre-divided by 255).
      - mobilenet_v2.preprocess_input converts [0,255] → [-1,1] internally.
      - Augmentation is intentionally absent here; it is applied upstream
        in the tf.data pipeline (build_dataset with augment=True) so it
        never runs at inference or Grad-CAM time.
    """
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # ── Normalization ────────────────────────────────────────────────────────
    # preprocess_input maps raw [0,255] → [-1,1] as the pretrained weights expect.
    # If you ever pass [0,1] images here you will get wrong activations and the
    # model will fall back to background shortcuts.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # ── Feature extractor ────────────────────────────────────────────────────
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # frozen during Stage 1 warm-up

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def train_zar3y(train_ds, val_ds, class_names=None, class_weights=None):
    """
    Two-phase training:
      Stage 1 — warm up the classification head (base frozen, 3 epochs).
      Stage 2 — fine-tune top layers of MobileNetV2 (low LR, up to 15 epochs).

    Both datasets must supply raw [0,255] pixels — see build_dataset().
    """
    num_classes = len(class_names) if class_names is not None else len(train_ds.class_names)

    model, base_model = build_model(num_classes)

    # ── Stage 1: Head warm-up ────────────────────────────────────────────────
    print("\n--- Stage 1: Warm-up classification head (base model frozen) ---")
    base_model.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        class_weight=class_weights
    )

    # ── Stage 2: Fine-tuning ─────────────────────────────────────────────────
    print("\n--- Stage 2: Fine-tuning base model (unfreezing top layers) ---")
    base_model.trainable = True

    # Keep the early general-purpose feature extractors frozen;
    # only adapt the high-level layers to plant-disease features.
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"Layers unfrozen from index {fine_tune_at} onwards "
          f"({len(base_model.layers) - fine_tune_at} layers trainable)")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    checkpoint_path = os.path.join(MODELS_DIR, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=4, restore_best_weights=True, monitor='val_loss'
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
        class_weight=class_weights
    )

    print(f"Model saved to {checkpoint_path}")
    return model