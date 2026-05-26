import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate a Grad-CAM heatmap.

    img_array must be raw pixels in [0, 255] with shape (1, H, W, 3).
    The model's internal preprocess_input layer handles normalization,
    so do NOT divide by 255 before calling this function.

    Because augmentation is no longer inside the model, the forward pass
    here is deterministic and the heatmap correctly reflects the actual
    input image rather than a randomly-transformed version of it.
    """

    # Find the nested base_model (MobileNetV2) if present
    inner_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            inner_model = layer
            break

    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        tape.watch(img_tensor)

        x = img_tensor
        last_conv_layer_output = None

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            is_last_dense = (
                layer == model.layers[-1] and
                isinstance(layer, tf.keras.layers.Dense)
            )

            try:
                if layer == inner_model:
                    # Run the base model in inference mode (no dropout/BN randomness)
                    x = layer(x, training=False)
                    last_conv_layer_output = x
                    tape.watch(last_conv_layer_output)
                elif is_last_dense:
                    # Bypass softmax to compute raw logits — required for Grad-CAM
                    x = tf.matmul(x, layer.kernel) + layer.bias
                else:
                    x = layer(x, training=False)

            except TypeError:
                # Some custom layers don't accept the training kwarg
                if layer == inner_model:
                    x = layer(x)
                    last_conv_layer_output = x
                    tape.watch(last_conv_layer_output)
                elif is_last_dense:
                    x = tf.matmul(x, layer.kernel) + layer.bias
                else:
                    x = layer(x)

            # Fallback: capture by name when there is no nested model
            if (inner_model is None and
                    last_conv_layer_name is not None and
                    layer.name == last_conv_layer_name):
                last_conv_layer_output = x
                tape.watch(last_conv_layer_output)

        preds = x

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    if last_conv_layer_output is None:
        raise ValueError(
            "Could not find the last conv layer output. "
            "Pass last_conv_layer_name explicitly if the model has no nested tf.keras.Model."
        )

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def display_gradcam(img, heatmap, alpha=0.4):
    """
    Superimpose the Grad-CAM heatmap on the original image.

    img should be in [0, 255] uint8 or float32.
    Returns a uint8 RGB image.
    """
    img = np.float32(img)
    heatmap_uint8 = np.uint8(255 * heatmap)

    # applyColorMap returns BGR; convert to RGB to match our pipeline
    jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    jet_rgb = np.float32(cv2.resize(jet_rgb, (img.shape[1], img.shape[0])))

    superimposed = jet_rgb * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255)
    return np.uint8(superimposed)


def visualize_gradcam(image_path, model, class_names, alpha=0.4):
    """
    Convenience wrapper: load an image, run inference, and plot the Grad-CAM.

    image_path: path to a JPEG/PNG file
    model:      loaded Keras model (best_model.keras)
    class_names: ordered list matching the model's output classes
    """
    from PIL import Image

    img = Image.open(image_path).convert('RGB').resize((224, 224))
    # Keep as [0,255] — the model handles normalization internally
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])
    pred_class = class_names[pred_idx]

    heatmap = get_gradcam_heatmap(img_batch, model, pred_index=pred_idx)
    overlay = display_gradcam(img_array, heatmap, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\n{pred_class} ({confidence:.1%})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/gradcam_result.png", dpi=150)
    plt.show()
    print(f"Predicted: {pred_class}  |  Confidence: {confidence:.1%}")
    return pred_class, confidence, heatmap