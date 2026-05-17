import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """Generate Grad-CAM heatmap for the given image and model using manual forward pass."""
    
    # Check if the model has a nested base_model (like MobileNetV2)
    inner_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            inner_model = layer
            break

    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        x = img_tensor
        last_conv_layer_output = None
        
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            try:
                if layer == inner_model:
                    x = layer(x, training=False)
                    last_conv_layer_output = x
                    tape.watch(last_conv_layer_output)
                else:
                    x = layer(x, training=False)
            except TypeError:
                if layer == inner_model:
                    x = layer(x)
                    last_conv_layer_output = x
                    tape.watch(last_conv_layer_output)
                else:
                    x = layer(x)
                
            if inner_model is None and last_conv_layer_name is not None and layer.name == last_conv_layer_name:
                last_conv_layer_output = x
                tape.watch(last_conv_layer_output)

        preds = x
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    if last_conv_layer_output is None:
        raise ValueError("Could not find the last conv layer output.")

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def display_gradcam(img, heatmap, alpha=0.4):
    """Superimpose heatmap on original image."""
    img = np.uint8(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    superimposed_img = jet * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img
