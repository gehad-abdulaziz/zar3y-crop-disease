# Zar3y Final Report 📄

## 1. Executive Summary
Zar3y is a computer vision solution for detecting diseases in major Egyptian crops. The model uses transfer learning on MobileNetV2 and is optimized for mobile deployment via INT8 quantization.

## 2. Methodology
- **Dataset**: PlantVillage (Subset of 10 classes).
- **Split**: 70% Train, 15% Val, 15% Test (Seed 42).
- **Architecture**: MobileNetV2 backbone with a custom Dense head.
- **Training**: 15 epochs with Early Stopping and learning rate reduction.

## 3. Results
- **Test Accuracy**: 95.1%
- **Macro F1-Score**: 94.8%
- **Latency**: ~180ms per image on local CPU.

## 4. Quantization
- **FP32 Model**: 9.3 MB
- **INT8 Model**: 2.6 MB
- **Compression Ratio**: 3.6x

## 5. Explainability
Grad-CAM was implemented to visualize the model's focus. Most correct predictions highlight the lesions and discoloration on the leaf surface.

## 6. Limitations & Future Work
- Currently limited to 10 classes.
- Needs more out-of-distribution (OOD) testing with real field photos from different soil types.
