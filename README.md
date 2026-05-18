# Zar3y — Crop Disease Detection from Phone Photos 🌿

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)

**Zar3y ("my crop")** is an end-to-end AI solution designed for Egyptian smallholder farmers to diagnose crop diseases instantly using their phone cameras. 

## 📖 The Scenario
*Hassan, a smallholder farmer in Beheira, sees yellow spots on his tomato leaves. He doesn’t know if it’s a nutrient deficiency or a fungal infection. With Zar3y, he takes a photo and gets a diagnosis in seconds, allowing him to decide whether to apply treatment before the crop is lost.*

---

## ✨ Key Features
- **Mobile-Ready**: Quantized TFLite INT8 model optimized for edge deployment.
- **Explainable AI**: Integrated **Grad-CAM** overlays to show which part of the leaf the AI focused on.
- **Robust Performance**: Trained on 10 specific classes relevant to the Egyptian agricultural landscape.
- **Zero-Cost Infrastructure**: Optimized to run on free-tier Google Colab and local CPU.

## 🏗️ Architecture & Stack
- **Base Model**: MobileNetV2 (Transfer Learning)
- **Quantization**: Post-Training INT8 Quantization (3.6x size reduction)
- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping)

---

## 📁 Project Structure
```text
zar3y-crop-disease/
├── app.py                  # Streamlit Frontend Demo
├── backend/
│   └── main.py             # FastAPI Backend Service
├── src/
│   ├── data_prep.py        # Dataset Filtering & Augmentation
│   ├── train.py            # Transfer Learning Pipeline
│   ├── evaluate.py         # Performance Metrics & Confusion Matrix
│   ├── quantize_tflite.py  # INT8 Quantization & Benchmarking
│   ├── grad_cam.py         # Explainability Engine
│   └── inference.py        # TFLite Inference Wrapper
├── models/                 # Saved .keras and .tflite models
├── outputs/                # Evaluation plots and reports
└── data/                   # PlantVillage Dataset (Subset)
```

## 📊 Performance Results
| Metric | Value |
| :--- | :--- |
| **Overall Test Accuracy** | **95.1%** |
| **Model Size (FP32)** | 9.33 MB |
| **Model Size (INT8)** | **2.62 MB** |
| **Inference Latency** | **< 200ms (CPU)** |

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Kaggle API Key (to download PlantVillage)

### 2. Installation
```bash
git clone https://github.com/yourusername/zar3y-crop-disease.git
cd zar3y-crop-disease
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
```bash
# This will handle data prep, training, evaluation, and quantization
python train_zar3y.py
```

### 4. Launch the App
Start the Backend:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Start the Frontend:
```bash
streamlit run app.py
```

---

## 🛠️ Requirements & Toolbox Alignment
- ✅ **Requirement 1**: 70/15/15 Split, Data Augmentation implemented.
- ✅ **Requirement 2**: MobileNetV2 Transfer Learning with Early Stopping.
- ✅ **Requirement 3**: INT8 Quantization with representative dataset.
- ✅ **Requirement 4**: Grad-CAM integration for interpretability.
- ✅ **Requirement 5**: FastAPI + Streamlit deployment.

## 📚 Acknowledgments
- Dataset: [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Pre-trained Models: TensorFlow Hub / Keras Applications
- Frameworks: Sprints Graduation Project Guidelines
- models here https: https://drive.google.com/drive/folders/1MHF0ElBTxc7KoUCC7EFpbcktv4jN0RHU?usp=drive_link
