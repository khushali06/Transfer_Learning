# Transfer Learning with Pre-trained Models on CIFAR-100

This repository contains a complete pipeline and implementation for applying transfer learning to the **CIFAR-100** dataset using popular pre-trained Convolutional Neural Networks (CNNs). By leveraging models trained on the large-scale ImageNet dataset, this project adapts powerful feature extractors to tackle finer-grained image classification across 100 distinct classes, reducing training time and improving convergence compared to training from scratch.

---

## 🚀 Key Features & Models Explored
The notebook explores and evaluates three standard pre-trained architectures:
1. **ResNet50**
2. **VGG16**
3. **MobileNetV2**

---

## 📊 Project Workflow

1. **Data Loading and Preprocessing:**
   * Loads the CIFAR-100 dataset (`32x32` images).
   * Applies specific preprocessing functions tailored for each architecture (`preprocess_input` for ResNet50, VGG16, and MobileNetV2).

2. **Model Preparation & Custom Architecture:**
   * Loads base models with `weights='imagenet'` and `include_top=False`.
   * Appends custom classification heads (Global AveragePooling2D, Dense layers with ReLU activation, and a Softmax output layer for 100 classes).
   * Initially freezes the pre-trained base layers to retain learned feature representations.

3. **Fine-Tuning & Training:**
   * Unfreezes the top-most layers of the base models to allow fine-tuning on CIFAR-100 specific features.
   * Compiles models using the `adam` optimizer and `sparse_categorical_crossentropy` loss.
   * Trains the networks while tracking training and validation accuracy/loss.

4. **Evaluation & Visualization:**
   * Evaluates each model on the held-out test set.
   * Includes utility functions to plot training history curves (Accuracy & Loss).

---

## 🛠️ Requirements & Installation

To run this notebook, ensure you have the following dependencies installed:

```bash
pip install tensorflow matplotlib numpy
