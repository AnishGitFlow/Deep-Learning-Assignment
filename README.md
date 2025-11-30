# Deep-Learning-Assignment

This repository contains two deep learning projects demonstrating different neural network architectures and applications.

---

## Table of Contents
- [Assignment 1: Rainfall Prediction](#assignment-1-rainfall-prediction)
- [Assignment 2: Animal Image Classification](#assignment-2-animal-image-classification)
- [Requirements](#requirements)
- [Results Summary](#results-summary)

---

## Assignment 1: Rainfall Prediction

### Overview
Binary classification project predicting rainfall occurrence using meteorological features and a feedforward neural network.

### Dataset
- **Training samples**: 2,190
- **Test samples**: 730
- **Features**: 11 meteorological variables
  - Pressure, temperature (max/min/mean), dewpoint
  - Humidity, cloud cover, sunshine hours
  - Wind direction and speed
- **Target**: Binary (0 = No rain, 1 = Rain)

### Model Architecture
Feedforward Neural Network with regularization:
- **Input Layer**: 11 features
- **Hidden Layer 1**: 64 neurons (ReLU) + L2 regularization (λ=0.001) + 30% Dropout
- **Hidden Layer 2**: 32 neurons (ReLU) + L2 regularization (λ=0.001) + 30% Dropout
- **Output Layer**: 1 neuron (Sigmoid activation)
- **Total Parameters**: 2,881

### Training Configuration
- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Binary Cross-Entropy
- **Batch Size**: 16
- **Epochs**: 50 (with early stopping, patience=5)
- **Data Split**: 80% train, 20% validation (stratified)

### Preprocessing
1. Missing value imputation (median strategy)
2. Feature standardization (StandardScaler)
3. Stratified train-validation split

### Results
| Metric | Training | Validation | Best |
|--------|----------|------------|------|
| **Accuracy** | 87.0% | 87.2% | Epoch 25 |
| **Loss** | 0.337 | 0.367 | Epoch 38 |

**Key Observations**:
- Minimal overfitting due to effective regularization
- Rapid convergence (>86% accuracy in 10 epochs)
- Training terminated at epoch 43 (early stopping)

### File Structure
```
Assignment1.ipynb    # Complete implementation notebook
```

---

## Assignment 2: Animal Image Classification

### Overview
Multi-class image classification using transfer learning with ResNet-18 to classify 10 animal categories.

### Dataset
- **Training images**: 20,938
- **Validation images**: 2,614
- **Test images**: 2,627
- **Classes** (10): butterfly, cat, chicken, cow, dog, elephant, horse, ragno (spider), sheep, squirrel
- **Image size**: 224×224 pixels (RGB)

### Model Architecture
**Transfer Learning with ResNet-18**:
- Pre-trained on ImageNet1K
- Frozen convolutional backbone (all layers)
- Modified classifier head: 512 → 10 classes
- Total architecture: Conv layers (frozen) + Global Average Pooling + FC (trainable)

### Data Augmentation
**Training transformations**:
- RandomResizedCrop(224)
- RandomHorizontalFlip()
- ColorJitter (brightness, contrast, saturation ±20%)
- Normalization (ImageNet mean/std)

**Validation/Test transformations**:
- Resize(256) → CenterCrop(224)
- Normalization (ImageNet mean/std)

### Training Configuration
- **Framework**: PyTorch
- **Hardware**: NVIDIA Quadro P1000 GPU
- **Optimizer**: Adam (lr=0.001, only FC layer parameters)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 15
- **Scheduler**: ReduceLROnPlateau (factor=0.3, patience=3)

### Results

#### Overall Performance
| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 90.93% | 96.90% | **96.42%** |
| **Loss** | 0.276 | 0.099 | 0.113 |

#### Per-Class Performance (Test Set)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **butterfly** | 0.95 | 0.97 | 0.96 | 212 |
| **cat** | 0.97 | 0.99 | 0.98 | 168 |
| **chicken** | 0.97 | 0.98 | 0.98 | 311 |
| **cow** | 0.89 | 0.95 | 0.92 | 188 |
| **dog** | 0.96 | 0.97 | 0.97 | 487 |
| **elephant** | 0.99 | 0.97 | 0.98 | 146 |
| **horse** | 0.98 | 0.93 | 0.96 | 263 |
| **ragno** | 0.99 | 0.98 | 0.98 | 483 |
| **sheep** | 0.93 | 0.91 | 0.92 | 182 |
| **squirrel** | 0.99 | 0.97 | 0.98 | 187 |
| **Weighted Avg** | **0.96** | **0.96** | **0.96** | **2,627** |

**Key Observations**:
- Excellent transfer learning performance (>95% accuracy in epoch 1)
- Minimal overfitting (validation > training accuracy)
- Best performers: ragno, elephant, squirrel (F1 ≥ 0.98)
- Slight confusion between cow/sheep (visual similarity)

### File Structure
```
Assignment2.ipynb    # Complete implementation notebook
```

---

## Requirements

### Assignment 1 (Rainfall Prediction)
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tensorflow>=2.8.0
scikeras>=0.9.0
```

### Assignment 2 (Animal Classification)
```
numpy>=1.21.0
torch>=1.10.0
torchvision>=0.11.0
Pillow>=8.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
```

---

## Results Summary

### Comparative Overview

| Assignment | Task | Architecture | Accuracy | Parameters | Training Time |
|------------|------|--------------|----------|------------|---------------|
| **1** | Rainfall Prediction | Feedforward NN | 87.2% | 2,881 | ~43 epochs |
| **2** | Animal Classification | ResNet-18 (Transfer) | 96.4% | ~11M (512 trainable) | 15 epochs |

### Techniques Demonstrated

## Project Structure

```
.
├── Assignment1.ipynb          # Rainfall prediction notebook
├── Assignment2.ipynb          # Animal classification notebook
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── dataset/                   # Data directories (not included)
    ├── rainfall/
    │   ├── train.csv
    │   └── test.csv
    └── animals/
        ├── training_set/
        ├── val/
        ├── test_set/
        └── single_prediction/
```
