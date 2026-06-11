# CIFAR-10 Image Classification using CNN (PyTorch)

## 📌 Overview
This project implements a Convolutional Neural Network for CIFAR-10 image classification using PyTorch. It includes data augmentation, training pipeline, evaluation metrics, and visualization.

---

## 📊 Dataset
- CIFAR-10 (10 classes)
- 50,000 training images
- 10,000 test images
- Image size resized to 64×64

---

## 🧠 Model
Custom CNN:
- 3 Conv layers
- ReLU activations
- MaxPooling
- Fully connected layers

---

## 🔄 Data Augmentation
- RandomCrop
- RandomHorizontalFlip
- RandomRotation
- Normalization

---

## 📈 Outputs
Saved automatically in `/outputs`:
- accuracy.png
- confusion_matrix.png

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```



### 2. Train the model
```bash
python main.py
```
### 3. Evaluate the model
```
python evaluate.py
```

### Performance:
- Train Accuracy: 79.18%
- Validation Accuracy: 80.66%
- Final Test Accuracy: 80.66%

### Class-wise Performance:
Best performing classes: Car, Ship, Truck  
Challenging classes: Cat, Dog, Bird