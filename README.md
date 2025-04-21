# Conditional Vision Transformer (CViT) for Traffic Sign Classification 🚦

This repository contains the implementation of a **Conditional Vision Transformer (CViT)** designed for accurate and robust traffic sign classification using the [GTSRB (German Traffic Sign Recognition Benchmark)]([https://benchmark.ini.rub.de/gtsrb_dataset.html](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)) dataset. This model introduces a novel conditional attention mechanism that dynamically adapts the attention weights based on input context, significantly improving performance on misclassification-prone classes.

---

## 🚀 Key Features

- ✅ Conditional Attention Mechanism  
- ✅ Custom Patch Embedding and Tokenization  
- ✅ Gating Network for Adaptive QKV Generation  
- ✅ Superior Accuracy and Generalization  
- ✅ Well-structured and clean code, ready for training and evaluation

---

## 🧠 Model Overview

The Conditional ViT architecture is composed of three core phases:

1. **Preprocessing Phase**  
   - Patch extraction and embedding  
   - Positional encoding  
   - Token sequence formation  

2. **Feature Extraction Phase**  
   - Conditional attention blocks with gating mechanisms  
   - Adaptive generation of Query, Key, and Value matrices  

3. **Classification Phase**  
   - Fully connected layers  
   - Softmax output for multi-class classification  

---

## 📊 Results

| Model           | Accuracy (%) |
|----------------|--------------|
| Simple ViT      | 94.25        |
| **Conditional ViT (Proposed)** | **99.87**        |

The proposed model significantly reduces misclassification in challenging traffic sign classes such as T21 and T2, ensuring more stable and interpretable predictions.

---

