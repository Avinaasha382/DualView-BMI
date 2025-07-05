
---

# DualView BMI Prediction and Gender Classification

## Overview

This project predicts **Body Mass Index (BMI)** and classifies gender using dual-perspective facial images (front and side views). By leveraging the **FaceNet** model for feature extraction and machine learning models for prediction and classification, this project demonstrates the feasibility of deriving health-related metrics and demographic information from facial features.

### Features
- **BMI Prediction**: Approached as a regression problem, utilizing various machine learning models.
- **Gender Classification**: Solved as a binary classification problem.
- **Health Categorization**: BMI values categorized into health-related groups (underweight, normal, overweight).
- **Visualization**: Results visualized using scatter plots and confusion matrices.
- **Real-World Testing**: Evaluated on real-world data, including team members' images.

## Dataset

The dataset consists of:
- **60,000 facial images** with front and side views per individual.
- Associated labels:
  - Height (in meters)
  - Weight (in kilograms)
  - Gender (male/female)

BMI is calculated using the formula:

math
\text{BMI} = \frac{\text{Weight (kg)}}{\text{Height (m)}^2

## 🧹 Preprocessing Pipeline

- ✅ Face Detection & Alignment using **MTCNN**
- ✅ Resizing to 160×160 px
- ✅ Normalization of pixel values
- ✅ Augmentation: flipping, rotation, cropping
- ✅ Label mapping:
  - Underweight: BMI < 18.5
  - Normal: 18.5 ≤ BMI < 30
  - Overweight: BMI ≥ 30

---

## 🔬 Feature Extraction

- Used **FaceNet** (InceptionResnetV1) to get 512-dim embeddings
- Combined front + side = **1024-dimensional vector**
- Applied **PCA** for dimensionality reduction

---

## 🧠 Machine Learning Models

### 🧮 BMI Prediction (Regression)

| Model                 | R² Score | MAE   | RMSE  | PCC   |
|----------------------|----------|-------|-------|-------|
| Linear Regression     | 0.360    | 3.068 | 4.036 | 0.600 |
| Decision Tree         | 0.344    | 3.067 | 4.086 | 0.594 |
| Random Forest         | 0.464    | 2.874 | 3.694 | 0.718 |
| **XGBoost (Final)**   | **0.856**| **0.909** | **1.917** | **0.926** |

### 🚻 Gender Classification

| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | **95%**  | 0.96      | 0.94   | 0.95     |

---

## 📊 Visualizations

- 📌 BMI Prediction: Scatter plot of predicted vs. actual BMI
- 📌 Gender: Confusion matrix & classification report

---

## ✅ Real-World Testing

- 📷 Pipeline tested on real images from team members
- ✅ Accurate BMI and gender results on non-training data

---

## 🆚 Other Implemented Methods

| Method                        | Profile       | MAE   | RMSE  | R² Score | PCC   |
|------------------------------|---------------|-------|-------|----------|-------|
| Viola-Jones + ResNet         | Front         | 3.4047| 4.6341| 0.7823   | 0.8322|
| ResNet (Front & Side)        | Front & Side  | 3.1824| 4.3180| 0.5669   | 0.5679|
| MobileNet V2                 | Front         | 2.71  | 13.71 | 0.51     | -     |
| **Proposed (MTCNN + FaceNet + XGBoost)** | Front & Side | **0.7019** | **2.0186** | **0.8464** | **0.9202** |
| Proposed (Front Only)        | Front         | 1.8062| 2.5438| 0.7562   | 0.8902|

---

## 🔭 Future Enhancements

- 🌍 Expand dataset diversity across ethnicities and age groups  
- 📱 Optimize for mobile and real-time deployment  
- 🧠 Integrate multi-modal learning with metadata (age, emotion, etc.)

---

## 📚 References

1. Schroff, Florian, Dmitry Kalenichenko, and James Philbin.  
   "FaceNet: A unified embedding for face recognition and clustering."  
   *IEEE CVPR*, 2015.  
