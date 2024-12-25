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

```math
\text{BMI} = \frac{\text{Weight (kg)}}{\text{Height (m)}^2}
```

### Preprocessing
- **Image Resizing**: Resized to 160x160 pixels.
- **Normalization**: Pixel values scaled between 0 and 1.
- **Data Augmentation**: Techniques such as flipping, rotation, and cropping.
- **Label Preparation**:
  - Underweight: BMI < 18.5
  - Normal weight: 18.5 <= BMI < 30
  - Overweight: BMI >= 30

## Methodology

### Feature Extraction
- Used **FaceNet** to extract a 512-dimensional embedding for each image.
- Combined embeddings from front and side views into a single 1024-dimensional feature vector.

### Model Selection
#### BMI Prediction (Regression)
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **XGBoost Regressor**

#### Gender Classification
- Logistic Regression for binary classification.

### Evaluation Metrics
#### Regression Metrics
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**
- **Pearson Correlation Coefficient**

#### Classification Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

## Results

### BMI Prediction
| Model                | R²   | MAE   | RMSE  | Pearson Coefficient |
|----------------------|-------|-------|-------|---------------------|
| Linear Regression    | 0.360 | 3.068 | 4.036 | 0.600              |
| Decision Tree        | 0.344 | 3.067 | 4.086 | 0.594              |
| Random Forest        | 0.464 | 2.874 | 3.694 | 0.718              |
| **XGBoost**          | **0.856** | **0.909** | **1.917** | **0.926**          |

### Gender Classification
- Logistic Regression achieved **95% accuracy**.

### Visualizations
- **Regression**: Scatter plot comparing predicted vs. actual BMI.
- **Classification**: Confusion matrix for gender prediction.

## Testing on New Data
- Pipeline tested on team members' images.
- Predicted BMI and gender compared against ground truth.

## Conclusion
The project demonstrates the viability of using facial images for BMI prediction and gender classification. Future work includes exploring advanced deep learning architectures and multi-view learning techniques.

## References
1. Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "FaceNet: A unified embedding for face recognition and clustering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
