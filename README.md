# Rainfall Prediction Using Machine Learning

## Overview

This project focuses on predicting rainfall based on meteorological and environmental features using classical machine learning models. The goal is not only prediction but also understanding feature relationships, model behavior, and the impact of hyperparameter tuning.

The workflow includes:
* Exploratory data analysis and multicollinearity handling
* Data cleaning and preprocessing
* Feature engineering and scaling
* Training multiple classification models
* Hyperparameter tuning for best performance
* Model evaluation using cross-validation and classification metrics
* Visual interpretability through plots and feature importance

---

## Business / Real-World Problem

Rainfall prediction is critical for:
* Agriculture planning and crop management
* Flood risk assessment and disaster preparedness
* Water resource allocation
* Climate monitoring systems

Accurate early prediction helps reduce economic loss and improves planning efficiency in weather-dependent sectors.

---

## Dataset

The dataset contains meteorological features such as temperature, pressure, humidity, and other atmospheric variables, with a binary target:

### Target Variable
* rainfall = 1 → Rainfall occurs  
* rainfall = 0 → No rainfall  

---

## Data Preprocessing

### Missing Value Handling

* Missing values were detected using `isna().sum()`
* All missing values were replaced with `0` for simplicity and consistency

This ensured the dataset had no null entries before model training.

---

## Feature Selection & Multicollinearity Handling

To improve model stability and reduce redundancy:

* Correlation matrix was computed using feature-level correlations
* Highly correlated features (|correlation| > 0.8) were identified
* Redundant features such as:
  * `maxtemp`
  * `mintemp`
  * `temperature`

were removed from the dataset

This step reduced multicollinearity and improved model generalization.

---

## Feature Engineering

* Dataset was split into features (X) and target variable (y)
* Standard scaling was applied using `StandardScaler`
* Scaling was performed after train-test splitting to avoid data leakage

This ensured proper normalization while preserving evaluation integrity.

---

## Train-Test Split Strategy

The dataset was split into:

* 60% Training set
* 20% Validation set
* 20% Test set

Splitting was randomized using a fixed seed for reproducibility.

---

## Machine Learning Models Evaluated

### Models Tested

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

---

## Model Evaluation Method

* 5-Fold Cross Validation
* Primary Metric: F1 Score (used due to class imbalance considerations)

---

## Model Selection

Based on cross-validation results:

* Logistic Regression → Strong baseline performance
* Random Forest → Best overall performance
* XGBoost → Competitive but not selected as final model

Final selection prioritized validation stability and F1-score consistency.

---

## Hyperparameter Tuning

### Random Forest Tuning

Parameters explored:

* n_estimators
* max_depth
* min_samples_leaf

A manual grid-search style approach was used.

Best performing range:
* n_estimators: 200–300
* max_depth: 3–10
* min_samples_leaf: 3–5

---

### Logistic Regression Tuning

* Parameter tuned: C (regularization strength)
* Best value: C ≈ 10

---

## Final Models

### Logistic Regression
* Simple and interpretable baseline
* Stable across multiple splits
* Good linear benchmark

### Random Forest
* Captures non-linear relationships
* Strong feature interaction modeling
* Better robustness than linear models

---

## Model Performance

### Random Forest
* Training Accuracy: ~0.85
* Validation Accuracy: ~0.78–0.80
* Test Accuracy: ~0.80

### Logistic Regression
* Training Accuracy: ~0.83
* Validation Accuracy: ~0.78
* Test Accuracy: ~0.81

Both models showed similar generalization performance.

---

## Feature Importance Analysis

Random Forest feature importance showed:

* Strong influence from key environmental variables
* Some features had low contribution and could be removed later
* Non-linear feature interactions were important

---

## Model Comparison

Models were compared using:
* Cross-validation F1 scores
* Train vs validation performance gap

Key observations:
* Random Forest slightly outperformed Logistic Regression in stability
* Logistic Regression remained competitive due to simplicity
* Ensemble methods better captured complex interactions

---

## Visualizations

### Exploratory Data Analysis
* Rainfall class distribution
* Feature distributions
* Correlation heatmap

### Model Evaluation
* Cross-validation comparison
* Hyperparameter tuning curves
* Train vs validation accuracy plots

### Interpretability
* Feature importance (Random Forest)
* Confusion matrices

---

## Key Insights

* Moderate multicollinearity required feature pruning
* Random Forest handles feature interactions better than linear models
* Logistic Regression remains a strong baseline
* F1-score is more appropriate than accuracy
* Hyperparameter tuning improves performance slightly but consistently

---

## Technologies Used

### Programming Language
* Python

### Data Handling
* Pandas
* NumPy

### Visualization
* Matplotlib
* Seaborn

### Machine Learning
* Scikit-learn
* XGBoost

### Evaluation
* Cross-validation
* Confusion matrix
* Classification metrics

### Model Persistence
* Pickle

---

## Model Saving & Deployment

Final models were saved using Pickle:
* Enables reuse without retraining
* Supports deployment in APIs
* Ensures reproducible inference pipelines

---

## Future Improvements

* Feature selection using SHAP or mutual information
* Automated hyperparameter tuning (Optuna / GridSearchCV)
* Time-series rainfall forecasting models
* Real-time weather API integration
* FastAPI/Flask deployment
* Deep learning-based forecasting

---

## Conclusion

This project demonstrates an end-to-end machine learning pipeline for rainfall prediction, covering preprocessing, modeling, evaluation, and interpretability.

Random Forest and Logistic Regression both performed competitively, but ensemble methods provided better robustness and feature interaction capability.

Overall, the system balances predictive performance with interpretability, making it suitable for real-world meteorological decision-making.
