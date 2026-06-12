# Rainfall Prediction Using Machine Learning

## Overview

This project focuses on predicting rainfall based on meteorological and environmental features using classical machine learning models. The goal is not only prediction but also understanding feature relationships, model behavior, and the impact of hyperparameter tuning.

The workflow includes:
* Exploratory data analysis and multicollinearity handling
* Data cleaning and preprocessing
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

The dataset contains meteorological features such as temperature, pressure, and other atmospheric variables, with a binary or categorical target:

### Target Variable
* rainfall = 1 → Rainfall occurs  
* rainfall = 0 → No rainfall  

---

## Project Structure

```text
rainfall-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── ProjectCRainfallPrediction.ipynb
│
├── models/
│   └── models.pkl
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
'''text

---

## Data Preprocessing & Rainfall Prediction Analysis

### Overview

This project focuses on building a machine learning pipeline to predict rainfall using meteorological and environmental features. The workflow includes data cleaning, feature engineering, model evaluation, hyperparameter tuning, and interpretability analysis.

The objective is to compare multiple models and identify the most reliable approach for rainfall prediction while balancing performance and interpretability.

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

This step reduced multicollinearity and improved generalization capability of models.

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

Splitting was randomized using a fixed seed for reproducibility. Stratification was not explicitly applied.

---

## Machine Learning Models Evaluated

Multiple models were tested using cross-validation.

### Models Evaluated

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

---

### Evaluation Method

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

* `n_estimators`
* `max_depth`
* `min_samples_leaf`

A manual grid search approach was used, and configurations were ranked based on mean CV F1-score.

### Best Performing Range

* `n_estimators`: 200–300
* `max_depth`: 3–10
* `min_samples_leaf`: 3–5

---

## Final Models

Two models were finalized:

### Logistic Regression

* Simple and interpretable baseline
* Stable across multiple data splits
* Useful for benchmarking performance

### Random Forest

* Captures non-linear relationships effectively
* Better feature interaction modeling
* Lower bias compared to linear models

---

## Model Performance

### Random Forest Results

* Training Accuracy: ~0.85
* Validation Accuracy: ~0.78–0.80
* Test Accuracy: ~0.80

### Logistic Regression Results

* Training Accuracy: ~0.83
* Validation Accuracy: ~0.78
* Test Accuracy: ~0.81

Both models demonstrated similar generalization capability.

---

## Feature Importance Analysis

Random Forest feature importance revealed:

* Strong influence from key environmental variables
* Some features contributed minimally and may be removable in future iterations
* Model relies heavily on non-linear feature interactions

This provides interpretability into meteorological drivers of rainfall prediction.

---

## Model Comparison

Models were compared using:

* Cross-validation F1 scores
* Train vs validation performance gap

### Key Observations

* Random Forest slightly outperformed Logistic Regression in stability
* Logistic Regression remained competitive due to simplicity and linear structure
* Ensemble methods better captured feature interactions

---

## Visualizations

The project includes several visual analysis components:

### Exploratory Data Analysis

* Rainfall class distribution
* Feature distributions
* Correlation heatmap

### Model Evaluation

* Cross-validation comparison
* Hyperparameter tuning results
* Train vs validation accuracy plots

### Interpretability

* Feature importance (Random Forest)
* Confusion matrices for both models

---

## Key Insights

* Dataset shows moderate multicollinearity requiring feature pruning
* Random Forest handles feature interactions more effectively than linear models
* Logistic Regression remains a strong baseline despite simplicity
* F1-score is more suitable than accuracy for evaluation
* Hyperparameter tuning improves performance marginally but consistently

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

Final trained models were saved using Pickle:

* Enables reuse without retraining
* Supports deployment in APIs or applications
* Ensures reproducible inference pipelines

---

## Future Improvements

* Feature selection using SHAP or mutual information
* Automated hyperparameter tuning (Optuna / GridSearchCV)
* Time-series rainfall forecasting models
* Integration with real-time weather APIs
* Deployment using FastAPI or Flask
* Deep learning-based forecasting models

---

## Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for rainfall prediction, covering preprocessing, modeling, evaluation, and interpretability.

Random Forest and Logistic Regression both performed competitively, but ensemble methods provided better robustness and feature interaction modeling.

Overall, the project balances predictive performance and interpretability, making it suitable for real-world meteorological decision systems.
