
# 🍷 Wine Quality Prediction using Machine Learning

## Overview
This project aims to predict the quality of wine based on its chemical properties using various machine learning algorithms. The pipeline includes data preprocessing, exploratory data analysis (EDA), model building, evaluation, and hyperparameter tuning.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Usage](#usage)
- [Author](#author)
- [References](#references)

## Project Overview
The project explores multiple ML algorithms to determine which performs best for wine quality prediction. The workflow includes:
1. Data Preprocessing
2. Exploratory Data Analysis
3. Model Training & Evaluation
4. Hyperparameter Tuning
5. Model Comparison

## Dataset
- **Source:** Wine_Quality.csv
- **Rows:** 1599
- **Columns:** 12 numeric features + 1 target (`quality`)
- **Target:** `quality` (score 0–10)
- **Notes:** ID column removed before processing.

## Libraries Used
- Python 3.x
- pandas, numpy, seaborn, matplotlib
- scikit-learn (model_selection, metrics, preprocessing, linear_model, neighbors, naive_bayes, tree, svm)

## Features
The dataset includes chemical properties like:
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (Target)

## Machine Learning Models
- **Linear Regression** – Predict continuous wine quality
- **Logistic Regression** – Classify wine quality categories
- **K-Nearest Neighbors (KNN)** – Distance-based classification
- **Naive Bayes (Gaussian, Multinomial, Bernoulli)** – Probabilistic classifiers
- **Decision Tree Regressor** – Feature importance and regression
- **Support Vector Machine (SVM)** – Classification with RBF kernel

## Hyperparameter Tuning
- Performed using `GridSearchCV` for:
  - Logistic Regression (`C`, `solver`, `penalty`)
  - KNN (`n_neighbors`, `weights`, `metric`)
  - Linear Regression (`fit_intercept`, `positive`)
- 5-fold cross-validation used

## Results
| Model | Metric | Score |
|-------|--------|-------|
| Logistic Regression | Accuracy | 0.XXX |
| KNN | Accuracy | 0.XXX |
| Linear Regression | R2 | 0.XXX |

- **Best Performer:** KNN/Logistic Regression (based on final evaluation)
- Demonstrates full ML pipeline from data preprocessing to model evaluation

## Future Enhancements
- Implement ensemble models like Random Forest, XGBoost
- Feature scaling and selection for improved accuracy
- Evaluate on larger, diverse datasets
- Deploy as web-based prediction app

## Usage
1. Clone the repository:
```bash
git clone https://github.com/mohdrahman32/Wine-Quality-Prediction-using-Machine-Learning.git
````
2. Run the Jupyter Notebook to explore the analysis and model results.

## Author

**Abdul Rahman Mohammed**
Email: [rahmanmohammed620@example.com](mailto:rahmanmohammed620@example.com)

## References

* Scikit-learn documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
* UCI Machine Learning Repository – Wine Quality Dataset
* Python Official Documentation: [https://docs.python.org/](https://docs.python.org/)

```
```
