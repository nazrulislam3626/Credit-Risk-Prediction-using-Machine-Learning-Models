# Credit-Risk-Prediction-using-Machine-Learning-Models
# Credit Risk Prediction using Machine Learning Models

## Project Overview
This project focuses on predicting **credit risk (Good vs Bad borrowers)** using structured financial and demographic data. The objective is to evaluate and compare multiple **machine learning classification models** to identify the most reliable and stable model for credit risk assessment.

The project follows an **end-to-end applied machine learning workflow**, covering data preprocessing, exploratory data analysis, feature selection, model training, evaluation, and comparison.

---

## Business Context
Credit risk assessment is a critical function in banking and financial institutions. Incorrect classification can lead to:
- Financial losses due to loan defaults
- Reputational and regulatory risks
- Inefficient capital allocation

This project demonstrates how machine learning can support **data-driven credit decision-making**.

---

## Dataset
- **Source:** German Credit Risk Dataset  
- **Observations:** 1,000  
- **Features:** Demographic, financial, and behavioral variables  
- **Target Variable:** `Risk` (Good / Bad credit)

---

## Project Workflow

### 1. Data Loading & Preprocessing
- Loaded data from CSV format
- Removed irrelevant columns
- Handled missing values using **mode imputation**
- Checked and removed duplicate records
- Encoded categorical variables using Label Encoding
- Prepared clean dataset for modeling

---

### 2. Exploratory Data Analysis (EDA)
- Univariate, bivariate, and multivariate analysis
- Visualizations used:
  - Bar charts & pie charts
  - Scatter plots & violin plots
  - Correlation heatmaps
- Identified relationships between borrower attributes and credit risk

---

### 3. Feature Engineering & Selection
- Correlation-based feature importance analysis
- Heatmap visualization of feature relationships
- Attempted Recursive Feature Elimination (RFE) for dimensionality reduction
- Identified key predictors influencing credit risk

---

### 4. Machine Learning Models Implemented
The following models were trained and evaluated:

- Logistic Regression  
- Support Vector Machines (SVC, Linear SVC)  
- k-Nearest Neighbors (KNN)  
- Naive Bayes  
- Perceptron  
- Stochastic Gradient Descent (SGD)  
- Decision Tree Classifier  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- XGBoost  

---

### 5. Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Mean Squared Error (MSE)
- Training, Testing, and Validation accuracy
- Computational time comparison

---

## Results Summary
- Ensemble models (Random Forest, Gradient Boosting, XGBoost) showed strong performance
- Decision Tree models exhibited overfitting
- Logistic Regression and SVM provided a stable baseline performance
- Best validation accuracy achieved: **~74%**

---

## Key Learnings
- Ensemble methods outperform single classifiers on structured tabular data
- Model interpretability and stability are crucial in financial risk applications
- Accuracy alone is insufficient; precision and recall are essential for credit risk decisions

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Google Colab
