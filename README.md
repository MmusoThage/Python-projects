# 🧠 Breast Cancer Classification Using Support Vector Machine (SVM)


This project uses a Breast Cancer Dataset to build a machine learning model that classifies tumors as benign or malignant. The model is trained using a Support Vector Machine (SVM), a powerful algorithm for binary classification tasks.

## 🧬 Dataset

Source: kaggle: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

Features: 30 numeric features derived from images of fine needle aspirates (FNAs) of breast masses

Target: Diagnosis (M = malignant, B = benign)




## Technologies Used ⚙️

Python 🐍

Pandas & NumPy

Scikit-learn

Matplotlib & Seaborn

Jupyter Notebook




## 📊 Key Steps
Data Preprocessing

Loaded data and handled missing values

Converted categorical target labels to binary (0/1)

Standardized features using StandardScaler

Exploratory Data Analysis (EDA)

Visualized feature distributions

Correlation heatmap to detect multicollinearity

Model Training

Split data into training and test sets

Trained an SVM classifier with grid search for hyperparameter tuning

Model Evaluation

Evaluated using accuracy, precision, recall, F1 score, and ROC-AUC

Plotted confusion matrix and ROC curve




## 📈 Results

Accuracy: ~98%

Precision (Malignant): High (minimal false positives)

Recall (Malignant): High (minimal false negatives)


## 📌 Future Work

Add cross-validation for model robustness



# 💳 Loan Default Classification Using Logistic Regression and Decision Trees
This project uses a loan approval dataset to predict whether a borrower will default or not. Two models are trained — Logistic Regression and Decision Tree Classifier — to assess credit risk based on borrower characteristics.

## 📂 Dataset
Source: Kaggle
Loan Approval Prediction Dataset

Features:
Income, CIBIL score, asset values, loan term, employment status, etc.

Target:
Loan Status (0 = Approved, 1 = Rejected/Defaulted)

## ⚙️ Technologies Used
Python 🐍

Pandas & NumPy

Scikit-learn

Seaborn & Matplotlib

Jupyter Notebook

## 📊 Key Steps
Data Preprocessing
Loaded and cleaned the dataset

Encoded categorical features

Created new features (loan-to-income, asset-to-loan ratios)

Scaled features for logistic regression

Exploratory Data Analysis (EDA)
Boxplots and pairplots to compare defaulters vs non-defaulters

Identified predictive features like CIBIL score and income

Model Training
Split data into training and test sets

Trained Decision Tree (max depth = 4) and Logistic Regression models

Model Evaluation
Accuracy, precision, recall, F1-score

Confusion matrices and classification reports

Visual comparison of model performance

## 📈 Results
Decision Tree Accuracy: ~99.5%

Logistic Regression Accuracy: ~92%

CIBIL Score and Asset Ratios were key predictors

Over-borrowing trends identified via loan-to-income ratio

## 📌 Future Work
Add k-fold cross-validation

Tune hyperparameters using GridSearchCV

Integrate into a risk assessment dashboard or loan approval system using Flask or Streamlit

Experiment with other classifiers (Random Forest, XGBoost)

Deploy using Flask or Streamlit for a web app interface


