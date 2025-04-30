# 🧠 Breast Cancer Classification Using Support Vector Machine (SVM)


This project uses a Breast Cancer Dataset to build a machine learning model that classifies tumors as benign or malignant. The model is trained using a Support Vector Machine (SVM), a powerful algorithm for binary classification tasks.

##🧬 Dataset

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

Experiment with other classifiers (Random Forest, XGBoost)

Deploy using Flask or Streamlit for a web app interface


