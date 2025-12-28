# Loan Approval Risk Analysis using Machine Learning

## 📌 Problem Statement
The objective of this project is to analyze and predict loan approval probability based on applicant financial and credit attributes.  
The focus of the project is on **risk analysis and decision support**, rather than fully automated loan approval.

---

## 📊 Dataset
- Source: Loan Approval Dataset (public financial dataset)
- Records: 4,269 applicants
- Features: 12
- Target variable: Loan Status (Approved / Rejected)

The dataset includes applicant income, credit score, loan details, asset values, education, and employment status.

---

## 🛠️ Methodology

### 1. Data Understanding & Cleaning
- Identified and cleaned inconsistent column naming
- Removed identifier column (`loan_id`)
- Verified absence of missing values
- Encoded target variable for binary classification

### 2. Feature Engineering & Preprocessing
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- Used `ColumnTransformer` for structured preprocessing
- Implemented end-to-end preprocessing within a pipeline to prevent data leakage

### 3. Model Development
- Trained a **Logistic Regression** classifier
- Handled class imbalance using balanced class weights
- Selected Logistic Regression for interpretability and stability

### 4. Model Evaluation
- Performed train–test split with stratification
- Evaluated using:
  - Confusion Matrix
  - Precision, Recall, F1-score
  - ROC–AUC
- Conducted **5-fold stratified cross-validation**

---

## 📈 Results

- Test ROC–AUC: **~0.97**
- Cross-validation mean ROC–AUC: **~0.97**
- Cross-validation standard deviation: **~0.005**

The model demonstrated **excellent and stable generalization performance**.

---

## 🔍 Feature Importance & Insights
- Credit score (CIBIL) was the strongest predictor
- Applicant income and loan amount significantly influenced approval probability
- Asset values and loan term provided secondary influence
- Demographic features had comparatively lower impact

The learned patterns closely align with real-world financial decision criteria.

---

## ⚠️ Important Note on Deployment
This project focuses on **loan risk analysis and probability estimation**, not automated decision-making.

In real-world financial systems:
- Machine learning models are used for **risk scoring**
- Final decisions incorporate regulatory rules and business policies

Therefore, this project intentionally avoids public deployment to prevent misleading interpretations.

---

## 📁 Project Structure

loan_approval_ml/
│
├── data/
│   └── raw/
│       └── loan_data.csv
│
├── notebooks/
│   └── loan_approval.ipynb
│
├── models/
│   └── loan_approval_logistic_pipeline.pkl
│
├── .gitignore
└── README.md


---

## 🚀 Skills Demonstrated
- Binary classification
- Data preprocessing pipelines
- Handling class imbalance
- Model evaluation & cross-validation
- Feature importance interpretation
- Responsible ML decision framing

---

## 🧰 Tools & Libraries
- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib / Seaborn
- Git & GitHub
