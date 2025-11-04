# CHURN-ANALYSIS

# 🚀 Customer Churn Prediction – Machine Learning Project

## 🧠 Overview
This project predicts **customer churn** (i.e., whether a customer is likely to leave a service) using **Logistic Regression** and modern **evaluation metrics** such as ROC-AUC, Precision, Recall, and F1-Score.

It demonstrates **end-to-end data science workflow** — from data preprocessing, feature engineering, and model training to evaluation and visualization.

---

## 🎯 Objectives
- Understand the key drivers of customer churn.  
- Build predictive models to classify customers who may leave.  
- Evaluate model performance using metrics beyond accuracy.  
- Derive actionable business insights for customer retention.

---

## 📊 Dataset Description
The dataset `customer_churn.csv` (or synthetic data) includes telecom customer information such as:

| Column | Description |
|:--|:--|
| `CustomerID` | Unique ID of the customer |
| `Gender` | Male / Female |
| `Tenure` | Duration of relationship with company |
| `InternetService` | Type of internet plan |
| `ContractType` | Month-to-month / One-year / Two-year |
| `TotalCharges` | Total amount billed |
| `Churn` | Target variable (Yes = churned, No = retained) |

---

## ⚙️ Technologies Used
| Category | Tools / Libraries |
|:--|:--|
| Language | Python 🐍 |
| IDE | Jupyter Notebook / VS Code |
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Model Saving | `joblib` |
| Explainability | `SHAP` (optional) |

---

## 🔍 Workflow

### 1️⃣ Data Preprocessing
- Convert data types and handle missing values.  
- Encode categorical features using One-Hot Encoding.  
- Scale numeric features using StandardScaler.  

### 2️⃣ Model Building
Implemented:
- **Logistic Regression** (Baseline Model)  
- **Random Forest & Gradient Boosting** (for comparison)

### 3️⃣ Model Evaluation
Metrics used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

```python
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))
