#!/usr/bin/env python
# coding: utf-8

# In[5]:


import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Unzipping the provided dataset (make sure the file path is correct on your local system)
zip_file_path = r"C:\Users\Snigdha\Downloads\Lead+Scoring+Case+Study.zip"
output_dir = r"C:\Users\Snigdha\Downloads\Lead_Scoring_Case_Study"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

# Loading the dataset
assignment_folder = os.path.join(output_dir, 'Lead Scoring Assignment')
leads_data_path = os.path.join(assignment_folder, 'Leads.csv')
leads_df = pd.read_csv(leads_data_path)

# Data Cleaning
columns_to_drop = ['Lead Quality', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
                   'Asymmetrique Activity Score', 'Asymmetrique Profile Score']
leads_df_cleaned = leads_df.drop(columns=columns_to_drop)
leads_df_cleaned['Country'].fillna('Unknown', inplace=True)
leads_df_cleaned['Specialization'].fillna('Unknown', inplace=True)
leads_df_cleaned['How did you hear about X Education'].fillna('Unknown', inplace=True)
leads_df_cleaned['What is your current occupation'].fillna('Unknown', inplace=True)
leads_df_cleaned['What matters most to you in choosing a course'].fillna('Unknown', inplace=True)
leads_df_cleaned['City'].fillna('Unknown', inplace=True)
leads_df_cleaned['TotalVisits'].fillna(leads_df_cleaned['TotalVisits'].mean(), inplace=True)
leads_df_cleaned['Page Views Per Visit'].fillna(leads_df_cleaned['Page Views Per Visit'].mean(), inplace=True)
leads_df_cleaned['Lead Source'].fillna(leads_df_cleaned['Lead Source'].mode()[0], inplace=True)
leads_df_cleaned['Last Activity'].fillna(leads_df_cleaned['Last Activity'].mode()[0], inplace=True)
leads_df_cleaned['Tags'].fillna('Unknown', inplace=True)
leads_df_cleaned['Lead Profile'].fillna('Unknown', inplace=True)

# Encoding categorical variables
categorical_columns = leads_df_cleaned.select_dtypes(include=['object']).columns
categorical_columns = [col for col in categorical_columns if col not in ['Prospect ID', 'Lead Number']]
leads_df_encoded = pd.get_dummies(leads_df_cleaned, columns=categorical_columns, drop_first=True)

# Splitting the dataset into training and testing sets
X = leads_df_encoded.drop(['Converted', 'Prospect ID', 'Lead Number'], axis=1)
y = leads_df_encoded['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building the logistic regression model with increased max_iter
log_reg = LogisticRegression(max_iter=2000, solver='lbfgs')  # You can also try solver='liblinear' or 'saga'
log_reg.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = log_reg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1])
conf_matrix = confusion_matrix(y_test, y_pred)

# Results
evaluation_metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "ROC-AUC Score": roc_auc,
    "Confusion Matrix": conf_matrix
}

# Print the evaluation metrics
print(evaluation_metrics)


# In[14]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns

# 1. Confusion Matrix Visualization using seaborn
y_pred = log_reg.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()  # Display confusion matrix

# 2. Feature Importance Plot (Logistic Regression Coefficients)
feature_importance = np.abs(log_reg.coef_[0])
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 important features
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances")
plt.show()  # Display feature importance

# 3. ROC Curve
y_score = log_reg.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()  # Display ROC curve


# In[ ]:




