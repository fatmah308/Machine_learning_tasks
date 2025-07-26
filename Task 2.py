#importing libraries for mathematical operations and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing machine learning libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

#fetching dataset
data = pd.read_csv('loan_approval_dataset.csv')

#strip extra space before each columns name
data.columns = data.columns.str.strip()

#drop 'loan_id' as it is just an identifier 
data.drop('loan_id', axis=1, inplace=True)

#encoding all non-numeric data 
le = LabelEncoder()
data['education'] = le.fit_transform(data['education'])       
data['self_employed'] = le.fit_transform(data['self_employed']) 
data['loan_status'] = le.fit_transform(data['loan_status'])     

#scale data for logistic regression 
X = data.drop('loan_status', axis=1)
y = data['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#plotting to see the imbalance between loan statues values
sns.countplot(x='loan_status', data=data)  
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()

#correlation heatmap
corr = data.corr()
sns.heatmap(corr, annot=True)
plt.show()

#splitting my data into train test model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#using SMOTE rto balance classes(this is part of a bonus task)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

#testing new class distribution
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_balanced.value_counts())

#using logistic regression
# Train model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_balanced, y_train_balanced)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
print("Logistic Regression Report:\n")
print(classification_report(y_test, y_pred_lr, target_names=["Approved", "Rejected"]))

# Confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr), display_labels=["Approved", "Rejected"]).plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

#using decision tree
# Train model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_balanced, y_train_balanced)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate
print("Decision Tree Report:\n")
print(classification_report(y_test, y_pred_dt, target_names=["Approved", "Rejected"]))

# Confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt), display_labels=["Approved", "Rejected"]).plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()
