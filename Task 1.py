import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv('StudentPerformanceFactors.csv')

#basic structure of the dataset
print(data.head())
print(data.info())
print(data.describe())

#visualize the original data
plt.figure(figsize=(8,6))
plt.scatter(data['Hours_Studied'], data['Exam_Score'], color='purple')
plt.title('Study Time vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.grid(True)
plt.show()

#correlation between study hours and score
correlation = data['Hours_Studied'].corr(data['Exam_Score'])
print("Correlation:", correlation)

#splitting data into train and test splits
X = data[['Hours_Studied']]
y = data['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_linear = lr_model.predict(X_test)

#evaluation of linear regression model
print("🔹 Linear Regression Performance:")
print("R² Score:", r2_score(y_test, y_pred_linear))
print("MAE:", mean_absolute_error(y_test, y_pred_linear))
mse = mean_squared_error(y_test, y_pred_linear)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

#visualization of linear regression
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='purple', label='Actual')
plt.plot(X_test, y_pred_linear, color='orange', linewidth=2, label='Linear Fit')
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()

#bonus task
# Transform features to polynomial
poly = PolynomialFeatures(degree=2)  
X_poly = poly.fit_transform(X)

# Split the polynomial-transformed data
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Fit polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)

# Predict using polynomial model
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate performance
print("\n🔹 Polynomial Regression Performance:")
print("R² Score:", r2_score(y_test_poly, y_pred_poly))
print("MAE:", mean_absolute_error(y_test_poly, y_pred_poly))
mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
print("RMSE:", rmse_poly)

#visulaization of polynomial regression
# Create smooth X range for curve
X_range = pd.DataFrame({'Hours_Studied': np.linspace(1, 44, 200)})
X_range_poly = poly.transform(X_range)

# Predict using the polynomial model
y_range_pred = poly_model.predict(X_range_poly)

# Plot curve
plt.figure(figsize=(8,6))
plt.scatter(data['Hours_Studied'], data['Exam_Score'], color='purple', alpha=0.5, label='Actual')
plt.plot(X_range, y_range_pred, color='green', linewidth=2, label='Polynomial Fit')
plt.title('Polynomial Regression Fit')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()

#comparison of both models
print("\n🔍 Model Comparison Summary")
print(f"Linear Regression R²: {r2_score(y_test, y_pred_linear):.4f}, RMSE: {rmse:.4f}")
print(f"Polynomial Regression R²: {r2_score(y_test_poly, y_pred_poly):.4f}, RMSE: {rmse_poly:.4f}")




