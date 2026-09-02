import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load built-in Diabetes dataset
data = load_diabetes()

# Use one feature: BMI
X = data.data[:, [2]]

# Target: disease progression
y = data.target

# -------------------------
# Linear Regression
# -------------------------

linear_model = LinearRegression()
linear_model.fit(X, y)

y_linear = linear_model.predict(X)

# -------------------------
# Polynomial Regression
# -------------------------

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

y_poly = poly_model.predict(X_poly)

# -------------------------
# Evaluation
# -------------------------

linear_mse = mean_squared_error(y, y_linear)
linear_r2 = r2_score(y, y_linear)

poly_mse = mean_squared_error(y, y_poly)
poly_r2 = r2_score(y, y_poly)

# -------------------------
# Output
# -------------------------

print("LINEAR REGRESSION")
print("------------------")
print("MSE:", round(linear_mse, 2))
print("R2 Score:", round(linear_r2, 2))

print("\nPOLYNOMIAL REGRESSION")
print("---------------------")
print("MSE:", round(poly_mse, 2))
print("R2 Score:", round(poly_r2, 2))

# Prediction for a new BMI value
new_value = np.array([[0.05]])

linear_prediction = linear_model.predict(new_value)

new_value_poly = poly.transform(new_value)
poly_prediction = poly_model.predict(new_value_poly)

print("\nPrediction for New Input")
print("BMI:", new_value[0][0])
print("Linear Regression Prediction:",
      round(linear_prediction[0], 2))
print("Polynomial Regression Prediction:",
      round(poly_prediction[0], 2))
